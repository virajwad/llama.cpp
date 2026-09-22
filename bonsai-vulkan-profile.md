# Bonsai Q1_0 Vulkan profiling - Intel Arc Pro B70

Local investigation, 2026-09-22. This is a profiling report, not a submission description. The baseline investigation below made no production changes. The subsequent experiments and their validation status are recorded here separately.

## Final decision - reject the dedicated shader against the 10% target

The final bounded cycle is complete. After screening all 16 row/workgroup combinations, the best common setting was **four output rows and a 64-invocation workgroup**, using full 16-lane subgroups. Fresh uninstrumented ABBA measurements gave:

| Workload | Original tokens/s | Tuned v2 tokens/s | Throughput gain | Criterion |
|---|---:|---:|---:|---|
| TG128, empty starting context | 52.618 | 53.679 | +2.016% | FAIL (>10% required) |
| TG128, 16384-token starting context | 45.439 | 46.647 | +2.658% | FAIL (>10% required) |
| PP512 | 1033.513 | 1033.881 | +0.036% | Effectively unchanged |

Each path has ten samples per workload (five per process, two processes per path). Both context depths used the same tuned setting. Means average the two process means; pooled sample standard deviations for baseline/candidate were 0.164/0.194 t/s at short context and 0.118/0.068 t/s at 16K. Clocks/power were not locked, so these are observations on this system, not universal hardware guarantees. No performance logger or pipeline-statistics flags were enabled for the acceptance runs.

The user rebuilt v2 and reported **43/43 backend tests passed**. No test output or tuning-environment details were supplied. This does not independently certify all 16 settings, the selected override, or model-level PPL/KLD. Because performance already fails the acceptance threshold, this cycle stops without requesting further numerical validation or designing a third kernel.

**Recommendation:** discard the experimental production changes rather than retain the shader for a roughly 2-3% gain. No source rollback or default-policy change was made during evaluation; changes remain available for review. The compiled unset-default candidate is slower than the original (50.05 versus approximately 52.7 t/s in TG64 screening), so do not leave the patch enabled as-is for regular use. `GGML_VK_DISABLE_Q1_0_DECODE=1` selects the original path without rebuilding. The 4-row/64-thread overrides were process-local to benchmark runs and were not installed as defaults.

No AMD/NVIDIA performance claim is made. Their runtime dispatch and all existing shader sources remain unchanged, including both protected attention shaders. This result rejects this implementation/tuning cycle, not the possibility of a different binary-weight algorithm outperforming it.

### Screening results (selection only, not acceptance evidence)

TG64 with three repeats per configuration, fixed shuffled order, and an original-path check after every four configurations. Baseline checks ranged from 52.45 to 52.85 t/s.

| Rows | WG16 | WG32 | WG64 | WG128 |
|---|---:|---:|---:|---:|
| 2 | 49.05 | 50.36 | 49.65 | 45.69 |
| 4 | 49.88 | 52.95 | 54.47 | 51.88 |
| 8 | 42.18 | 50.67 | 53.75 | 53.90 |
| 16 | 30.45 | 43.17 | 51.44 | 51.55 |

The two strongest settings were screened at 16K with TG128: rows4/WG64 gave 46.97 t/s, rows8/WG128 gave 46.87, with baseline brackets 45.66/45.58. The fresh ABBA results above supersede these screening numbers. No setting demonstrated a 10% improvement even during screening.

The best setting's compiler statistics were 3034 instructions, 201 basic blocks, nine loops, zero scratch bytes, and 1024 workgroup-memory bytes. Both dispatch slots specialize to WG64 under that override. The original variants report 3019/3133 instructions and zero scratch. These static counts do not establish dynamic instruction throughput, occupancy, or a DRAM bottleneck.

Every v2 process exited zero and used backend SHA256 `d43738dfcd0d622d9c9b31c36fed088ee655975d56f3ab247bd2583082cbd1fe`. The [complete v2 summary](build-bonsai-profile/v2-summary.csv) and raw manifests preserve the evidence.

## Candidate v1 - rejected on performance

The user built the first dedicated sign-XOR shader and reported that the Q1_0 MUL_MAT CPU-reference checker passed. No checker output was supplied, so this is a user-reported result, not an independently inspected test log. Runtime pipeline statistics confirmed that the new pipeline was selected on B70.

Balanced original/candidate/candidate/original runs used the same rebuilt DLL, TG128, five repetitions per configuration, FP16 KV and FA on. Means below average the two process means per path:

| Starting context | Original tokens/s | Candidate v1 tokens/s | Change |
|---|---:|---:|---:|
| Empty | 52.71 | 48.75 | -7.5% |
| 16384 tokens | 45.35 | 42.72 | -5.8% |

Driver instruction counts were 3019/3133 for the original subgroup/hybrid variants and 3020/3170 for the candidate, with zero scratch for both. The candidate increased basic-block counts. Serialized profiling found regressions in the dominant FFN shapes, despite an output-head improvement. This version does not meet either acceptance condition.

Raw paired runs: [baseline A](build-bonsai-profile/v1-ab-baseline-a.json), [candidate A](build-bonsai-profile/v1-ab-candidate-a.json), [candidate B](build-bonsai-profile/v1-ab-candidate-b.json), [baseline B](build-bonsai-profile/v1-ab-baseline-b.json). The respective manifests retain command lines, environment and DLL/source hashes.

## Candidate v2 - evaluated implementation

- Revised [mul_mat_vec_q1_0_decode.comp](ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_q1_0_decode.comp) to precompute four FP32 pair sums and differences for each eight activations and reuse them across output rows. Each sign byte selects and signs those pairs; four contributions are summed and accumulated with the block scale using FP32 FMA. No activation quantization or GGUF change.
- Removed the activation-load alignment branch: N=1 and K divisible by 128 give four-float-aligned offsets with the existing contiguous/copy-to-contiguous dispatch. Reuses the existing batch/broadcast offsets, row-tail handling, bias fusion, and subgroup/hybrid reductions.
- Uses `GL_KHR_shader_subgroup_basic` and `GL_KHR_shader_subgroup_arithmetic` through the shared base, plus existing 8/16-bit storage extensions. The host requests full 16-lane subgroups. No new vendor-specific shader extension is required.
- B70-only runtime controls allow one build to cover rows 2/4/8/16 (`GGML_VK_Q1_0_DECODE_ROWS`) and workgroup sizes 16/32/64/128 (`GGML_VK_Q1_0_DECODE_WG`). Unset defaults are eight rows and the existing shape-based choice of workgroup 16 or 64. Invalid override values retain these defaults. The shared-memory reduction variant is used whenever the workgroup exceeds 16.
- Selected only for Intel Arc Pro B70 device `0xe223`, with required subgroup features, Q1_0 weights, FP32 activations, one activation column and K divisible by 128. Other GPUs, quant types, FP16 activations, multi-column matvecs, matmul and expert-ID matvecs retain their existing paths. A single-column output projection at the end of prefill can also select this candidate.
- A/B switch: set `GGML_VK_DISABLE_Q1_0_DECODE=1` before process launch for the original path; unset for the candidate. Any defined value disables it, including `0`.
- All existing GLSL files, both merged attention shaders and the B70 core-count lookup remain untouched.
- Pre-build checks: no reported editor diagnostics; whitespace check passed; 3072 host FP32 pair/sign checks, 40 column-partition cases across all workgroup choices, and 36 row-coverage cases including tails and z-split output-head dispatch passed. These checks do not validate compiled shader execution or model quality. FMA and summation-order changes still require backend tolerance and numerical model checks.
- The user-built v2 DLL is dated 2026-09-22 15:28:43; generated shader output is dated 15:28:36. Pipeline statistics verified the dedicated candidate and original paths in that same DLL. The [runner](build-bonsai-profile/run_bench.py) rejects a DLL older than the candidate sources, records source hashes, and refuses to overwrite an existing run name.

### Bounded acceptance protocol

1. User builds v2 once. Confirm generated shader/binary freshness and candidate pipeline selection. Obtain and inspect CPU-reference checker results for the revised shader; v1's reported pass does not cover v2.
2. Screen the 16 row/workgroup combinations with short-context decode, with baseline brackets. Short screening runs select candidates only; they are not acceptance evidence.
3. Check the best two settings at 16K. Use one setting that works at both contexts, not separately cherry-picked settings.
4. Revalidate the selected setting, including fusion and tail/broadcast cases, then run fresh uninstrumented ABBA comparisons at both contexts with at least five repetitions per leg. Compare the same DLL with and without the disable switch. Keep K/V types, FA, batch, CPU threads, model and device fixed.
5. Accept only a repeatable throughput gain greater than 10% at both contexts, with satisfactory correctness and no prefill regression. Results within run-to-run noise of the threshold are inconclusive, not a pass. If this cycle misses the threshold, recommend discarding the experimental production changes; do not claim success or start a third design cycle.

New artifacts are outside the build directory under the ignored sibling profiling directory. The original baseline artifacts were removed during the user's clean rebuild; the historical tables below remain. The protocol's selected-configuration numerical revalidation was not independently completed; the implementation is rejected on performance regardless.

## Initial profiling summary (before the kernel experiments)

- Prioritize a Q1_0-specific decode matvec experiment, not changes to the two newly merged attention shaders. Q1_0 matvecs account for 74.3% of short-context serialized GPU timings; flash attention accounts for 2.2%.
- Preserve cooperative-matrix prefill. Disabling KHR cooperative matrices reduced PP512 throughput from approximately 1,040 to 256 tokens/s.
- An immediate local tuning result: for PP2048, increasing the microbatch from 512 to 2048 improved throughput from 942.8 to 996.4 tokens/s, about 5.7%, without shader changes. This is a performance result, not a numerical-equivalence certification.
- B70 device ID `0xe223` is missing from the Intel shader-core-count lookup. Investigate this separately; no speedup from changing the lookup has been measured.
- A portable binary kernel was worth investigating; the subsequent bounded experiment above demonstrated only a small gain, below the requested threshold. AMD/NVIDIA non-regression has not been measured.

## Environment and model

- Source and Release binary: commit `4ceb17191`, build 11111, MSVC 19.51.36256.0. This includes the Intel decode shader merge.
- GPU: Intel Arc Pro B70, device ID `0xe223`, driver 101.8993 (Windows driver 32.0.101.8993). Only this GPU is exposed by the Vulkan loader.
- Vulkan reports FP16, integer dot product, KHR cooperative matrices, 49,152 bytes of workgroup memory, default subgroup size 32, and supported subgroup sizes 16-32. The matvec dispatch explicitly requests 16 on this device; the startup banner's 32 is not every kernel's subgroup size.
- CPU: Core Ultra 7 270K Plus. Benchmarks used 8 CPU threads, one GPU, all 65/65 layers offloaded, FP16 K/V, batch 2048, microbatch 512 unless noted, and flash attention enabled unless noted.
- Loader allocation: Vulkan model buffer 3,446.26 MiB, CPU-mapped buffer 170.51 MiB. The mapped buffer matches the embedding table size; the repeating and output layers are offloaded. No model-weight VRAM capacity problem was observed.
- GGUF: `qwen35`, 26,895,998,464 parameters, 64 layers, embedding 5120, FFN 17408, vocabulary 248320. Full attention every fourth layer: 16 full-attention layers and 48 linear-attention/Gated DeltaNet layers. Full attention has 24 query heads, 4 KV heads, head dimensions 256, GQA ratio 6.
- Tensor mix: 498 Q1_0 tensors (26,893,352,960 elements, 3,781,877,760 bytes) and 353 F32 tensors (2,645,504 elements, 10,582,016 bytes). Both embedding and output weights are Q1_0.
- Q1_0 stores 128 binary signs plus an FP16 scale in 18 bytes: 1.125 bits/weight including scale. Values are `+d` or `-d`, not ternary. See [the block definition](ggml/src/ggml-common.h#L180-L186). The original metadata-dump artifact was lost in the clean rebuild.

## Measured inference throughput

Uninstrumented runs, mean +/- sample standard deviation in tokens/s. Five repeats for the baseline, three for the other entries. Warmup enabled. Between-run variation is larger than some within-run deviations; clocks and power were not locked.

| Workload | Configuration | Tokens/s |
|---|---|---:|
| PP512 | Default test configuration | 1040.23 +/- 1.34 |
| TG128, empty starting context | Default test configuration | 53.11 +/- 0.06 |
| TG128, 4096-token starting context | FA on | 50.48 +/- 0.06 |
| TG128, 4096-token starting context | FA off | 50.47 +/- 0.04 |
| TG128, 16384-token starting context | FA on | 45.65 +/- 0.17 |
| TG128, 16384-token starting context | FA off | 44.81 +/- 0.03 |
| PP512 | KHR cooperative matrices disabled | 255.89 +/- 0.81 |
| TG128, empty starting context | KHR cooperative matrices disabled | 52.47 +/- 0.05 |
| PP512 | Fusion disabled | 1003.03 +/- 0.88 |
| TG128, empty starting context | Fusion disabled | 51.69 +/- 0.06 |

The separate initial baseline was PP512 1029.43 and TG128 52.45. Treat the short-context baseline as approximately 1,030-1,040 PP and 52.5-53.1 TG, not a fixed hardware constant.

At short context, a same-invocation FA on/off comparison gave PP512 1030.78/1057.76 and TG128 52.53/52.57. This is not evidence to disable FA globally: the 16K-context comparison favors FA, and this comparison does not isolate the newly merged shaders from all other attention differences. Likewise, disabling cooperative matrices is a whole-backend ablation, not a decode-shader-only ablation.

Prefill microbatch sweep, with `n_batch=2048`:

| Prompt tokens | Microbatch | Tokens/s |
|---|---:|---:|
| 512 | 128 | 556.43 |
| 512 | 256 | 948.78 |
| 512 | 512 | 1031.18 |
| 2048 | 128 | 528.12 |
| 2048 | 256 | 879.73 |
| 2048 | 512 | 947.75 |

A second, same-invocation PP2048 sweep gave 942.80 at microbatch 512, 986.97 at 1024, and 996.38 at 2048. Larger microbatches were not applied as a global default. Recheck memory and latency under concurrent serving.

These are llama-bench random-token evaluation workloads. They exclude sampling, tokenization, server scheduling, and model-loading time; they are not real-chat TTFT or quality tests.

## GPU operation attribution

The following fractions use **serialized** `GGML_VK_PERF_LOGGER` timings with normal fusion enabled. These timings insert synchronization and change performance; they are useful for prioritization, not uninstrumented throughput predictions.

| Workload | Q1_0 matmul/matvec | Flash attention | Gated DeltaNet |
|---|---:|---:|---:|
| Short-context decode | 74.33% | 2.21% | 1.30% |
| PP512 | 75.54% | 3.66% | 7.65% |
| Decode at 16K context | 65.95% | 12.95% | 1.12% |

Short-context timing includes one warmup plus 16 generated tokens; PP includes warmup plus one measured 512-token pass. The 16K table uses only the final 16 decode timing blocks, excluding context preparation and warmup. The long-context KV dimension was padded to 16640.

Largest short-context entries, dimensions expressed as output rows `M`, activation columns `N`, contraction `K`:

| Operation | Mean per invocation | Share of recorded GPU time |
|---|---:|---:|
| FFN gate/up, M=17408, N=1, K=5120 | 53.38 us | 31.20% combined |
| FFN down plus fused add, M=5120, N=1, K=17408 | 49.16 us | 14.37% |
| Linear-attention QKV, M=10240, N=1, K=5120 | 34.97 us | 7.66% |
| Output head, M=248320, N=1, K=5120 | 817.46 us | 3.73% |
| Small alpha/beta projections, M=48, N=1, K=5120 | 5.69 us | 2.49% combined |
| Full-attention operation, 24/4 heads, D=256, padded KV=256 | 30.27 us | 2.21% |

For the two FFN matrix shapes, 12,533,760 weight bytes divided by the observed kernel times gives approximately 235 and 255 GB/s. The output head gives approximately 219 GB/s. These are **effective unique-weight byte rates**, not hardware DRAM-counter measurements. They suggest room to investigate unpack/issue efficiency, load layout, and scheduling, but do not prove a specific bottleneck or a guaranteed speedup.

Compiler-reported statistics for the two compiled Q1_0 matvec variants: 3019/3133 instructions, 134/150 basic blocks, zero scratch bytes, and 0/1024 workgroup-memory bytes. This is static pipeline information, not dynamic instruction counts or measured occupancy. There is no evidence here for a register-spill explanation.

The concurrent timing mode was also tried. Its short-context table did not include the final output-head matvec, whereas serialized timing did. Do not use that concurrent table as a complete per-op accounting. `GGML_VK_PERF_LOGGER_FREQUENCY` controls printing/aggregation frequency, not every-Nth-operation sampling. Instrumentation overhead is not assumed to be a fixed percentage.

## Code gaps and candidate experiments

### 1. Q1_0-specific floating-activation matvec - first kernel experiment

The current [generic matvec inner loop](ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp#L46-L119) calls `dequantize4` twice per eight activations, forms floating signs, calculates two dot products, and applies the block scale to that partial sum. [Q1_0 unpacking](ggml/src/ggml-vulkan/vulkan-shaders/dequant_funcs.glsl#L129-L144) extracts the two nibbles from the same sign byte. A compiler may already eliminate duplicate loads; disassembly is required before calling them redundant memory transactions.

Intel [dispatch](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2670-L2735) gives Q1_0 four output rows, subgroup size 16, and one- or four-subgroup variants. The [workgroup heuristic](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5352-L5376) selects the larger variant for `M <= 8192` and `K >= 1024`, not according to Q1_0's own unpacking cost.

Candidate: keep the existing format and fusion interface, but prototype an eight-sign decode primitive with explicit single-byte extraction and sign-bit selection/XOR on floating activations. Sweep rows per workgroup, elements per lane, and reduction structure for Q1_0 only. Preserve FP32 accumulation and existing bias/scale handling. Do not assume a sign-XOR spelling beats the driver's existing lowering.

A second candidate is a tiled floating-activation lookup table. For four input activations, precompute the 16 signed sums; each weight nibble selects one sum. Accumulate 32 selected sums per 128-weight block and apply that block's scale. This avoids activation quantization, but changes summation order. Use enough output-row reuse to amortize table construction, and tile K: a full K=5120 FP32 table would require 80 KiB, exceeding this device's 48 KiB workgroup limit. A K=128 tile needs 2 KiB. Subgroup shuffles versus shared-memory gathers require separate measurement across vendors.

Do not use plain XNOR/popcount: the activations are floating point, not binary. Q1_0, IQ1_S, IQ1_M, and ternary types are not interchangeable.

**Potential, not a forecast:** applying Amdahl's law to the serialized short-context share, a 1.5x Q1_0 speedup implies approximately 1.33x overall GPU-time speedup; a 2x Q1_0 speedup implies approximately 1.59x. Host overhead, changed overlap, and long-context attention reduce the applicability of those estimates. No such kernel speedup was measured.

### 2. Integer-dot path - real gap, separate numerical tradeoff

Q1_0 is absent from the [Q8_1 matvec whitelist](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5291-L5318), although the device supports integer dot products. It is mathematically possible to expand binary signs into packed int8 lanes in registers and dot them against quantized activations. The gap is implementation/dispatch support, not an impossibility of integer arithmetic for binary weights.

This requires activation quantization and therefore a different accuracy/performance tradeoff. Compare fused activation quantization with a shared/reused quantization pass, especially for Q/K/V and FFN gate/up. Avoid expanding the entire model to int8 in device memory; that would sacrifice the weight-bandwidth advantage. Keep this experiment separate from the floating-activation path and require PPL/KLD validation.

### 3. Preserve and tune prefill separately

The [current Q1_0 matmul loader](ggml/src/ggml-vulkan/vulkan-shaders/mul_mm_funcs.glsl#L428-L442) already reads one byte, reconstructs eight signed scaled values, and stages them for the matrix path. It does not lack Q1_0 support, and it does not perform a separate full-matrix dequantization just because weights are Q1_0.

The paper's FP16 sign-reconstruction approach is conceptually close to this existing code. Inspect the generated ISA and tile occupancy before replacing it. Q1_0-specific tile/staging and scale-load amortization are candidates; replacing all prefill with a scalar binary shader is contradicted by the cooperative-matrix ablation. Prioritize the M=17408/K=5120 and M=5120/K=17408 FFN shapes, which jointly account for 49.3% of recorded PP512 GPU time.

### 4. Missing B70 core-count lookup - small, isolated host-side candidate

The [Intel device map](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L15954-L15996) contains B50/B570/B580/B60 but not this device's `0xe223`. It returns zero. The supplied paper identifies B70 as 32 Xe2 cores.

Consequences visible in code: [matmul split-K](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L5638-L5680) requires a nonzero count; [generic attention](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7972-L8025) substitutes 16; other selectors also consume the field. This does **not** establish a fixed occupancy loss or prove that the new dual-phase decode is underfilled. Its actual launch geometry is computed separately after selection. Verify the SKU mapping and A/B benchmark the lookup change across small and large shapes and other models before retaining it.

### 5. Keep the new Intel attention shaders intact

Both `xe_fa_decode_ph1` and `xe_fa_decode_ph2` were compiled in the runtime pipeline-statistics run. Their reported scratch usage is zero; shared memory is 16384 and 16576 bytes respectively. GQA ratio 6 and head dimension 256 do reach these pipelines for this FP16-KV workload. They are not Q1_0-weight kernels.

The [selection logic](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L7940-L8060) checks Intel architecture, layout, FP16 KV/mask, and split conditions. [Creation](ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2994-L3016) uses a minimum phase-1 workgroup of 64, not 16. No change was made to either [phase 1](ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_1.comp) or [phase 2](ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_decode_phase_2.comp).

At 16K context, attention reaches approximately 13% of serialized GPU time, so it deserves a later long-context investigation. Any such experiment must preserve the existing fast path, use shape/feature-based selection rather than a model-name check, and retain the merged shader performance on the broader model matrix.

## How the supplied paper applies

Reviewed the supplied local v3 of **Pushing the Envelope of LLM Inference with Ultra-Low-Bit Quantized Models**, Georganas, Kalamkar, Heinecke, and Dubey ([arXiv 2508.06753](https://arxiv.org/abs/2508.06753)). Relevant sections:

- Section II-D, Figure 2, pages 4-5: fused activation quantization, VNNI16 weight layout, rectangular tiles that reuse quantized activations, native Xe2 int2 x int8 DPAS, and split-K/autotuning.
- Figure 3 and following discussion, page 5: reconstruct signed FP16 scales and use FP16 matrix operations, avoiding activation quantization. This is the closest starting point for preserving current model numerics.
- Section III-C, pages 7-8: compare kernels against an attainable memory/compute roofline, include scale-generation overhead, and distinguish GEMV from compute-bound GEMM. The paper reports up to roughly 500 GB/s for selected B70 GEMVs, not a universal rate for every shape.
- Section III-F, Figure 21, page 10: approximately 47.9 tokens/s for **ternary/2-bit** Bonsai-27B on B70 in vLLM. This is not an apples-to-apples comparison against this binary Q1_0 GGUF and llama-bench setup.

Native hardware int2 DPAS availability does not imply that the same packed-int2 operand is expressible through the currently used Vulkan KHR cooperative-matrix interface. Confirm supported component types and extension/compiler support before planning a direct port. A portable FP16 reconstruction path and a packed-int8 integer-dot path are distinct alternatives.

## Validation gate before any production change

1. Reuse the existing backend-op test infrastructure for Q1_0 `MUL_MAT`, fused matvec results, and CPU reference comparisons. Cover actual M/K shapes above, N=1..8, the matvec/matmul crossover, and N=128/256/512/2048. Include tails, strides, batch/broadcast behavior, extreme scales, all sign-byte patterns, and zero inputs.
2. For attention changes, test D=64/128/256, GQA=1/2/4/6/8/16, small multi-token decode and long KV, masks and sinks, and FP16 plus quantized-KV fallbacks. Protect models for which the merge supplied a speedup.
3. Run PPL/KLD and deterministic-logit comparisons against the unmodified build, especially for activation-quantized or reassociated-sum kernels. A successful performance run is not a correctness test.
4. Use matched baseline/candidate builds with interleaved repeated runs, identical model/context/cache settings, steady clocks, and no simultaneous GPU work. Require no reproducible regression beyond measured noise, not just a higher best run.
5. Test Intel B70 plus another Intel generation, AMD wave32/wave64 as applicable, and NVIDIA. Keep other types and vendor dispatch paths unchanged. Initially enable a candidate only on configurations validated to improve; retain the old path for everything else. Portable shader source does not guarantee portable speedup.
6. Preserve all existing fusion semantics. Existing fusions measurably matter here. Check that unrelated shader variants remain unchanged, and measure pipeline creation as well as steady-state inference when adding variants.

**Validation limitations:** the user reports 43/43 backend tests passed after building v2, without supplying raw output or override settings. No independently captured CTest result, exhaustive tuned-setting validation, PPL/KLD, hardware-counter/ISA capture, or AMD/NVIDIA measurements are available. The VS Code CMake integration could not run tests because no kit was selected. All manifest-recorded benchmark processes exited zero. That is performance-run completion, not an additional correctness certification.

Related existing work found during the issue/PR search: [Q1_0 unpacking on HIP, draft #28398](https://github.com/ggml-org/llama.cpp/pull/28398), [Q1_0 integer-dot discussion #27127](https://github.com/ggml-org/llama.cpp/issues/27127), and [Vulkan vendor-scoped matvec tuning #27909](https://github.com/ggml-org/llama.cpp/pull/27909). These are useful context, not evidence that their measured gains transfer to Vulkan/B70. No posts or submissions were made.

## Reproducibility artifacts

- [Complete v2 summary](build-bonsai-profile/v2-summary.csv).
- Final decode ABBA runs: [baseline A](build-bonsai-profile/v2-final-baseline-a.json), [candidate A](build-bonsai-profile/v2-final-candidate-a.json), [candidate B](build-bonsai-profile/v2-final-candidate-b.json), [baseline B](build-bonsai-profile/v2-final-baseline-b.json). Each contains both starting context depths and individual samples.
- Example reproduction manifest: [candidate A](build-bonsai-profile/v2-final-candidate-a.manifest.json), with exact command, process-local environment and DLL/source hashes.
- Prefill ABBA runs: [baseline A](build-bonsai-profile/v2-prefill-baseline-a.json), [candidate A](build-bonsai-profile/v2-prefill-candidate-a.json), [candidate B](build-bonsai-profile/v2-prefill-candidate-b.json), [baseline B](build-bonsai-profile/v2-prefill-baseline-b.json).
- Driver statistics: [best tuned setting](build-bonsai-profile/v2-best-stats.log), [default candidate](build-bonsai-profile/v2-default-stats.log), [original path](build-bonsai-profile/v2-baseline-stats.log).
- [Local benchmark runner](build-bonsai-profile/run_bench.py) writes separate raw stdout/stderr and a command/environment/exit-code manifest for each invocation. It launches only the supplied model with the existing Release executable, and rejects stale binaries and reused output names.
- The **Bonsai Vulkan baseline (PP512 TG128)** task in [.vscode/tasks.json](.vscode/tasks.json) points to the surviving runner and explicitly disables the candidate. Change the run name before repeating if its output already exists. The profiling artifacts and task are ignored by git in this checkout; this report is a separate local document.
- Both protected attention hashes remain unchanged: phase 1 `482619FC4B3CE33517D2D936D554C338F548AAAB7666A473D228F7E718B76590`; phase 2 `10D4572F34679904983C49FC52F71B8AC322AA6D6FDFD5A9862306FDF4569A17`.

Important reproduction detail: repeated llama-bench test parameters append configurations rather than necessarily overriding them. The FA on/off and microbatch tables above were read from each JSON row's actual settings, not inferred from output order or run name. The first PowerShell-redirection run also produced UTF-16 files and a misleading shell error status from native stderr; use the raw-output runner for subsequent runs.