# Qwen3 seamless PD V5 final validation

Canonical evidence root: `/root/autodl-tmp/experiments/qwen3_multimodel_v5_defaults_20260819`; relative evidence paths below resolve against this directory.

Date: 2026-08-19; final lazy-profile update: 2026-08-20
Device: Meizu 21
Runner SHA256:
- original V5 validation runner: `92cefb162b1dc2d16fac33b3de4f479c2c26ed3bc78980bb6953faedbd93e2a3`
- dated-release dispatch-logging runner: `2570609daeb97f2bff6b58f0ffdc0d35e4b216b71e17339b65a81444d094daff`
- monitor-lifecycle repair runner: `c453ee1ad045457b6a40f06abf7f30863dbe15eb75fbb31b0b79d91b141a56d5`
- final 14B lazy-profile runner: `22e063b1e10b552fe0fa4dd92ac568e5eabd2b7400947c59d5c9aa53c168bfb8`

## Validated lifecycle

- One in-process joint runner; QNN Prefill hands its in-memory U8 KV directly
  to the staged llama.cpp Decode runtime without process restart or a
  resource-wide Prefill teardown.
- The 4B/8B runtime is initialized before active Prefill. The 14B default
  stages the model and V5 transport before Prefill, then parses `.meta` and
  creates the TG context at handoff; it never performs a synthetic warmup
  between Prefill and Decode.
- Prefill uses the stage-major three-stage pipeline. Every shard detaches its
  QNN execution shell at Load tail, releases rebuilt PTE backing, and uses the
  two-buffer rebuild pool plus the elastic two-slot KV pool.
- V5 Decode metadata is one `.meta`; payload is one `.bin`. Shard payloads
  are asynchronously read into stable per-shard malloc buffers while QNN
  Prefill executes. Decode waits only for unfinished reads at the boundary.
- A16 graph I/O is preserved. Decode linears use the default dynamic-A8,
  GS32 I8MM/DOTPROD path where the model and CPU support it.

## Common test configuration

- tokenized prompt: 1024 tokens, 1023 cached Prefill tokens
- AR128, context 4096
- 32 generated tokens, temperature 0; repaired 8B validation uses TG64
- six Decode threads, `taskset fc`
- 5 ms process-tree memory sampler through Prefill only; it is stopped and
  joined before synchronous joint Decode
- no mincore residency scan and no per-operator probe

## Phone results

| Model | Shards | V5 payload | QNN Prefill | Prefill speed | Decode speed | First Decode token | Post-first mean | First-token excess vs post-first mean | PD boundary | Peak RSS | Peak PSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B (monitor fix, runs 1/2) | 18 | 141,010,816 B | 2088.986 / 1651.620 ms | 489.711 / 619.392 tok/s | 16.630 / 16.340 tok/s | 61.274 / 78.059 ms | 16.650 / 16.497 tok/s | +1.213 ms (+2.0%) / +17.441 ms (+28.8%) | 6.889 / 4.722 ms | 2670.05 / 2668.95 MiB | 2440.46 / 2440.74 MiB |
| Qwen3-8B (monitor fix, TG64) | 18 | 249,692,032 B | 4279.691 ms | 239.036 tok/s | 8.600 tok/s | 140.717 ms | 8.626 tok/s | +24.786 ms (+21.4%) | 11.946 ms | 4197.61 MiB | 3767.14 MiB |
| Qwen3-14B (lazy profile, TG32) | 20 | 459,454,080 B | 11133.783 ms | 91.970 tok/s | 4.819 tok/s (5.270 tok/s excluding `.meta` load) | 961.388 ms | 5.516 tok/s | +780.089 ms (+430.3%) | 49.293 ms | 6816.84 MiB | 6113.06 MiB |

All dated-release validation runs exited zero and produced the same coherent continuation
beginning ` jumps over the lazy dog. Modern graphics processors...`.
The logs explicitly report `pipeline_enabled=1`, dynamic A8 scope `all`,
resident joint Decode ready before the boundary, and V5 sidecar completion.

The monitor-lifecycle repair was retested without graph/operator probes.
4B sidecar boundary waits were 0.272 and 0.086 ms; repaired 8B TG64 waited
1.340 ms; the current 14B lazy-profile run waited 47.068 ms for sidecar
completion and measured a 49.293 ms full PD boundary. The retained 8B TG32
row is the pre-repair historical control. Repaired 8B measured 239.036
Prefill / 8.600 Decode tok/s, a 140.717 ms first token, 8.626 tok/s post-first
mean, 11.946 ms PD boundary and 4197.61 MiB HWM.

Both 4B runs were stable at 16.630 and 16.340 tok/s (16.485 mean), 84.8%
above the pre-repair 8.92 tok/s row. The current 14B run uses the post-Prefill
lazy quant-profile lifecycle: its 5.270 tok/s core rate becomes 4.819 tok/s
when deferred metadata/context preparation and the handoff are included. Its
961.388 ms first token intentionally includes that deferred work; HWM fell to
6816.84 MiB with zero process VmSwap. All current rows exited zero and selected
`I8MM_NATIVE_16ROW_EMBEDDED`. The TG32 4B/14B runs emitted SHA256
`4c384fda8a340ef424784c43f085b078745319e93cb19332be649f436003d864`;
the longer TG64 8B run emitted
`3008182fe206c569d3ff3e4351630a9d19099de5e54707062c3007eee6f438a8`.
All retained model/embedding/manifest/sidecar assets were preserved.

Peak RSS/HWM remains a kernel process metric across the complete run. Peak PSS
in the repaired rows is sampled through Prefill only because the perturbing
5 ms sampler is deliberately joined before Decode.

## Zero-transfer 4B/8B repetitions

The two 4B rows in this subsection predate the GGUF correction and used the
old GS32 layout, so Decode selected `GS32_GROUPWISE_FALLBACK`. They are kept
as historical measurements and are not the current native-I8MM result.

The retained phone workdirs were executed again without fetching or pushing
any runner, sidecar, Prefill bundle, embedding, or model.

| Model | Run | Decode core | First token | Post-first mean | First-token excess | Post-first median | QNN Prefill | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B | 1 | 6.89 tok/s | 181.473 ms | 6.777 tok/s | +33.921 ms (+23.0%) | 131.622 ms | 3269.818 ms | 2663.70 MiB |
| 4B | 2 | 8.08 tok/s | 184.955 ms | 8.210 tok/s | +63.159 ms (+51.9%) | 120.777 ms | 1756.900 ms | 2668.62 MiB |
| 8B | 1 | 5.51 tok/s | 214.806 ms | 5.540 tok/s | +34.293 ms (+19.0%) | 179.864 ms | 6317.067 ms | 4388.07 MiB |
| 8B | 2 | 5.14 tok/s | 207.433 ms | 5.150 tok/s | +13.257 ms (+6.8%) | 192.531 ms | 4469.095 ms | 4390.12 MiB |

Both models used six generation and batch threads and reported dynamic-A8
scope all. The stable first-token pairs and opposite throughput movement
(4B improved while 8B declined) reject a fixed thread-count or A8-dispatch
regression. The old approximately 13 tok/s 4B result used prompt/depth 32 and
TG128; these runs use prompt/depth 1024 and TG32. Earlier same-depth 4B was
5.30-6.25 tok/s. Earlier 8B prompt1024 samples ranged from 4.90 to 9.19 tok/s,
so the new 5.14-5.51 samples are low but remain inside the documented device
state range.

## Unified PD and independent-Decode comparison

This is intentionally one table. It places historical joint PD, current V5
joint PD, independent GPTQ2/A8 Decode, and stock llama.cpp Q2_K/Q4_0 beside
each other so that weak points remain visible. `--` means that no valid formal
measurement exists; it is not filled using another model or an extrapolation.
The condition columns expose differences in depth, TG length, and thread count
instead of using those differences to hide an unfavorable result.

| Model | Path / version | Joint PD? | Prompt or prepared depth | TG | Threads | QNN Prefill | Decode | Result status / exposed weakness |
|---|---|:---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | historical GPTQ2/U16 PD, 2026-08-02 | yes | 1024 | 128 | 6 | 1057.1 / 1215.2 tok/s | 27.29 / 25.21 tok/s; aggregate **26.21** | Valid two-run historical joint result |
| Qwen3-1.7B | **current V5** | yes | 1024 | 32 | 6 | **--** | **--** | No valid no-output Prefill bundle; current model-size coverage is incomplete |
| Qwen3-1.7B | independent GPTQ2/A16 | no | d0 / d32 | 128 | 6 | -- | 23.725 / 23.024 tok/s | Old activation path |
| Qwen3-1.7B | independent GPTQ2/A8 I8MM | no | d0 / d32 | 128 | 6 | -- | **38.843 / 36.458 tok/s** | Fastest retained 1.7B native Decode result |
| Qwen3-4B | original GPTQ2/U16 PD, 2026-08-05 | yes | 1024 | 128 | 6 | **450.35 tok/s** | **8.50 tok/s** | Old U16 result; Decode kernel timing was 8.52 tok/s |
| Qwen3-4B | pre-V5 GPTQ2/A8 single-GGUF PD, 2026-08-11 | yes | 1024 | 128 | 6 | about 387.4 tok/s | **6.25 tok/s** median | Decode samples 6.68 / 5.82; large device-state variance |
| Qwen3-4B | **V5 pre-correction GS32 fallback / zero-transfer repeat** | yes | 1024 | 32 | 6 | **312.861 / 582.276 tok/s** | **6.892 / 8.08 tok/s** | Historical asset error: native source was absent, so both runs fell back |
| Qwen3-4B | **current V5 + corrected native-I8MM GGUF, pre-monitor-fix 2026-08-19** | yes | 1024 | 32 | 6 | **609.714 tok/s** | **8.92 tok/s** | Actual `I8MM_NATIVE_16ROW_EMBEDDED`; Decode still overlapped the 5 ms process monitor |
| Qwen3-4B | **current monitor-fixed seamless PD, 2026-08-20** | yes | 1024 | 32 | 6 | **489.711 / 619.392 tok/s** | **16.630 / 16.340 tok/s** | 16.485 mean; first token 61.274 / 78.059 ms; stable two-run repair result |
| Qwen3-4B | independent GPTQ2/A8 I8MM, latest depth curve | no | d0 / d32 | 128 | 6 | -- | **17.561 / 16.528 tok/s** | No formal d1024 GPTQ2-only result, so the PD gap is not fully isolated |
| Qwen3-4B | independent Q2_K, matched | no | d1024 | 32 | 6 | -- | **7.155 tok/s** | Monitor-fixed PD two-run mean is 130.4% faster |
| Qwen3-4B | independent Q4_0, matched | no | d1024 | 32 | 6 | -- | **10.729 tok/s** | Monitor-fixed PD two-run mean is 53.6% faster |
| Qwen3-8B | fixed-export GPTQ2/A8 PD, 2026-08-13 | yes | 1024 | 64 | 6 | **196.96 tok/s** median | **7.97 tok/s** median | Decode range 4.90-9.19 tok/s shows poor stability |
| Qwen3-8B | **current V5 run 1 / zero-transfer repeat** | yes | 1024 | 32 | 6 | **161.942 / 228.905 tok/s** | **5.507 / 5.14 tok/s** | 31-36% below the old 7.97 median; current regression is exposed |
| Qwen3-8B | **current dated native-I8MM release, pre-monitor-fix 2026-08-19** | yes | 1024 | 32 | 6 | **186.423 tok/s** | **4.89 tok/s** | Actual `I8MM_NATIVE_16ROW_EMBEDDED`; Decode still overlapped the 5 ms process monitor |
| Qwen3-8B | **current monitor-fixed seamless PD, 2026-08-20** | yes | 1024 | 64 | 6 | **239.036 tok/s** | **8.60 tok/s** | First token 140.717 ms; post-first 8.626 tok/s; HWM 4197.61 MiB |
| Qwen3-8B | independent GPTQ2/A8 | no | d1024 | -- | 6 | -- | **--** | Missing formal independent benchmark prevents exact PD-overhead isolation |
| Qwen3-8B | independent Q2_K | no | d1024 | 64 | 6 | -- | **4.216 tok/s** | Matched monitor-fixed PD is 104.0% faster |
| Qwen3-8B | independent Q4_0 | no | d1024 | 64 | 6 | -- | **6.872 tok/s** | Matched monitor-fixed PD is 25.1% faster |
| Qwen3-14B | probe-free pre-V5 GPTQ2/A8 PD, 2026-08-18 | yes | 1024 | 32 | 6 | **68.998 tok/s** | **3.46 tok/s** | Superseded boundary-init lifecycle, but valid arithmetic speed record |
| Qwen3-14B | V5 sidecar precursor | yes | 1024 | 4 | 6 | **77.11 tok/s** | **3.22 tok/s** | Short TG only |
| Qwen3-14B | **V5 pre-correction GS32 model path** | yes | 1024 | 32 | 6 | **119.053 tok/s** | **2.418 tok/s** | Historical script selected the old GS32 file; native dispatch was not demonstrated |
| Qwen3-14B | **current dated native-I8MM release run 1 / rested run 2, pre-monitor-fix 2026-08-19** | yes | 1024 | 32 | 6 | **84.030 / 115.203 tok/s** | **1.16 / 2.48 tok/s** | Both actual I8MM; monitor still overlapped Decode; run 1 also had a 9140.472 ms residency outlier |
| Qwen3-14B | **current monitor-fixed seamless PD, 2026-08-20** | yes | 1024 | 32 | 6 | **108.168 / 81.021 / 98.312 tok/s** | **2.790 / 5.250 / 2.530 tok/s** | 2.790 median; best 5.250 is not stable; run 3 first token 2406.123 ms exposes residual residency sensitivity |
| Qwen3-14B | **post-Prefill lazy quant profile, original parser, 2026-08-20** | yes | 1024 | 32 | 6 | **91.97 tok/s** | **4.819 tok/s** | Full PD Decode rate includes deferred profile/context and handoff; first token 961.388 ms; HWM 6816.84 MiB; VmSwap 0 MiB |
| Qwen3-14B | independent GPTQ2/A8 | no | d1024 | -- | 6 | -- | **--** | Missing formal independent benchmark prevents exact PD-overhead isolation |
| Qwen3-14B | independent Q2_K | no | d1024 | 32 | 6 | -- | **1.978 tok/s** | Monitor-fixed PD median is 41.1% faster; best is 165.4% faster but unstable |
| Qwen3-14B | independent Q4_0 | no | d1024 | 32 | 6 | -- | **0.0864 tok/s** | Not a kernel result: 8.51 GB working set repeatedly faults under RAM pressure |

A matched 4B rerun used the current llama.cpp Android build with `taskset fc`,
six threads, depth 1024, TG32, two measured repetitions, normal warmup, F16
KV, and no per-token or memory probe. Q4_0 reached 10.729 tok/s (10.7568,
10.7013); Q2_K reached 7.155 tok/s (7.26184, 7.04881). Both exited zero.

The 14B Q4_0 result is an end-to-end memory-residency failure case: its
8.51 GB model working set exceeded the available physical-memory budget and
repeatedly faulted mmap pages during autoregressive Decode. It must not be
used as a Q4_0 kernel-throughput result. Historical llama-bench output stores
aggregate throughput only, so first-token latency is unavailable for these
rows. Sources are
`experiments/qwen3_4b_llamacpp_quant_bench_20260805/REPORT.md`,
`experiments/qwen3_8b_pd_18shards_20260813/results/llamacpp_q2k_q40_phone/`,
and `experiments/qwen3_14b_1024_bench_20260816/RESULT.md`.

## I8MM dispatch observability follow-up

The GPTQ2/A8 dispatch point now emits one process-level line naming the
actual selected branch: `I8MM_NATIVE_16ROW_EMBEDDED`,
`I8MM_NATIVE_16ROW_SIDECAR`, or `GS32_GROUPWISE_FALLBACK`. The same line
records the runtime I8MM+DOTPROD hardware gate, native-weight source, first
projection, and matrix shape. This is instrumentation only; no arithmetic,
weight, scheduling, or activation behavior changed.

The Android PD joint runner rebuilt successfully and its stripped diagnostic
copy has SHA256
`2570609daeb97f2bff6b58f0ffdc0d35e4b216b71e17339b65a81444d094daff`.
Three diagnostic reruns exposed a model-asset error rather than a CPU gate
error: `i8mm_dotprod=1`, but the retained phone GGUF logged
`GS32_GROUPWISE_FALLBACK embedded_native=0 sidecar_native=0`. Its SHA256 was
`a11e525911d5dfb7543617490970767b5f30cca083950a87d941220d9dd88f2d`,
identical to the old GS32 host asset. The three core Decode results were
7.671, 7.567, and 7.580 tok/s (mean 7.606 tok/s).

The phone asset was then replaced by the previously validated same-size
`i8mm_native_v1` GGUF (1,782,500,320 bytes, SHA256
`30ad1f3f860919c22a13355dc1c0afd638fa317173abcb75e3cf4461e5b52882`;
252/252 GPTQ2 tensor roundtrips passed). A real seamless-PD rerun logged
`I8MM_NATIVE_16ROW_EMBEDDED i8mm_dotprod=1 embedded_native=1
sidecar_native=0`. At prompt/depth 1024 and TG32, QNN Prefill took
1679.476 ms (609.714 tok/s), core Decode reached 8.92 tok/s, generation took
3600.201 ms for 32 tokens, and the first Decode token took 147.165 ms. Output
SHA256 remained
`4c384fda8a340ef424784c43f085b078745319e93cb19332be649f436003d864`,
identical to the fallback runs. Evidence is in
`results/4b_native_i8mm_verify_20260819/`.

## Device-asset retention

The initial 4B and 8B tests reused their already installed GGUF and embedding
files; only the newly exported sidecars, Prefill bundle, runner and backend
were sent. The dispatch audit subsequently replaced the incorrect persistent
4B GS32 GGUF with the validated same-size native-I8MM GGUF. After validation,
the complete 4B persistent asset directory (GGUF plus embedding) was moved to
the dated version directory
`/data/local/tmp/pd/models/Qwen3-4B/2026-08-19-v5-native-i8mm/`. The legacy
path no longer exists. The retained GGUF and embedding SHA256 values are
`30ad1f3f...b52882` and `b6d2d871...c916d`.

The verified 8B fixed native GGUF and embedding were moved to
`/data/local/tmp/pd/models/Qwen3-8B/2026-08-19-v5-native-i8mm/`; their full
SHA256 values are
`75425e447ded9e2447944d0ae05861ea54a7eaab1dbd214dda4d9c30ef2c142c`
and `7281c37c89695034aa28002be91168262ac63b56412246e764c70baa49c21bfb`.
The verified 14B native GGUF and embedding were moved to
`/data/local/tmp/pd/models/Qwen3-14B/2026-08-19-v5-native-i8mm/`; their SHA256
values are
`2783a53fe1b863a043309ab5d83ae65c196f18b832cafb8a3ae622825b228412`
and `86cedc2870b571702b6882a38ccef6ea7bdaca7175841e577f14c4e07cb497a5`.
No additional 8B or 14B model was deleted: the superseded 8B non-fixed
native file and 14B
GS32 file remain outside the dated release directories, while all default
scripts now select only the verified dated versions.

The matched llama.cpp rerun retained `Qwen3-4B-Q2_K.gguf` and
`Qwen3-4B-Q4_0.gguf` under phone session directory
`/data/local/tmp/adb-hub/lNNKQjv0IHrX1fXd4ofRyr_N`. Their phone SHA256
values matched the host originals (`285e880e...` and `1e5bdf0e...`).

Qwen3-1.7B is intentionally excluded at user direction. Its GGUF exists, but
the available matching QNN export still contains the Prefill logits graph; the
locally staged 459 MiB bundle is not a valid no-output deployment and was
never pushed to the phone.

## Evidence

- `results/4b/4b.stderr`, `4b.stdout`
- `llamacpp_4b_exact_20260819/llamacpp_4b_exact_results.tar.gz`
- `llamacpp_4b_exact_20260819/results/*.json` and `*.stderr`
- `results/8b/8b.stderr`, `8b.stdout`
- `results/14b/14b.stderr`, `14b.stdout`
- `results/14b/14b-retained-assets-before.txt`
- `results/14b/14b-retained-assets-after.txt`
- `results/4b_native_i8mm_verify_20260819/4b.stderr`, `4b.stdout`, and the original result tar
- `results/native_release_8b_14b_20260819/` (8B run and both 14B runs)
- `results/phone_4b_archive_move_20260819_ledger.json` (4B/8B/14B moves and reruns)
- ADB Hub reports under `results/phone_*_response.json`
- `../qwen3_8b_pd_ctx_barrier_20260819/results/multimodel_monitor_fix_4b14b/`
  (two repaired 4B runs and three repaired 14B runs)
- `../qwen3_8b_pd_ctx_barrier_20260819/results/multimodel_monitor_fix_4b14b_ledger.json`

Repository revisions:

- llama.cpp: `53e01a622` (`feat: defer 14B decode quant profile until handoff`)
- ExecuTorch: `74c3c0c4a7` (`feat: default 14B to lazy decode quant profile`)
