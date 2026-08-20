# PD Stack Layout

The current final 4B/8B/14B phone-result record is [RESULTS.md](RESULTS.md).

This PD split inference stack is logically one project, even though it is
currently implemented across two git repositories:

- `executorch/`
  - Qualcomm prefill export
  - PTE rebuild / quant attrs / PD handoff generation
  - Android and x86 PD runner entrypoints
- `T-MAC/`
  - GGUF model runtime
  - `llama-pd-cli` decode/import side
  - llama.cpp-based native comparison and roundtrip diagnostics

## Current Logical Boundary

The boundary between the two repos is:

1. ExecuTorch runs prefill and exports a PD handoff directory.
2. T-MAC imports that handoff and resumes decode from GGUF state.

The handoff payload contains:

- prompt tokens
- first token
- canonicalized KV cache
- manifest / metadata

## Recommended "Big Repo" Mental Model

Treat the workspace like this:

```text
pd-stack/
├── executorch/
│   └── examples/qualcomm/oss_scripts/llama/
├── T-MAC/
│   └── 3rdparty/llama.cpp/
└── models/
    ├── *.pte
    └── *.gguf
```

This is the recommended setup for now because:

- each upstream repo keeps its own git history clean
- local experimentation stays easy
- ExecuTorch can still explicitly reference T-MAC paths
- future migration to a true super-repo or git submodule remains possible

## Recommended Integration Contract

Instead of hardcoding absolute paths, use environment variables:

- `EXECUTORCH_ROOT`
- `TMAC_ROOT`
- `TMAC_LLAMA_CPP_ROOT`
- `TMAC_LLAMA_PD_CLI_X86`
- `TMAC_LLAMA_PD_CLI_ANDROID`

See `pd_stack_env.sh` in the same directory for a helper that exports them.

## ExecuTorch-side Tools

For the stripped-PTE workflow, use the canonical helper under:

- `examples/qualcomm/oss_scripts/llama/tools/pte_qat_checkpoint_reverse_strip.py`

It rebuilds quantized blocks from the QAT checkpoint, strips them from a source
`.pte`, and writes:

- `stripped.pte`
- `index.json`
- `index.bin`
- `report.txt`

Usage notes and ready-to-run examples are documented in:

- `examples/qualcomm/oss_scripts/llama/tools/README.md`

## Suggested Ownership Split

Keep code ownership roughly like this:

- `executorch/.../runner/pd_runner.cpp`
  - prefill execution
  - KV export
  - quant attrs loading
  - PD handoff serialization
- `executorch/.../qnn_llama_pd*_runner.cpp`
  - orchestration
  - device/host entrypoints
  - path passing into T-MAC
- `T-MAC/3rdparty/llama.cpp/examples/pd-cli/pd_cli.cpp`
  - PD import
  - GGUF resume decode
  - native compare diagnostics

## If You Later Want a True Single Repository

The cleanest migration path is not to copy code manually. Use a super-repo:

```text
pd-stack/
├── executorch/        # submodule or subtree
└── external/
    └── T-MAC/         # submodule or subtree
```

Recommended order of escalation:

1. Current sibling-repo layout with shared env vars
2. Add `T-MAC` as a git submodule under `executorch/external/T-MAC`
3. Only if necessary, convert to a true monorepo

For your current PD debugging, step 1 is the best cost/performance point.

## Recorded llama.cpp Decode Baselines

The following Meizu 21 CPU results were recovered from the retained raw
`llama-bench` JSON on 2026-08-19. Historical output contains aggregate
throughput only, not first-token latency.

| Model | Quantization | Decode | Conditions | Repetitions |
|---|---|---:|---|---:|
| Qwen3-4B | Q2_K | 7.772 tok/s | TG128, depth 0, 4 threads | 5 |
| Qwen3-4B | Q4_0 | 16.252 tok/s | TG128, depth 0, 4 threads | 5 |
| Qwen3-4B (matched) | Q2_K | **7.155 tok/s** | TG32, depth 1024, 6 threads, `taskset fc` | 2 |
| Qwen3-4B (matched) | Q4_0 | **10.729 tok/s** | TG32, depth 1024, 6 threads, `taskset fc` | 2 |
| Qwen3-8B | Q2_K | 4.216 tok/s | TG64, depth 1024, 6 threads | 3 |
| Qwen3-8B | Q4_0 | 6.872 tok/s | TG64, depth 1024, 6 threads | 3 |
| Qwen3-14B | Q2_K | 1.978 tok/s | TG32, depth 1024, 6 threads | 2 |
| Qwen3-14B | Q4_0 | 0.0864 tok/s | TG32, depth 1024, 6 threads | 2 |

The historical 4B rows are not directly comparable with the depth-1024 PD
runs. A matched rerun used the current llama.cpp Android build with
`taskset fc`, six threads, depth 1024, TG32, two measured repetitions, normal
warmup, F16 KV, and no probes. Q4_0 reached 10.729 tok/s (10.7568, 10.7013);
Q2_K reached 7.155 tok/s (7.26184, 7.04881). Both exited zero.

The 14B Q4_0 result is an end-to-end residency failure case: the 8.51 GB
model working set exceeded available physical memory and repeatedly faulted
mmap pages. It must not be interpreted as Q4_0 kernel throughput.

## Current Default Configuration and Complete Phone Results

The accepted production path is one in-process seamless QNN Prefill to
llama.cpp Decode lifecycle. Code defaults are context 4096, six Decode
threads, temperature 0, no layer offload, V5 sidecar overlap enabled, and
decode_n_predict=128. The measurements below deliberately override only TG
to 32 for 4B/14B and 64 for 8B. All use a 1024-token prompt/depth, AR128,
taskset fc, stage-major three-stage Prefill, detached QNN execution shells,
external embeddings, no Prefill output graph, no Decode operator probes, and
the memory monitor stopped and joined before synchronous Decode.

The tri-state lazy-profile default is deliberately conditional. `-1` enables
post-Prefill `.meta` parsing for a 40-layer 14B run only when the single-request
V5 sidecar-overlap lifecycle is active; unsupported modes automatically cancel
lazy loading and preload the profile eagerly. `0` always uses eager preload,
while explicit `1` requires the supported lifecycle and fails instead of
silently changing the requested policy.

Decode eval is the inner llama.cpp evaluation rate. Decode generation uses
the joint generation timer. Full PD Decode additionally includes the PD
handoff boundary; for 14B both Decode rates also include the deferred
quant-profile/context load. The table is intentionally wide so configuration
or performance weaknesses remain visible.

| Model | Export and Decode configuration | TG / profile lifecycle | Sidecar overlap | Initialization once | QNN Prefill | Decode eval | Decode generation | Full PD Decode | First token / post-first | PD boundary | Memory | Correctness |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|---|
| Qwen3-4B, runs 1/2 | 36 layers; 18 shards; QNN U16 AR128; GPTQ2_32 GS32 native-I8MM; dynamic A8 scope=all; ctx4096; 6 threads | TG32; eager profile, auto lazy policy resolves off | V5 per-shard malloc; 141,010,816 B; wait 0.272 / 0.086 ms | 1949.214 / 1109.471 ms | 2088.988 / 1651.622 ms; **489.711 / 619.392 tok/s** | **16.630 / 16.340 tok/s** | 16.564 / 16.270 tok/s | **16.506 / 16.231 tok/s** | 61.274 / 78.059 ms; post-first 16.650 / 16.497 tok/s | 6.889 / 4.722 ms | HWM **2670.05 / 2668.95 MiB**; final RSS 2443.06 / 2443.47 MiB; sampled peak PSS 2440.46 / 2440.74 MiB; VmSwap 0 | exit 0; I8MM native; SHA256 4c384fda...d864 |
| Qwen3-8B | 36 layers; 18 shards; QNN U16 AR128; GPTQ2_32 GS32 native-I8MM; dynamic A8 scope=all; ctx4096; 6 threads | TG64; eager profile, auto lazy policy resolves off | V5 per-shard malloc; 249,692,032 B; wait 1.340 ms | 3385.499 ms | 4279.737 ms; **239.036 tok/s** | **8.600 tok/s** | 8.572 tok/s | **8.558 tok/s** | 140.717 ms; post-first 8.626 tok/s | 11.946 ms | HWM **4197.61 MiB**; final RSS 3769.93 MiB; sampled peak PSS 3767.14 MiB; VmSwap 0 | exit 0; I8MM native; SHA256 3008182f...8a8 |
| Qwen3-14B | 40 layers; 20 shards; QNN U16 AR128; GPTQ2_32 GS32 native-I8MM; dynamic A8 scope=all; ctx4096; 6 threads | TG32; auto enables post-Prefill lazy profile; original serial meta parser; no synthetic handoff warmup | V5 per-shard malloc; 459,454,080 B; wait 47.068 ms; .bin stays overlapped | 4617.327 ms; metadata 296.506 ms; metadata plus TG context 498.463 ms | 11133.783 ms; **91.97 tok/s** | **5.27 tok/s** | 4.855 tok/s | **4.819 tok/s** | 961.388 ms; post-first 5.516 tok/s | 49.293 ms | HWM **6816.84 MiB**; final RSS 6115.95 MiB; sampled peak PSS 6113.06 MiB; VmSwap 0; final profile release -533.75 MiB RSS | exit 0; I8MM native; SHA256 4c384fda...d864 |

The 4B figures are two retained zero-transfer runs. The 8B row is the accepted
TG64 monitor-lifecycle-repair run. The 14B row supersedes the old 7.30-GiB-HWM
summary for the current default: only the expanded .meta runtime object is
deferred, while the V5 .bin keeps its Prefill-overlapped read path. Its load
is intentionally counted in first-token and Decode speed, and the profile is
the first Decode resource released after the final token.

## Decode Instrumentation Safety Contract

The production Decode interval starts immediately before synchronous joint
handoff and ends after generation. It MUST NOT overlap any dense observation
work, including:

- `/proc` RSS/PSS/status/maps/smaps scanning;
- `mincore` residency walks or page-by-page residency probes;
- per-node/per-operator timing, graph-shape dumps, or token-by-token external
  polling used only for diagnosis.

This is a correctness condition for performance measurement, not merely a
benchmark preference. A former unconditional 5 ms memory monitor remained
alive through synchronous joint Decode, consumed 186 system jiffies in a
three-second window, periodically descheduled GGML workers, and reduced 8B
TG64 from 8.60 to 4.54 tok/s. The affected graph node appeared slow even when
its maximum worker compute was only 0.008 ms.

The required lifecycle is:

1. Dense memory sampling may run during initialization and Prefill diagnostics.
2. At `PrefillComplete`, request monitor shutdown.
3. Before `begin_resident_decode_handoff()`, join the monitor and verify it is
   no longer running.
4. Enter the already initialized Decode runtime without releasing all Prefill
   state, restarting Decode, warming up, or starting another observer.
5. Record kernel `VmHWM` and boundary RSS for normal reports. Dense Decode PSS
   or residency traces are diagnostic-only and their throughput is invalid.

The joint runner enforces step 3 with a runtime invariant and emits
`PD memory monitor stopped before joint Decode: active=0`. Any future runner,
test harness, or refactor that cannot produce this invariant must not publish
its Decode throughput as a production result.
