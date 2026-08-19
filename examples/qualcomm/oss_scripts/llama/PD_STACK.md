# PD Stack Layout

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
model working set exceeded available physical memory and repeatedly faulted mmap
pages. It
must not be interpreted as Q4_0 kernel throughput.
