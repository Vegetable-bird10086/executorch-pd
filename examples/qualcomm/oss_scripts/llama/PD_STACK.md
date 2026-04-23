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
