# MTK incremental prefill and diagnostics

Qwen exports new-token-only KV by default (`--output-new-cache-only`).
`--no-output-new-cache-only` retains the legacy full-cache output ABI.
The runner detects the ABI from method metadata, allocates separate incremental
outputs, and updates the right-aligned history on the CPU. It moves only the
valid history suffix; the first block only appends new tokens. Default-layout
clone normalization and grouped GQA are also applied automatically.

Run the host regression (including the actual C++ updater) in the MTK Python
environment:

```sh
python examples/mediatek/tests/test_incremental_kv.py
```

## Device validation: OPPO PHZ110 / Dimensity 9300, 2026-09-10

Qwen3-1.7B, 14 two-layer chunks, AR128, 1024-token standard prompt (1023
prefilled tokens), 32 decode tokens, same existing GQA PTEs and decode settings.
Two independent speed processes per implementation; detailed KV timing was
collected separately. Only the valid-history update changed between runners.

| Cache length | Full-history update prefill | Valid-history update prefill | Throughput after |
|---|---:|---:|---:|
| 4096 | 6219.899 ms | 5807.968 ms | 176.31 tokens/s |
| 1024 | 3073.289 ms | 3018.060 ms | 339.29 tokens/s |

Detailed CPU KV update totals: 498.888 → 82.205 ms (4096),
134.040 → 84.213 ms (1024). PPL was unchanged: 13.8113 and 14.2742,
respectively. Speed-run stdout and 129/257-token handoff bytes matched.
These numbers compare valid-history vs full-history CPU movement, not GQA
vs another graph. The optimization requires no PTE re-export once using the
incremental KV ABI.

## Optional diagnostics (disabled by default)

- `MTK_PD_DETAIL_TIMING=1`: backend prepare/bind/compute and host KV timing.
- `MTK_PD_BIND_TIMING=1`: per-IO cache, allocator Find, mutex acquisition,
  dynamic-library lookup, SDK binding wall time and SDK return code. Prints
  after compute. `find_us` includes `lock_us` and `lookup_us`; do not sum them
  twice. On cache hits no SDK call occurs; the default status field is not
  evidence of a call. Timing instrumentation affects measured execution.

The binding instrumentation deliberately preserves the existing cache/error
handling behavior. It does not fix pointer-only cache keys, stale entries on
the raw-pointer path, or caching before successful SDK binding.

An isolated full-method probe can be built by setting the existing Android
example CMake configuration's `MTK_CHUNK_PROBE_SOURCE` to the absolute path of
`examples/mediatek/tests/chunk_probe.cpp`, then building target `mtk_chunk_probe`.
Invoke `mtk_chunk_probe model.pte fixture_dir output_dir iterations`, creating
output_dir first. The fixture contains contiguous input_N.bin files matching
model input bytes; the probe writes output_N.bin and times three warmups before
measurement. Its finite-output scan assumes FP32 outputs, as in this experiment.

SDK diagnostics additionally require compiling with `--profiling-level=2
--gen-debug-info` and setting `MTKNN_ENABLE_PROFILER=1`,
`MTKNN_PROFILING_LEVEL=2`, `MTKNN_PER_OP_PROFILE=1`, and
`MTKNN_PROFILER_LOG_PATH` to an absolute **file** path. These compilation flags
were used only in experiment wrappers, not enabled in production exports.
On the tested phone the report exposed MDLA/EDPA device times, but all MDLA
per-op entries were grouped under unknown location 4194303; no reliable
attention/projection/MLP attribution has yet been obtained.
