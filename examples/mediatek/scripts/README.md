# Qwen3-1.7B shared Int16 KV workflow

The default PD configuration is A16W4 per-channel, 14 chunks / 28 layers,
grouped GQA, AR128 + AR1, new-token KV outputs, compiler opt3 on DX3.
Corresponding KV input/output/concat observers share one scale. Only KV IO is
Int16; hidden states, mask and RoPE remain FP32. Runner cache updates append
Int16 codes directly, with one history cache. QNN-U8 handoff overlaps the next
prefill chunk. Decode remains the validated llama.cpp CPU path.

## Export

```sh
bash executorch/examples/mediatek/scripts/export_qwen3_1_7b_pd.sh
CACHE_SIZE=1024 bash executorch/examples/mediatek/scripts/export_qwen3_1_7b_pd.sh
```

Default cache is 4096; both validated cache sizes are supported. The default
root is `/root/autodl-tmp/mtk_models/qwen3_1_7b/current`. The existing source QAT
weights, direct qparams and calibration file are required. Override WORK_ROOT,
ET_ROOT, MTK_ENV_PREFIX, EXPERIMENT_ROOT, QPARAMS_DIR or CALIBRATION_FILE when
moving the workspace. START_CHUNK/END_CHUNK permit bounded diagnostic exports.

The export entry invokes normal `qwen.py`, not an experiment hook. The backend
resolves KV outputs by graph identity (delegate outputs may be reordered),
validates shared symmetric Int16 scales, removes only KV boundary Q/DQ and
writes method scale records. The exporter validates every method, publishes
SHORT PTE KV tensors and `kv_io_qparams.txt`; the script stages these under
`ctx<CACHE>/io_qparams/chunk_XX.txt` alongside stripped models. Export-only
compile metadata is removed from the final PTE. A legacy/debug export can opt
out with `qwen.py --no-shared-kv-quantization`.

## Run on OPPO through adb-hub

Default model root: `/data/local/tmp/et_mtk/qwen3_1_7b_shared_int16`.
Deploy `ctx1024/` and `ctx4096/` (each containing `gguf_stripped_chunks/` and
`io_qparams/`), `mtk_llama_pd_joint_runner`, `libneuron_backend.so`, and this
launcher as `run.sh`. The launcher resolves KV sidecars automatically.
Existing shared GGUF, embedding, QNN ABI/profile and benchmark inputs at the
paths in the launcher are required; these are current dependencies, not
obsolete prefill models.

```sh
sh /data/local/tmp/et_mtk/qwen3_1_7b_shared_int16/run.sh
CACHE=1024 sh /data/local/tmp/et_mtk/qwen3_1_7b_shared_int16/run.sh
MODE=ppl sh /data/local/tmp/et_mtk/qwen3_1_7b_shared_int16/run.sh
```

Run these commands through adb-hub. Set OUT to a new output prefix for each
measurement. Defaults preserve taskset fc, six decode threads, ctx2048,
greedy TG32, pipeline and memory monitoring. PPL uses the established 512-token
prefix / 1024-target protocol. `N_PREDICT=128` requests a longer generation;
`MTK_PD_VERIFY_KV_HANDOFF=1` enables a slow, separate canonical-FP16 handoff
comparison and must be excluded from timing runs.

Validated full-model PPL: 8.96015 (cache1024), 8.92234 (cache4096).
Full-prefill latency fell 7.4% / 14.6% against the previous FP32 KV version.
Decode retest was 24.30 tok/s and TG128 was 24.57 tok/s. One original TG32
run was 20.41 tok/s; it remains in the record (four-run mean 23.15 vs baseline
24.06). Full evidence is in
`/root/autodl-tmp/experiments/mtk_shared_kv_full_20260911/RESULT.md`.
