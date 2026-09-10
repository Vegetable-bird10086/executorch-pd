#!/usr/bin/env bash

set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp}"
ET_ROOT="${ET_ROOT:-$WORK_ROOT/executorch}"
MTK_ENV_PREFIX="${MTK_ENV_PREFIX:-$WORK_ROOT/miniconda3/envs/mtk}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$WORK_ROOT/mtk_models/qwen3_1_7b/current}"
CACHE_SIZE="${CACHE_SIZE:-4096}"
[[ "$CACHE_SIZE" = 1024 || "$CACHE_SIZE" = 4096 ]] || { echo "CACHE_SIZE must be 1024 or 4096" >&2; exit 2; }
QPARAMS_DIR="${QPARAMS_DIR:-$WORK_ROOT/experiments/qwen3_1_7b_mtk_per_channel_v2_20260824/v2/mtk_qparams_direct}"
START_CHUNK="${START_CHUNK:-0}"
END_CHUNK="${END_CHUNK:-13}"
CALIBRATION_FILE="${CALIBRATION_FILE:-$WORK_ROOT/pd_docs/backend/mtk/qwen3_zh_calibration.txt}"

MODEL_CONFIG="models/llm_models/weights/Qwen3-1.7B-E2E-QP-per-channel/config.json"
EXPORT_ROOT="$EXPERIMENT_ROOT/ctx$CACHE_SIZE/mtk_chunk_exports"
STRIPPED_ROOT="$EXPERIMENT_ROOT/ctx$CACHE_SIZE/gguf_stripped_chunks"
IO_ROOT="$EXPERIMENT_ROOT/ctx$CACHE_SIZE/io_qparams"
mkdir -p "$IO_ROOT"
STRIP_TOOL="$ET_ROOT/examples/mediatek/scripts/mtk_pte_weight_strip.py"

[[ "$START_CHUNK" =~ ^[0-9]+$ && "$END_CHUNK" =~ ^[0-9]+$ ]] || {
  echo "START_CHUNK and END_CHUNK must be integers" >&2
  exit 2
}
(( START_CHUNK <= END_CHUNK && END_CHUNK < 14 )) || {
  echo "expected 0 <= START_CHUNK <= END_CHUNK < 14" >&2
  exit 2
}
[[ -x "$MTK_ENV_PREFIX/bin/python" ]] || { echo "missing MTK Python" >&2; exit 2; }
[[ -f "$QPARAMS_DIR/manifest.json" ]] || { echo "missing direct qparams" >&2; exit 2; }
[[ -f "$CALIBRATION_FILE" ]] || { echo "missing calibration file" >&2; exit 2; }

cd "$ET_ROOT/examples/mediatek"
for chunk in $(seq "$START_CHUNK" "$END_CHUNK"); do
  tag=$(printf '%02d' "$chunk")
  output="$EXPORT_ROOT/chunk_$tag"
  stripped="$STRIPPED_ROOT/chunk_$tag"
  mkdir -p "$output" "$stripped"

  mapfile -t existing < <(find "$output" -maxdepth 1 -type f -name "*_${chunk}.pte" | sort)
  if [[ "${#existing[@]}" -eq 0 ]]; then
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="$WORK_ROOT:$ET_ROOT/third-party/ao${PYTHONPATH:+:$PYTHONPATH}" \
      "$MTK_ENV_PREFIX/bin/python" model_export_scripts/qwen.py \
        "$MODEL_CONFIG" \
        -p A16W4 \
        --num_chunks 14 \
        --preformatter aot_utils/llm_utils/preformatter_templates/qwen3.json \
        -shapes "128t${CACHE_SIZE}c" "1t${CACHE_SIZE}c" \
        --platform DX3 \
        --output-folder "$output" \
        --calibration-mode prompt-only \
        --calibration-response-steps 0 \
        --no-import-forever \
        --export-chunk "$chunk" \
        --lm-head-precision same \
        --lm-head-shard-size 0 \
        --prefill-no-lm-head \
        --compiler-opt-level 3 \
        --dump-qweights \
        --shared-kv-quantization \
        --mtk-gguf-qparams-dir "$QPARAMS_DIR" \
        -d "$CALIBRATION_FILE" \
        2>&1 | tee "$output/export.log"
    mapfile -t existing < <(find "$output" -maxdepth 1 -type f -name "*_${chunk}.pte" | sort)
  fi
  [[ "${#existing[@]}" -eq 1 ]] || {
    echo "chunk $chunk: expected exactly one PTE, found ${#existing[@]}" >&2
    exit 1
  }

  manifest="$output/qweights/chunk_$tag/manifest.json"
  [[ -f "$manifest" ]] || { echo "chunk $chunk: missing qweight manifest" >&2; exit 1; }
  if [[ ! -s "$stripped/stripped.pte" || ! -s "$stripped/index.bin" ]]; then
  "$MTK_ENV_PREFIX/bin/python" "$STRIP_TOOL" strip-gguf \
    --pte "${existing[0]}" \
    --qweight-manifest "$manifest" \
    --chunk-index "$chunk" \
    --layers-per-chunk 2 \
    --source-group-size 32 \
    --output-dir "$stripped" \
    > "$stripped/strip.log"
  fi
  [[ -s "$output/kv_io_qparams.txt" ]] || { echo "Missing shared Int16 KV sidecar for chunk $chunk; use a new export directory" >&2; exit 1; }
  cp "$output/kv_io_qparams.txt" "$IO_ROOT/chunk_$tag.txt"
  echo "chunk=$tag pte=$(stat -c %s "${existing[0]}") stripped=$(stat -c %s "$stripped/stripped.pte")"
done

# Reused directories must satisfy the same Int16 ABI; never silently select an
# older FP32-KV or unshared-scale PTE just because its filename already exists.
PYTHONPATH="$WORK_ROOT:$ET_ROOT/third-party/ao${PYTHONPATH:+:$PYTHONPATH}" \
  "$MTK_ENV_PREFIX/bin/python" -m executorch.backends.mediatek.quantized_kv_io "$EXPORT_ROOT"
