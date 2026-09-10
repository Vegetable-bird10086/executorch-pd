#!/system/bin/sh
set -eu

# OPPO/Dimensity 9300 reproduction script for the Qwen3-1.7B MTK Prefill +
# llama.cpp Decode joint runner.  KV is handed off in process; this script does
# not use --pd_export_dir or --pd-import.
MODE=${MODE:-speed}
MODEL_ROOT=${MODEL_ROOT:-/data/local/tmp/et_mtk/qwen3_1_7b_shared_int16}
CACHE=${CACHE:-4096}
case "$CACHE" in 1024|4096) ;; *) echo "CACHE must be 1024 or 4096" >&2; exit 2;; esac
PTE_ROOT=${PTE_ROOT:-$MODEL_ROOT/ctx$CACHE}
export MTK_PD_KV_IO_QPARAMS=${MTK_PD_KV_IO_QPARAMS:-$PTE_ROOT/io_qparams}
DECODE_THREADS=${DECODE_THREADS:-6}
MODEL=/data/local/tmp/et_mtk/qwen3_1_7b_per_channel_pd_20260823
V2=/data/local/tmp/et_mtk/qwen3_1_7b_per_channel_pd_pc_direct_v2
RUNNER=${RUNNER:-$MODEL_ROOT/mtk_llama_pd_joint_runner}
GGUF=${GGUF:-$V2/model.gguf}
PROFILE=${PROFILE:-$V2/qnn_u16_runtime_profile_pc_compact.meta}
INPUT=/data/local/tmp/et_mtk/qwen3_1_7b_mtk_14chunk_kv_only/benchmark_128_1024_inputs
OUT=${OUT:-$PWD/mtk_shared_int16_${CACHE}_${MODE}}

stripped=
indexes=
i=0
while [ "$i" -lt 14 ]; do
  suffix=$(printf '%02d' "$i")
  chunk=$PTE_ROOT/gguf_stripped_chunks/chunk_$suffix
  test -s "$chunk/stripped.pte"
  test -s "$chunk/index.bin"
  test -s "$MTK_PD_KV_IO_QPARAMS/chunk_$suffix.txt"
  if [ -n "$stripped" ]; then
    stripped=$stripped,
    indexes=$indexes,
  fi
  stripped=$stripped$chunk/stripped.pte
  indexes=$indexes$chunk/index.bin
  i=$((i + 1))
done

case "$MODE" in
  speed|check)
    prompt=${PROMPT_PATH:-$INPUT/prompt_1024.u64}
    decode_args="--pd_joint_decode_n_predict=${N_PREDICT:-32}"
    if [ "$MODE" = check ]; then decode_args="$decode_args --pd_export_dir=$OUT.handoff"; fi
    ;;
  ppl)
    prompt=$INPUT/ppl_prompt_512.u64
    decode_args="--pd_joint_decode_n_predict=0 --decode_ppl_tokens_path=$INPUT/ppl_continuation_1025.u64 --decode_ppl_output_path=$OUT.ppl.txt --decode_ppl_max_tokens=1024"
    ;;
  *)
    echo "MODE must be speed or ppl" >&2
    exit 2
    ;;
esac

mkdir -p "$(dirname "$OUT")"
chmod +x "$RUNNER"
taskset fc env \
  LD_LIBRARY_PATH=$MODEL_ROOT:$MODEL \
  GGML_QNN_U16_ACTIVATIONS=1 \
  GGML_QNN_U16_BLOCKWISE_REQUANT=1 \
  GGML_QNN_DYNAMIC_A8_GEMV=1 \
  GGML_QNN_DYNAMIC_A8_SCOPE=all \
  GGML_QNN_PC_I8MM_GS32_COMPAT=0 \
  GGML_QNN_PC_I8MM_VALIDATE_GS32=0 \
  LLAMA_QNN_U16_QPARAMS_MANIFEST=$PROFILE \
  "$RUNNER" \
  --prompt_token_batch_size=128 --cache_size=$CACHE --hidden_size=2048 \
  --num_head=16 --num_layer=28 --head_dim=128 --max_token_length=40960 \
  --rot_emb_base=1000000 --input_type=fp32 --output_type=fp32 \
  --cache_type=fp32 --mask_type=fp32 --rot_emb_type=fp32 \
  --vocab_size=151936 --logit_shard_count=1 --bos_token=151643 \
  --eos_token=151645 --token_embedding_path=$MODEL/embedding_fp16.semb \
  --pd_prompt_tokens_path=$prompt \
  --pd_qnn_kv_abi_path=$MODEL/qwen3_1_7b_per_channel_qnn_kv_abi.bin \
  --pd_stage_major_stripped_pte_paths=$stripped \
  --pd_stage_major_index_paths=$indexes \
  --pd_stage_major_three_stage=1 --pd_stage_major_async_release=0 \
  --pd_stage_major_persistent_chunk0=1 \
  --pd_stage_major_detach_pte_after_load=1 \
  --pd_stage_major_session_repeats=1 --pd_prefill_pipeline=1 \
  --pd_joint_gguf_path=$GGUF \
  --pd_joint_disk_embedding_path=$MODEL/embedding_fp16.semb \
  --pd_joint_decode_ctx=2048 --pd_joint_decode_threads=$DECODE_THREADS \
  --pd_joint_decode_temp=0 $decode_args \
  > "$OUT.stdout" 2> "$OUT.stderr"
