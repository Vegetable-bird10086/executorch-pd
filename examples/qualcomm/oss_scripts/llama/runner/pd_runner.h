#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/separate_embed.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace example {

template <typename T>
class PDPrefillRunner {
 public:
  struct MemoryHandoff {
    int fd{-1};
    uint64_t size_bytes{0};
    int32_t prompt_length{0};
    int32_t num_layers{0};
    int32_t num_kv_heads{0};
    int32_t head_dim{0};
    uint64_t first_token{0};
    bool first_token_is_prompt_tail{false};
  };

  struct StaticMetadata {
    bool enabled{false};
    int32_t context_len{0};
    int32_t prompt_ar_len{0};
    int32_t token_generator_ar_len{1};
    int32_t vocab_size{0};
    int32_t sliding_window{0};
    int64_t num_layers{0};
    int64_t num_heads{0};
    int64_t head_dim{0};
    bool use_int64_token{false};
    CacheMode cache_mode{CacheMode::StaticCahce};
    bool outputs_logits{true};
    bool use_separate_embed{false};
    std::string embedding_matrix_path;
    bool resident_embedding{true};
  };

  struct RuntimeStats {
    int32_t prompt_tokens{0};
    double tokenize_ms{0.0};
    double prefill_ms{0.0};
    double handoff_total_ms{0.0};
    double kv_layout_ms{0.0};
    double kv_write_ms{0.0};
    double fingerprint_ms{0.0};
  };

  explicit PDPrefillRunner(
      std::unique_ptr<executorch::extension::Module> module,
      std::vector<std::string> prefill_shard_paths,
      std::vector<std::string> prefill_shard_index_paths,
      DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild,
      bool prefill_qwen3_static_plan,
      int32_t prefill_static_aux_size,
      int32_t prefill_static_hidden_size,
      bool prefill_outputs_logits,
      bool separate_embed,
      const std::string& embedding_matrix_path,
      bool resident_embedding,
      StaticMetadata static_metadata,
      const std::string& decoder_model,
      const std::string& model_path,
      const std::string& tokenizer_path,
      std::shared_ptr<std::vector<uint8_t>> pte_bytes = nullptr,
      const int eval_mode = 1,
      const bool shared_buffer = false,
      std::unique_ptr<tokenizers::Tokenizer> tokenizer = nullptr,
      std::unique_ptr<executorch::extension::Module>
          attention_sink_rope_module = nullptr);

  bool is_loaded() const;
  executorch::runtime::Error load();
  void reset();

  executorch::runtime::Result<DecoderModelVersion> get_decoder_model_version();
  const RuntimeStats& last_runtime_stats() const;
  std::vector<DecoderRunner::PrefillShardRuntimeStats>
  prefill_shard_runtime_stats() const;
  double prefill_qnn_backend_prewarm_ms() const;
  bool prefill_qnn_backend_prewarmed() const;
  void set_prefill_etdump_config(DecoderRunner::PrefillEtDumpConfig config);

  executorch::runtime::Error export_prefill_memory_handoff(
      const std::string& prompt,
      bool tokenized_prompt,
      int32_t seq_len,
      MemoryHandoff* memory_handoff);

  executorch::runtime::Error export_prefill_handoff_files(
      const std::string& prompt,
      bool tokenized_prompt,
      int32_t seq_len,
      const std::string& export_dir,
      const std::string& kv_quant_attrs_path = "");

  executorch::runtime::Error evaluate_wikitext_ppl(
      const std::string& wikitext_path,
      int32_t start_token,
      int32_t max_eval_tokens,
      float logits_scale,
      int32_t logits_zero_point,
      double* ppl_out,
      int64_t* scored_tokens_out = nullptr);

 private:
  executorch::runtime::Error export_prefill_handoff_impl(
      const std::string& prompt,
      bool tokenized_prompt,
      int32_t seq_len,
      const std::string& export_dir,
      const std::string& kv_quant_attrs_path,
      MemoryHandoff* memory_handoff,
      bool write_files);

  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module_;
  std::vector<std::string> prefill_shard_paths_;
  std::vector<std::string> prefill_shard_index_paths_;
  DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild_;
  DecoderRunner::PrefillEtDumpConfig prefill_etdump_config_;
  bool prefill_qwen3_static_plan_{false};
  int32_t prefill_static_aux_size_{64};
  int32_t prefill_static_hidden_size_{2048};
  bool prefill_outputs_logits_{true};
  bool separate_embed_{false};
  std::string embedding_matrix_path_;
  bool resident_embedding_{true};
  SeparateEmbedding separate_embedding_;
  StaticMetadata static_metadata_;
  std::string model_path_;
  std::string tokenizer_path_;
  std::shared_ptr<std::vector<uint8_t>> pte_bytes_;
  EvalMode eval_mode_;
  bool shared_buffer_;

  int32_t context_len_{0};
  int32_t prompt_processor_ar_len_{0};
  int32_t token_generator_ar_len_{0};
  int32_t max_cache_len_{0};
  int32_t prefill_cache_stride_{0};
  int32_t vocab_size_{0};
  int64_t cur_pos_{0};
  int64_t num_layers_{0};
  int64_t num_heads_{0};
  int64_t head_dim_{0};
  CacheMode cache_mode_{CacheMode::StaticCahce};
  DecoderModelVersion decoder_model_version_;

  std::unique_ptr<IMemAlloc> buffer_manager_;
  std::unique_ptr<KVManager<T>> kv_manager_;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
  std::unique_ptr<DecoderRunner> decoder_runner_;
  std::unique_ptr<AttentionSinkRopeRunner> attention_sink_rope_runner_;
  std::unique_ptr<PromptProcessor<T>> prompt_processor_;
  RuntimeStats last_runtime_stats_;
};

} // namespace example
