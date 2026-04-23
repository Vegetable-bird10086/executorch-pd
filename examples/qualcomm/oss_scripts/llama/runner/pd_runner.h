#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/attention_sink_rope_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/cache_utils.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/runner.h>
#include <executorch/extension/module/module.h>
#include <pytorch/tokenizers/tokenizer.h>

#include <cstdint>
#include <memory>
#include <string>

namespace example {

template <typename T>
class PDPrefillRunner {
 public:
  explicit PDPrefillRunner(
      std::unique_ptr<executorch::extension::Module> module,
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

  executorch::runtime::Error export_prefill_handoff(
      const std::string& prompt,
      bool tokenized_prompt,
      int32_t seq_len,
      const std::string& export_dir,
      const std::string& kv_quant_attrs_path = "");

 private:
  enum EvalMode {
    kKVCached = 0,
    kHybrid,
    kLookaheadDecoding,
    kUnsupported,
  };

  std::unique_ptr<executorch::extension::Module> module_;
  std::unique_ptr<executorch::extension::Module> attention_sink_rope_module_;
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
};

} // namespace example
