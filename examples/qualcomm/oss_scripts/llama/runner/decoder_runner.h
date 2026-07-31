/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/devtools/etdump/etdump_flatcc.h>
#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <future>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace example {

class DecoderRunner {
 public:
  struct PrefillEtDumpConfig {
    // ETDump is intentionally opt-in: a QNN PTE must also have been exported
    // with dump_intermediate_outputs enabled for delegate tensors to appear.
    std::string output_dir;
    int32_t shard_index{-1};
    size_t debug_buffer_bytes{0};

    bool enabled() const {
      return !output_dir.empty() && shard_index >= 0 &&
          debug_buffer_bytes > 0;
    }
  };

  struct PrefillShardRebuildConfig {
    enum class SourceKind {
      None,
      QatCheckpoint,
      TmacGguf,
      Gguf,
    };
    SourceKind source_kind{SourceKind::None};
    std::shared_ptr<std::vector<uint8_t>> source_bytes;
    std::shared_ptr<ReadOnlyMappedFile> mapped_source_bytes;
    int bits_hint{2};
    int group_size{32};
    std::string qweight_mode{"qweight_minus_qzeros"};
    PteGgufRecipeRelayoutKind gguf_relayout_kind{
        PteGgufRecipeRelayoutKind::None};
    bool pipeline_rebuild{false};
    bool pipeline_qnn_load{false};
    bool stage_major_execution{false};
    // Called once, after all rebuild inputs and idle rebuild buffers have been
    // released, immediately before the final shard starts executing.
    std::function<void()> final_shard_overlap_callback;
    bool prewarm_qnn_backend{false};
    // Materialize and load shard 0 during runner preparation, then keep its
    // QNN context alive until the DecoderRunner is destroyed.
    bool persistent_shard0_context{false};
    // Keep GGUF recipes, stripped shard inputs, and reusable rebuild buffers
    // alive so the same DecoderRunner can serve another request.
    bool retain_session_rebuild_resources{false};
    // Raw qnn_compile_spec copied from a complete, unstripped PTE. Stripped
    // PTEs are not valid FlatBuffers and must never be parsed for this.
    std::shared_ptr<std::vector<uint8_t>> qnn_compile_spec_bytes;
  };
  struct PrefillShardRuntimeStats {
    size_t layer_offset{0};
    size_t layer_count{0};
    size_t execution_count{0};
    double preload_ms{0.0};
    double qat_checkpoint_context_ms{0.0};
    double qat_recipe_ms{0.0};
    double gguf_checkpoint_context_ms{0.0};
    double gguf_recipe_ms{0.0};
    double gguf_relayout_ms{0.0};
    size_t gguf_relayout_bytes{0};
    double materialize_ms{0.0};
    double rebuild_ms{0.0};
    double rebuild_allocation_ms{0.0};
    double rebuild_static_copy_ms{0.0};
    double rebuild_weight_materialization_ms{0.0};
    double pipeline_wait_ms{0.0};
    double qnn_load_method_ms{0.0};
    double input_binding_ms{0.0};
    double output_binding_ms{0.0};
    double execute_ms{0.0};
    double output_copy_ms{0.0};
    double release_ms{0.0};
    double total_ms{0.0};
    uint64_t rss_before_bytes{0};
    uint64_t rss_after_materialize_bytes{0};
    uint64_t rss_after_load_bytes{0};
    uint64_t rss_after_execute_bytes{0};
    uint64_t rss_after_release_bytes{0};
    uint64_t hwm_before_bytes{0};
    uint64_t hwm_after_materialize_bytes{0};
    uint64_t hwm_after_load_bytes{0};
    uint64_t hwm_after_execute_bytes{0};
    uint64_t hwm_after_release_bytes{0};
  };
  struct PrefillShardStageState {
    std::vector<executorch::extension::TensorPtr> aux;
    executorch::extension::TensorPtr hidden;
  };

  DecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature);
  DecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature,
      std::vector<std::string> prefill_shard_paths);
  DecoderRunner(
      executorch::extension::Module* module,
      int32_t vocab_size,
      float temperature,
      std::vector<std::string> prefill_shard_paths,
      std::vector<std::string> prefill_shard_index_paths,
      PrefillShardRebuildConfig prefill_shard_rebuild);
  ~DecoderRunner();

  void use_qwen3_prefill_static_plan(
      bool enabled,
      int32_t aux_size = 64,
      int32_t hidden_size = 2048);
  void set_prefill_outputs_logits(bool enabled);
  void set_prefill_separate_embed(bool enabled);
  void set_prefill_etdump_config(PrefillEtDumpConfig config);
  void configure_prefill_shards(
      int64_t num_layers,
      int32_t context_len,
      int32_t prompt_ar_len,
      int32_t vocab_size,
      bool has_window_attention_mask);
  std::vector<PrefillShardRuntimeStats> prefill_shard_runtime_stats() const;
  double prefill_qnn_backend_prewarm_ms() const;
  bool prefill_qnn_backend_prewarmed() const;
  double prefill_persistent_shard0_prepare_ms() const;
  bool prefill_persistent_shard0_prepared() const;
  void begin_prefill_request();
  bool uses_prefill_shard_stage_major() const;
  size_t prefill_shard_count() const;
  size_t prefill_shard_layer_offset(size_t shard_index) const;
  size_t prefill_shard_layer_count(size_t shard_index) const;
  executorch::runtime::Error begin_prefill_shard_stage(size_t shard_index);
  void prepare_final_prefill_shard_overlap();
  void set_prefill_shard_release_callback(
      std::function<void(size_t, size_t, size_t)> callback);
  executorch::runtime::Result<PrefillShardStageState> step_prefill_shard_stage(
      size_t shard_index,
      std::vector<executorch::runtime::EValue>& inputs,
      const PrefillShardStageState* previous_stage);
  executorch::runtime::Error end_prefill_shard_stage(size_t shard_index);
  /**
   * Run LLM text decoder with inputs to generate next token.
   * @param inputs The inputs to the LLM Module.
   * @return The output of the LLM Module. This will be a tensor of logits.
   */
  executorch::runtime::Result<executorch::aten::Tensor> step(
      const std::string& method_name,
      std::vector<executorch::runtime::EValue>& inputs);

  /**
   * Once KV Cache output data pointer change, need to set
   * the output for specify method name in the module.
   * @return The error code.
   */
  executorch::runtime::Error set_outputs(
      const std::string& method_name,
      std::vector<executorch::aten::Tensor> output_values);

  /**
   * Load the Module for text decode purpose.
   * @return The error code.
   */
  executorch::runtime::Error load(const std::vector<std::string>& method_names);
  /**
   * Check if the required methods in the Module is loaded.
   * @return True if the Module is loaded, false otherwise.
   */
  bool is_method_loaded(const std::vector<std::string>& method_names);

  /**
   * Sample the next token from the logits tensor.
   * @param logits_tensor The logits tensor.
   * @return The next token.
   */
  inline int32_t logits_to_token(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos) {
    auto* logits_last = logits_ptr_for_pos(logits_tensor, pos);
    auto vocab_size = logits_tensor.size(2);
    static std::vector<float> logits_f(vocab_size);
    // Discard dequantization (converting uint16_t to float) because the
    // relative order of elements remains the same without conversion
    for (int i = 0; i < vocab_size; i++) {
      logits_f[i] = logits_last[i];
    }
    return sampler_->sample(logits_f.data());
  }

  inline int32_t logits_to_argmax_token(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos) {
    auto* logits_last = logits_ptr_for_pos(logits_tensor, pos);
    const auto vocab_size = logits_tensor.size(2);
    int32_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    bool found_finite_logit = false;
    for (int32_t token = 0; token < vocab_size; ++token) {
      executorch::aten::Half fp16;
      std::memcpy(&fp16, &logits_last[token], sizeof(logits_last[token]));
      const float logit = static_cast<float>(fp16);
      if (!std::isnan(logit) &&
          (!found_finite_logit || logit > best_logit)) {
        best_logit = logit;
        best_token = token;
        found_finite_logit = true;
      }
    }
    if (found_finite_logit) {
      return best_token;
    }

    // Keep the legacy ordering as a last-resort fallback for unsupported
    // logit payloads, but never let a NaN at token zero force token zero.
    uint16_t best_raw = logits_last[0];
    for (int32_t token = 1; token < vocab_size; ++token) {
      if (logits_last[token] > best_raw) {
        best_raw = logits_last[token];
        best_token = token;
      }
    }
    return best_token;
  }

 protected:
  inline uint16_t* logits_ptr_for_pos(
      const executorch::aten::Tensor& logits_tensor,
      int64_t pos) {
    auto* logits = logits_tensor.mutable_data_ptr<uint16_t>();
    auto num_tokens = logits_tensor.size(1);
    auto vocab_size = logits_tensor.size(2);
    auto* logits_last = logits;
    if (num_tokens > 1) {
      logits_last += pos * vocab_size;
    }
    return logits_last;
  }

  struct PrefillShardPlan {
    enum class InputKind {
      Tokens,
      AttentionMask,
      WindowAttentionMask,
      Position,
      KCache,
      VCache,
      PreviousAux,
      PreviousHidden,
    };
    enum class OutputKind {
      FinalLogits,
      FinalKCache,
      FinalVCache,
      IntermediateAux,
      IntermediateHidden,
    };
    struct InputBinding {
      InputKind kind;
      size_t index;
      size_t owned_index{static_cast<size_t>(-1)};
    };
    struct OutputBinding {
      OutputKind kind;
      size_t index;
      size_t owned_index;
    };
    std::unique_ptr<executorch::extension::Module> module;
    std::shared_ptr<std::vector<uint8_t>> rebuilt_pte_bytes;
    std::shared_ptr<PteRebuildBuffer> rebuilt_pte_buffer;
    std::shared_ptr<std::vector<uint8_t>> stripped_pte_bytes;
    std::shared_ptr<std::vector<uint8_t>> index_bytes;
    std::shared_ptr<PteQatShardRecipe> qat_rebuild_recipe;
    std::shared_ptr<PteGgufShardRecipe> gguf_rebuild_recipe;
    std::string pte_path;
    std::string index_path;
    std::string method_name;
    size_t shard_index{0};
    bool rebuild_on_execute{false};
    bool etdump_armed{false};
    bool etdump_written{false};
    std::unique_ptr<executorch::etdump::ETDumpGen> etdump_gen;
    std::vector<uint8_t> etdump_debug_buffer;
    size_t layer_offset{0};
    size_t layer_count{0};
    PrefillShardRuntimeStats runtime_stats;
    std::vector<InputBinding> input_bindings;
    std::vector<OutputBinding> output_bindings;
    std::vector<executorch::extension::TensorPtr> owned_inputs;
    std::vector<executorch::extension::TensorPtr> owned_outputs;
    std::vector<executorch::aten::Tensor> output_tensors;
  };

  struct PrefillShardThreeStagePipeline {
    std::mutex mutex;
    std::condition_variable cv;
    bool stop{false};
    size_t rebuild_permit_index{0};
    size_t load_permit_index{0};
    std::vector<std::optional<PteRebuildResult>> rebuild_results;
    std::vector<bool> rebuilt;
    std::vector<bool> loaded;
    std::thread rebuild_worker;
    std::thread load_worker;
  };

  void preload_prefill_shard(PrefillShardPlan& shard);
  PteRebuildResult rebuild_prefill_shard(PrefillShardPlan& shard);
  void attach_rebuilt_prefill_shard(
      PrefillShardPlan& shard,
      PteRebuildResult rebuild_result);
  void materialize_prefill_shard(PrefillShardPlan& shard);
  void release_prefill_shard(PrefillShardPlan& shard);
  std::shared_ptr<PteRebuildBuffer> acquire_prefill_rebuild_buffer(
      size_t required_size);
  void release_prefill_rebuild_buffer(
      std::shared_ptr<PteRebuildBuffer> buffer);
  executorch::runtime::Error load_prefill_shard_method(PrefillShardPlan& shard);
  bool should_arm_prefill_etdump(const PrefillShardPlan& shard) const;
  executorch::runtime::Error write_prefill_etdump(PrefillShardPlan& shard);
  bool uses_prefill_shard_three_stage_pipeline() const;
  void prepare_persistent_prefill_shard0();
  void start_prefill_shard_three_stage_pipeline();
  void stop_prefill_shard_three_stage_pipeline();
  void permit_prefill_shard_three_stage_pipeline(
      size_t rebuild_index,
      size_t load_index);
  double wait_for_prefill_shard_three_stage_load(size_t shard_index);
  executorch::runtime::Result<PrefillShardStageState> execute_prefill_shard(
      PrefillShardPlan& shard,
      std::vector<executorch::runtime::EValue>& inputs,
      const PrefillShardStageState* previous_stage);
  void configure_qwen3_static_prefill_shards();
  executorch::runtime::Result<executorch::aten::Tensor> step_prefill_shards(
      std::vector<executorch::runtime::EValue>& inputs);
  executorch::runtime::Error set_outputs_prefill_shards(
      std::vector<executorch::aten::Tensor> output_values);
  bool uses_prefill_shards() const;

  executorch::extension::Module* module_;
  std::unique_ptr<executorch::extension::llm::Sampler> sampler_;
  std::vector<std::string> prefill_shard_paths_;
  std::vector<std::string> prefill_shard_index_paths_;
  PrefillShardRebuildConfig prefill_shard_rebuild_;
  std::function<void(size_t, size_t, size_t)>
      prefill_shard_release_callback_;
  std::shared_ptr<PteQatRebuildContext> prefill_qat_rebuild_context_;
  std::shared_ptr<PteGgufRebuildContext> prefill_gguf_rebuild_context_;
  double prefill_qnn_backend_prewarm_ms_{0.0};
  bool prefill_qnn_backend_prewarmed_{false};
  double prefill_persistent_shard0_prepare_ms_{0.0};
  bool prefill_persistent_shard0_prepared_{false};
  std::mutex prefill_rebuild_buffer_pool_mutex_;
  std::vector<std::shared_ptr<PteRebuildBuffer>> prefill_rebuild_buffer_pool_;
  std::vector<PrefillShardPlan> prefill_shards_;
  std::vector<executorch::aten::Tensor> prefill_output_values_;
  int64_t prefill_num_layers_{0};
  int32_t prefill_context_len_{0};
  int32_t prefill_prompt_ar_len_{0};
  int32_t prefill_vocab_size_{0};
  bool prefill_has_window_attention_mask_{false};
  bool prefill_qwen3_static_plan_{false};
  int32_t prefill_static_aux_size_{64};
  int32_t prefill_static_hidden_size_{2048};
  bool prefill_outputs_logits_{true};
  bool prefill_separate_embed_{false};
  PrefillEtDumpConfig prefill_etdump_config_;
  std::vector<std::chrono::steady_clock::time_point> prefill_shard_stage_starts_;
  std::future<PteRebuildResult> prefill_shard_stage_pending_rebuild_;
  size_t prefill_shard_stage_pending_rebuild_index_{
      std::numeric_limits<size_t>::max()};
  std::unique_ptr<PrefillShardThreeStagePipeline>
      prefill_shard_three_stage_pipeline_;
};
} // namespace example
