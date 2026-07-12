/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/extension/llm/sampler/sampler.h>
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor_ptr.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace example {
class DecoderRunner {
 public:
  struct PrefillShardRebuildConfig {
    enum class SourceKind {
      None,
      QatCheckpoint,
      TmacGguf,
      Gguf,
    };
    SourceKind source_kind{SourceKind::None};
    std::shared_ptr<std::vector<uint8_t>> source_bytes;
    int bits_hint{2};
    int group_size{32};
    std::string qweight_mode{"qweight_minus_qzeros"};
  };
  struct PrefillShardRuntimeStats {
    size_t layer_offset{0};
    size_t layer_count{0};
    size_t execution_count{0};
    double materialize_ms{0.0};
    double rebuild_ms{0.0};
    double execute_ms{0.0};
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

  void use_qwen3_prefill_static_plan(
      bool enabled,
      int32_t aux_size = 64,
      int32_t hidden_size = 2048);
  void configure_prefill_shards(
      int64_t num_layers,
      int32_t context_len,
      int32_t prompt_ar_len,
      int32_t vocab_size,
      bool has_window_attention_mask);
  std::vector<PrefillShardRuntimeStats> prefill_shard_runtime_stats() const;
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
    std::string pte_path;
    std::string index_path;
    std::string method_name;
    bool rebuild_on_execute{false};
    size_t layer_offset{0};
    size_t layer_count{0};
    PrefillShardRuntimeStats runtime_stats;
    std::vector<InputBinding> input_bindings;
    std::vector<OutputBinding> output_bindings;
    std::vector<executorch::extension::TensorPtr> owned_inputs;
    std::vector<executorch::extension::TensorPtr> owned_outputs;
    std::vector<executorch::aten::Tensor> output_tensors;
  };

  void materialize_prefill_shard(PrefillShardPlan& shard);
  void release_prefill_shard(PrefillShardPlan& shard);
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
};
} // namespace example
