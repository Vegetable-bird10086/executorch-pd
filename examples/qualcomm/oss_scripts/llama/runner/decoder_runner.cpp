/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Given inputs, run a text decoder and return logits.

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/decoder_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iterator>
#include <numeric>
using executorch::aten::Tensor;
using executorch::extension::Module;
using executorch::extension::llm::Sampler;
using executorch::extension::TensorPtr;
using executorch::llm::kTopp;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::Result;

namespace example {
namespace {

TensorPtr make_tensor_ptr_from_meta(const executorch::runtime::TensorInfo& meta) {
  std::vector<executorch::aten::SizesType> sizes(
      meta.sizes().begin(), meta.sizes().end());
  size_t numel = 1;
  for (const auto dim : sizes) {
    numel *= static_cast<size_t>(dim);
  }
  std::vector<uint8_t> storage(
      numel * executorch::runtime::elementSize(meta.scalar_type()));
  return executorch::extension::make_tensor_ptr(
      sizes,
      std::move(storage),
      std::vector<executorch::aten::DimOrderType>(
          meta.dim_order().begin(), meta.dim_order().end()),
      {},
      meta.scalar_type());
}

TensorPtr make_tensor_ptr_from_sizes(
    const std::vector<executorch::aten::SizesType>& sizes,
    executorch::aten::ScalarType scalar_type) {
  size_t numel = 1;
  for (const auto dim : sizes) {
    numel *= static_cast<size_t>(dim);
  }
  std::vector<uint8_t> storage(
      numel * executorch::runtime::elementSize(scalar_type));
  std::vector<executorch::aten::DimOrderType> dim_order;
  dim_order.reserve(sizes.size());
  for (size_t i = 0; i < sizes.size(); ++i) {
    dim_order.push_back(static_cast<executorch::aten::DimOrderType>(i));
  }
  return executorch::extension::make_tensor_ptr(
      sizes, std::move(storage), dim_order, {}, scalar_type);
}

bool is_rank4_tensor(const executorch::runtime::TensorInfo& meta) {
  return meta.sizes().size() == 4;
}

bool looks_like_value_cache(const executorch::runtime::TensorInfo& meta) {
  return is_rank4_tensor(meta) && meta.sizes()[2] > meta.sizes()[3];
}

bool is_rank2_integral_tensor(const executorch::runtime::TensorInfo& meta) {
  return meta.sizes().size() == 2 &&
      (meta.scalar_type() == executorch::aten::ScalarType::Int ||
       meta.scalar_type() == executorch::aten::ScalarType::Long);
}

bool is_prompt_token_or_position_tensor(
    const executorch::runtime::TensorInfo& meta,
    int32_t prompt_ar_len) {
  return meta.sizes().size() == 2 && meta.sizes()[0] == 1 &&
      meta.sizes()[1] == prompt_ar_len;
}

bool is_rank2_float_tensor(const executorch::runtime::TensorInfo& meta) {
  return meta.sizes().size() == 2 &&
      meta.scalar_type() != executorch::aten::ScalarType::Int &&
      meta.scalar_type() != executorch::aten::ScalarType::Long;
}

bool is_rank2_tensor(const executorch::runtime::TensorInfo& meta) {
  return meta.sizes().size() == 2;
}

bool is_mask_tensor(
    const executorch::runtime::TensorInfo& meta,
    int32_t prompt_ar_len,
    int32_t context_len) {
  return meta.sizes().size() == 3 && meta.sizes()[1] == prompt_ar_len &&
      meta.sizes()[2] == context_len;
}

bool is_logits_tensor(
    const executorch::runtime::TensorInfo& meta,
    int32_t vocab_size) {
  return meta.sizes().size() == 3 && meta.sizes()[2] == vocab_size;
}

void copy_tensor_data(const Tensor& src, const Tensor& dst) {
  ET_CHECK_MSG(
      src.nbytes() == dst.nbytes(),
      "Shard output size mismatch: src=%zu dst=%zu",
      src.nbytes(),
      dst.nbytes());
  if (src.const_data_ptr() != dst.mutable_data_ptr()) {
    std::memcpy(dst.mutable_data_ptr(), src.const_data_ptr(), src.nbytes());
  }
}

bool prefill_shards_disabled() {
  return std::getenv("EXECUTORCH_PREFILL_SHARD_DISABLE") != nullptr;
}

bool prefill_shard_swap_aux() {
  return std::getenv("EXECUTORCH_PREFILL_SHARD_SWAP_AUX") != nullptr;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

bool first_square_kv_is_v_for_shard(size_t layer_offset) {
  const char* mode = std::getenv("EXECUTORCH_PREFILL_SHARD_KV_MODE");
  if (mode != nullptr && std::strcmp(mode, "legacy") == 0) {
    return layer_offset == 0;
  }
  if (mode != nullptr && std::strcmp(mode, "all_kv") == 0) {
    return false;
  }
  return true;
}

} // namespace

DecoderRunner::DecoderRunner(
    Module* module,
    int32_t vocab_size,
    float temperature)
    : DecoderRunner(module, vocab_size, temperature, {}, {}, {}) {}

DecoderRunner::DecoderRunner(
    Module* module,
    int32_t vocab_size,
    float temperature,
    std::vector<std::string> prefill_shard_paths)
    : DecoderRunner(
          module,
          vocab_size,
          temperature,
          std::move(prefill_shard_paths),
          {},
          {}) {}

DecoderRunner::DecoderRunner(
    Module* module,
    int32_t vocab_size,
    float temperature,
    std::vector<std::string> prefill_shard_paths,
    std::vector<std::string> prefill_shard_index_paths,
    PrefillShardRebuildConfig prefill_shard_rebuild)
    : module_(module),
      sampler_(std::make_unique<Sampler>(
          vocab_size,
          temperature,
          kTopp,
          static_cast<unsigned long long>(std::time(nullptr)))),
      prefill_shard_paths_(std::move(prefill_shard_paths)),
      prefill_shard_index_paths_(std::move(prefill_shard_index_paths)),
      prefill_shard_rebuild_(std::move(prefill_shard_rebuild)) {}

void DecoderRunner::use_qwen3_prefill_static_plan(
    bool enabled,
    int32_t aux_size,
    int32_t hidden_size) {
  prefill_qwen3_static_plan_ = enabled;
  prefill_static_aux_size_ = aux_size;
  prefill_static_hidden_size_ = hidden_size;
}

void DecoderRunner::materialize_prefill_shard(PrefillShardPlan& shard) {
  if (shard.module != nullptr) {
    return;
  }

  if (shard.rebuild_on_execute) {
    ET_CHECK_MSG(
        prefill_shard_rebuild_.source_kind !=
            PrefillShardRebuildConfig::SourceKind::None,
        "Prefill shard index was provided but no rebuild source is configured");
    ET_CHECK_MSG(
        prefill_shard_rebuild_.source_bytes &&
            !prefill_shard_rebuild_.source_bytes->empty(),
        "Prefill shard rebuild source bytes are empty");
    const std::vector<uint8_t> stripped_pte = read_binary_file(shard.pte_path);
    const std::vector<uint8_t> index_bytes = read_binary_file(shard.index_path);
    PteRebuildResult rebuild_result;
    switch (prefill_shard_rebuild_.source_kind) {
      case PrefillShardRebuildConfig::SourceKind::QatCheckpoint:
        rebuild_result = rebuild_pte_from_stripped_checkpoint(
            stripped_pte,
            index_bytes,
            *prefill_shard_rebuild_.source_bytes,
            prefill_shard_rebuild_.bits_hint,
            prefill_shard_rebuild_.group_size,
            prefill_shard_rebuild_.qweight_mode);
        break;
      case PrefillShardRebuildConfig::SourceKind::TmacGguf:
        rebuild_result = rebuild_pte_from_stripped_tmac_gguf(
            stripped_pte, index_bytes, *prefill_shard_rebuild_.source_bytes);
        break;
      case PrefillShardRebuildConfig::SourceKind::Gguf:
        rebuild_result = rebuild_pte_from_stripped_gguf(
            stripped_pte, index_bytes, *prefill_shard_rebuild_.source_bytes);
        break;
      case PrefillShardRebuildConfig::SourceKind::None:
        ET_CHECK_MSG(false, "Invalid prefill shard rebuild source");
        break;
    }
    shard.rebuilt_pte_bytes = rebuild_result.rebuilt_pte;
    auto data_loader = std::make_unique<executorch::extension::BufferDataLoader>(
        shard.rebuilt_pte_bytes->data(), shard.rebuilt_pte_bytes->size());
    shard.module = std::make_unique<Module>(std::move(data_loader));
    ET_LOG(
        Info,
        "rebuilt prefill shard: stripped=%s index=%s materialized_weight_bytes=%zu rebuilt_records=%zu rebuild_ms=%f",
        shard.pte_path.c_str(),
        shard.index_path.c_str(),
        rebuild_result.materialized_weight_bytes,
        rebuild_result.rebuilt_records,
        rebuild_result.rebuild_time_ms);
  } else {
    shard.module = std::make_unique<Module>(
        shard.pte_path.c_str(), Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
}

void DecoderRunner::release_prefill_shard(PrefillShardPlan& shard) {
  if (!shard.rebuild_on_execute) {
    return;
  }
  shard.module.reset();
  shard.rebuilt_pte_bytes.reset();
}

void DecoderRunner::configure_prefill_shards(
    int64_t num_layers,
    int32_t context_len,
    int32_t prompt_ar_len,
    int32_t vocab_size,
    bool has_window_attention_mask) {
  if (prefill_shard_paths_.empty() || prefill_shards_disabled()) {
    if (!prefill_shard_paths_.empty()) {
      ET_LOG(Info, "prefill shards disabled by EXECUTORCH_PREFILL_SHARD_DISABLE");
    }
    return;
  }

  prefill_num_layers_ = num_layers;
  prefill_context_len_ = context_len;
  prefill_prompt_ar_len_ = prompt_ar_len;
  prefill_vocab_size_ = vocab_size;
  prefill_has_window_attention_mask_ = has_window_attention_mask;
  prefill_shards_.clear();
  prefill_shards_.reserve(prefill_shard_paths_.size());

  if (prefill_qwen3_static_plan_) {
    configure_qwen3_static_prefill_shards();
    return;
  }

  size_t layer_offset = 0;
  for (const std::string& shard_path : prefill_shard_paths_) {
    PrefillShardPlan shard;
    shard.pte_path = shard_path;
    const bool rebuild_shard = prefill_shard_paths_.size() ==
            prefill_shard_index_paths_.size() &&
        !prefill_shard_index_paths_[prefill_shards_.size()].empty();
    if (rebuild_shard) {
      shard.index_path = prefill_shard_index_paths_[prefill_shards_.size()];
      shard.rebuild_on_execute = true;
    }
    materialize_prefill_shard(shard);
    auto method_names = shard.module->method_names();
    ET_CHECK_MSG(
        method_names.ok() && !method_names->empty(),
        "Failed to read shard method names: %s",
        shard_path.c_str());
    shard.method_name = *method_names->begin();
    ET_LOG(
        Info,
        "loading prefill shard: path=%s method=%s",
        shard_path.c_str(),
        shard.method_name.c_str());
    ET_CHECK_MSG(
        shard.module->load_method(shard.method_name) == Error::Ok,
        "Failed to load shard method %s from %s",
        shard.method_name.c_str(),
        shard_path.c_str());

    const auto meta = shard.module->method_meta(shard.method_name);
    ET_CHECK_MSG(meta.ok(), "Failed to read shard method meta: %s", shard_path.c_str());

    size_t shard_kv_inputs = 0;
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      auto input_meta = meta->input_tensor_meta(i);
      ET_CHECK_MSG(input_meta.ok(), "Expected tensor input for shard %s", shard_path.c_str());
      if (is_rank4_tensor(*input_meta)) {
        ++shard_kv_inputs;
      }
    }
    ET_CHECK_MSG(
        shard_kv_inputs % 2 == 0,
        "Shard %s has odd number of KV inputs",
        shard_path.c_str());
    shard.layer_count = shard_kv_inputs / 2;
    shard.layer_offset = layer_offset;
    layer_offset += shard.layer_count;

    size_t integral_seen = 0;
    size_t mask_seen = 0;
    size_t aux_seen = 0;
    size_t k_seen = 0;
    size_t v_seen = 0;
    for (size_t i = 0; i < meta->num_inputs(); ++i) {
      auto input_meta = meta->input_tensor_meta(i);
      ET_CHECK_MSG(input_meta.ok(), "Expected tensor input for shard %s", shard_path.c_str());
      if (is_rank4_tensor(*input_meta)) {
        if (looks_like_value_cache(*input_meta)) {
          shard.input_bindings.push_back(
              {PrefillShardPlan::InputKind::VCache, v_seen++});
        } else {
          shard.input_bindings.push_back(
              {PrefillShardPlan::InputKind::KCache, k_seen++});
        }
      } else if (
          is_rank2_integral_tensor(*input_meta) ||
          is_prompt_token_or_position_tensor(*input_meta, prompt_ar_len)) {
        shard.input_bindings.push_back(
            {integral_seen++ == 0 ? PrefillShardPlan::InputKind::Tokens
                                  : PrefillShardPlan::InputKind::Position,
             0});
      } else if (is_mask_tensor(*input_meta, prompt_ar_len, context_len)) {
        shard.input_bindings.push_back(
            {mask_seen++ == 0 ? PrefillShardPlan::InputKind::AttentionMask
                              : PrefillShardPlan::InputKind::WindowAttentionMask,
             0});
      } else if (is_rank2_float_tensor(*input_meta)) {
        shard.input_bindings.push_back(
            {PrefillShardPlan::InputKind::PreviousAux, aux_seen++});
      } else {
        shard.input_bindings.push_back(
            {PrefillShardPlan::InputKind::PreviousHidden, 0});
      }
    }

    size_t out_k_seen = 0;
    size_t out_v_seen = 0;
    size_t out_aux_seen = 0;
    const bool first_square_kv_is_v =
        first_square_kv_is_v_for_shard(shard.layer_offset);
    for (size_t i = 0; i < meta->num_outputs(); ++i) {
      auto output_meta = meta->output_tensor_meta(i);
      ET_CHECK_MSG(
          output_meta.ok(), "Expected tensor output for shard %s", shard_path.c_str());
      if (is_rank4_tensor(*output_meta)) {
        const bool square_kv = output_meta->sizes()[2] == output_meta->sizes()[3];
        const size_t square_kv_seen = out_k_seen + out_v_seen;
        const bool is_v_cache =
            square_kv ? (((square_kv_seen % 2) == 0) == first_square_kv_is_v)
                      : looks_like_value_cache(*output_meta);
        if (is_v_cache) {
          shard.output_bindings.push_back(
              {PrefillShardPlan::OutputKind::FinalVCache, out_v_seen++, 0});
        } else {
          shard.output_bindings.push_back(
              {PrefillShardPlan::OutputKind::FinalKCache, out_k_seen++, 0});
        }
      } else if (is_logits_tensor(*output_meta, vocab_size)) {
        shard.output_bindings.push_back(
            {PrefillShardPlan::OutputKind::FinalLogits, 0, 0});
      } else if (is_rank2_tensor(*output_meta)) {
        shard.owned_outputs.push_back(make_tensor_ptr_from_meta(*output_meta));
        shard.output_bindings.push_back(
            {PrefillShardPlan::OutputKind::IntermediateAux,
             out_aux_seen++,
             shard.owned_outputs.size() - 1});
      } else {
        shard.owned_outputs.push_back(make_tensor_ptr_from_meta(*output_meta));
        shard.output_bindings.push_back(
            {PrefillShardPlan::OutputKind::IntermediateHidden,
             0,
            shard.owned_outputs.size() - 1});
      }
    }
    ET_LOG(
        Info,
        "configured prefill shard: path=%s layer_offset=%zu layer_count=%zu inputs=%zu outputs=%zu owned_outputs=%zu",
        shard_path.c_str(),
        shard.layer_offset,
        shard.layer_count,
        shard.input_bindings.size(),
        shard.output_bindings.size(),
        shard.owned_outputs.size());
    if (shard.rebuild_on_execute) {
      release_prefill_shard(shard);
    }
    prefill_shards_.push_back(std::move(shard));
  }

  ET_CHECK_MSG(
      layer_offset == static_cast<size_t>(prefill_num_layers_),
      "Prefill shard layers mismatch: expected=%lld got=%zu",
      static_cast<long long>(prefill_num_layers_),
      layer_offset);
}

void DecoderRunner::configure_qwen3_static_prefill_shards() {
  ET_CHECK_MSG(
      !prefill_shard_paths_.empty(),
      "Qwen3 static prefill shard plan requires shard paths");
  ET_CHECK_MSG(
      prefill_num_layers_ > 0 &&
          static_cast<size_t>(prefill_num_layers_) % prefill_shard_paths_.size() == 0,
      "Qwen3 static prefill shard plan requires evenly split layers: layers=%lld shards=%zu",
      static_cast<long long>(prefill_num_layers_),
      prefill_shard_paths_.size());
  if (!prefill_shard_index_paths_.empty()) {
    ET_CHECK_MSG(
        prefill_shard_index_paths_.size() == prefill_shard_paths_.size(),
        "Qwen3 static prefill shard stripped_pte_paths/index_bin_paths size mismatch: %zu vs %zu",
        prefill_shard_paths_.size(),
        prefill_shard_index_paths_.size());
  }

  const size_t layers_per_shard =
      static_cast<size_t>(prefill_num_layers_) / prefill_shard_paths_.size();
  const executorch::aten::ScalarType activation_type =
      executorch::aten::ScalarType::Float;
  ET_CHECK_MSG(
      prefill_static_aux_size_ > 0 && prefill_static_hidden_size_ > 0,
      "Invalid Qwen3 static prefill metadata: aux_size=%d hidden_size=%d",
      prefill_static_aux_size_,
      prefill_static_hidden_size_);

  size_t layer_offset = 0;
  for (size_t shard_index = 0; shard_index < prefill_shard_paths_.size(); ++shard_index) {
    PrefillShardPlan shard;
    shard.pte_path = prefill_shard_paths_[shard_index];
    shard.method_name = "prefill_forward";
    shard.layer_offset = layer_offset;
    shard.layer_count = layers_per_shard;
    if (!prefill_shard_index_paths_.empty() &&
        !prefill_shard_index_paths_[shard_index].empty()) {
      shard.index_path = prefill_shard_index_paths_[shard_index];
      shard.rebuild_on_execute = true;
    }

    if (shard_index == 0) {
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::Tokens, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::Position, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::AttentionMask, 0});
    } else {
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousAux, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousAux, 1});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousHidden, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::AttentionMask, 0});
    }
    for (size_t local_layer = 0; local_layer < shard.layer_count; ++local_layer) {
      shard.input_bindings.push_back(
          {PrefillShardPlan::InputKind::VCache, local_layer});
      shard.input_bindings.push_back(
          {PrefillShardPlan::InputKind::KCache, local_layer});
    }

    size_t owned_aux = 0;
    if (shard_index == 0) {
      shard.owned_outputs.push_back(make_tensor_ptr_from_sizes(
          {prefill_prompt_ar_len_, prefill_static_aux_size_}, activation_type));
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::IntermediateAux, 0, owned_aux++});
      shard.owned_outputs.push_back(make_tensor_ptr_from_sizes(
          {prefill_prompt_ar_len_, prefill_static_aux_size_}, activation_type));
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::IntermediateAux, 1, owned_aux++});
    }
    for (size_t local_layer = 0; local_layer < shard.layer_count; ++local_layer) {
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::FinalVCache, local_layer, 0});
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::FinalKCache, local_layer, 0});
    }
    if (shard_index + 1 == prefill_shard_paths_.size()) {
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::FinalLogits, 0, 0});
    } else {
      shard.owned_outputs.push_back(make_tensor_ptr_from_sizes(
          {1, prefill_prompt_ar_len_, prefill_static_hidden_size_}, activation_type));
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::IntermediateHidden,
           0,
           shard.owned_outputs.size() - 1});
    }

    ET_LOG(
        Info,
        "configured qwen3 static prefill shard: path=%s layer_offset=%zu layer_count=%zu inputs=%zu outputs=%zu owned_outputs=%zu rebuild_on_execute=%d",
        shard.pte_path.c_str(),
        shard.layer_offset,
        shard.layer_count,
        shard.input_bindings.size(),
        shard.output_bindings.size(),
        shard.owned_outputs.size(),
        static_cast<int>(shard.rebuild_on_execute));
    prefill_shards_.push_back(std::move(shard));
    layer_offset += layers_per_shard;
  }

  ET_CHECK_MSG(
      layer_offset == static_cast<size_t>(prefill_num_layers_),
      "Qwen3 static prefill shard layers mismatch: expected=%lld got=%zu",
      static_cast<long long>(prefill_num_layers_),
      layer_offset);
}

Error DecoderRunner::set_outputs(
    const std::string& method_name,
    std::vector<executorch::aten::Tensor> output_values) {
  if (method_name == "prefill_forward" && uses_prefill_shards()) {
    return set_outputs_prefill_shards(std::move(output_values));
  }
  for (size_t i = 0; i < output_values.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        module_->set_output(method_name, output_values[i], i));
  }
  return Error::Ok;
}

Error DecoderRunner::load(const std::vector<std::string>& method_names) {
  if (is_method_loaded(method_names)) {
    return Error::Ok;
  }
  for (const std::string& method_name : method_names) {
    if (method_name == "prefill_forward" && uses_prefill_shards()) {
      for (auto& shard : prefill_shards_) {
        if (shard.module != nullptr) {
          ET_CHECK_OK_OR_RETURN_ERROR(shard.module->load_method(shard.method_name));
        }
      }
      continue;
    }
    ET_CHECK_OK_OR_RETURN_ERROR(module_->load_method(method_name));
  }
  return Error::Ok;
}

bool DecoderRunner::is_method_loaded(
    const std::vector<std::string>& method_names) {
  bool method_loaded = true;
  for (const std::string& method_name : method_names) {
    if (method_name == "prefill_forward" && uses_prefill_shards()) {
      bool shard_methods_loaded = true;
      for (const auto& shard : prefill_shards_) {
        shard_methods_loaded &=
            shard.rebuild_on_execute ||
            (shard.module != nullptr && shard.module->is_method_loaded(shard.method_name));
      }
      method_loaded &= shard_methods_loaded;
      continue;
    }
    method_loaded &= module_->is_method_loaded(method_name);
  }
  return method_loaded;
}

// This function is functional, meaning it shouldn't modify any state of the
// input. It should be safe to call multiple times with the same inputs. The
// outer loop (call site) is responsible for managing state.
Result<Tensor> DecoderRunner::step(
    const std::string& method_name,
    std::vector<EValue>& inputs) {
  if (method_name == "prefill_forward" && uses_prefill_shards()) {
    return step_prefill_shards(inputs);
  }
  Result<std::vector<EValue>> outputs_res =
      module_->execute(method_name, inputs);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res.get()[0].isTensor(),
      "Non Tensor Output returned from executing LLM");

  // Return the logits tensor
  return outputs_res.get()[0].toTensor();
}

Error DecoderRunner::set_outputs_prefill_shards(
    std::vector<executorch::aten::Tensor> output_values) {
  ET_CHECK_MSG(
      !prefill_shards_.empty(),
      "Prefill shard outputs requested before shard configuration");
  prefill_output_values_ = std::move(output_values);
  const size_t k_out_base = 1;
  const size_t v_out_base = 1 + static_cast<size_t>(prefill_num_layers_);

  for (auto& shard : prefill_shards_) {
    shard.output_tensors.clear();
    shard.output_tensors.reserve(shard.output_bindings.size());
    for (const auto& binding : shard.output_bindings) {
      switch (binding.kind) {
        case PrefillShardPlan::OutputKind::FinalLogits:
          shard.output_tensors.push_back(prefill_output_values_[0]);
          break;
        case PrefillShardPlan::OutputKind::FinalKCache:
          shard.output_tensors.push_back(
              prefill_output_values_[k_out_base + shard.layer_offset + binding.index]);
          break;
        case PrefillShardPlan::OutputKind::FinalVCache:
          shard.output_tensors.push_back(
              prefill_output_values_[v_out_base + shard.layer_offset + binding.index]);
          break;
        case PrefillShardPlan::OutputKind::IntermediateAux:
        case PrefillShardPlan::OutputKind::IntermediateHidden:
          shard.output_tensors.push_back(*shard.owned_outputs[binding.owned_index]);
          break;
      }
    }
    if (shard.module != nullptr) {
      for (size_t i = 0; i < shard.output_tensors.size(); ++i) {
        ET_CHECK_OK_OR_RETURN_ERROR(
            shard.module->set_output(shard.method_name, shard.output_tensors[i], i));
      }
    }
  }
  return Error::Ok;
}

Result<Tensor> DecoderRunner::step_prefill_shards(std::vector<EValue>& inputs) {
  ET_CHECK_MSG(
      !prefill_output_values_.empty(),
      "Prefill shard outputs are not bound before execution");
  const size_t full_mask_index = 1;
  const size_t full_window_mask_index =
      prefill_has_window_attention_mask_ ? 2 : static_cast<size_t>(-1);
  const size_t full_pos_index = prefill_has_window_attention_mask_ ? 3 : 2;
  const size_t full_k_base = full_pos_index + 1;
  const size_t full_v_base = full_k_base + static_cast<size_t>(prefill_num_layers_);


  std::vector<Tensor> previous_aux;
  Tensor* previous_hidden = nullptr;
  for (auto& shard : prefill_shards_) {
    std::vector<EValue> shard_inputs;
    shard_inputs.reserve(shard.input_bindings.size());
    for (const auto& binding : shard.input_bindings) {
      switch (binding.kind) {
        case PrefillShardPlan::InputKind::Tokens:
          shard_inputs.emplace_back(inputs[0]);
          break;
        case PrefillShardPlan::InputKind::AttentionMask:
          shard_inputs.emplace_back(inputs[full_mask_index]);
          break;
        case PrefillShardPlan::InputKind::WindowAttentionMask:
          ET_CHECK_MSG(
              prefill_has_window_attention_mask_,
              "Shard requested window attention mask but runner does not have one");
          shard_inputs.emplace_back(inputs[full_window_mask_index]);
          break;
        case PrefillShardPlan::InputKind::Position:
          shard_inputs.emplace_back(inputs[full_pos_index]);
          break;
        case PrefillShardPlan::InputKind::KCache:
          shard_inputs.emplace_back(inputs[full_k_base + shard.layer_offset + binding.index]);
          break;
        case PrefillShardPlan::InputKind::VCache:
          shard_inputs.emplace_back(inputs[full_v_base + shard.layer_offset + binding.index]);
          break;
        case PrefillShardPlan::InputKind::PreviousAux:
          ET_CHECK_MSG(
              binding.index < previous_aux.size(),
              "Missing previous shard aux tensor %zu",
              binding.index);
          shard_inputs.emplace_back(previous_aux[binding.index]);
          break;
        case PrefillShardPlan::InputKind::PreviousHidden:
          ET_CHECK_MSG(previous_hidden != nullptr, "Missing previous shard hidden state");
          shard_inputs.emplace_back(*previous_hidden);
          break;
      }
    }

    materialize_prefill_shard(shard);
    ET_CHECK_OK_OR_RETURN_ERROR(shard.module->load_method(shard.method_name));
    for (size_t i = 0; i < shard.output_tensors.size(); ++i) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          shard.module->set_output(shard.method_name, shard.output_tensors[i], i));
    }

    auto outputs_res = shard.module->execute(shard.method_name, shard_inputs);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res->size() == shard.output_tensors.size(),
        "Shard output count mismatch: returned=%zu bound=%zu",
        outputs_res->size(),
        shard.output_tensors.size());
    for (size_t i = 0; i < outputs_res->size(); ++i) {
      ET_CHECK_MSG(
          outputs_res.get()[i].isTensor(),
          "Non Tensor Output returned from prefill shard");
      copy_tensor_data(outputs_res.get()[i].toTensor(), shard.output_tensors[i]);
    }

    std::vector<Tensor> next_aux;
    previous_hidden = nullptr;
    for (const auto& binding : shard.output_bindings) {
      if (binding.kind == PrefillShardPlan::OutputKind::IntermediateAux) {
        next_aux.push_back(*shard.owned_outputs[binding.owned_index]);
      } else if (
          binding.kind == PrefillShardPlan::OutputKind::IntermediateHidden) {
        previous_hidden = shard.owned_outputs[binding.owned_index].get();
      }
    }
    if (!next_aux.empty()) {
      if (prefill_shard_swap_aux() && next_aux.size() == 2) {
        std::swap(next_aux[0], next_aux[1]);
      }
      previous_aux = std::move(next_aux);
    }
    release_prefill_shard(shard);
  }
  return prefill_output_values_[0];
}

bool DecoderRunner::uses_prefill_shards() const {
  return !prefill_shards_.empty();
}

} // namespace example
