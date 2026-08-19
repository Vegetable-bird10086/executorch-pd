/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/prompt_processor.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <type_traits>
using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
using executorch::runtime::Span;
using executorch::runtime::TensorInfo;
namespace example {

template <typename T>
int64_t PromptProcessor<T>::prepare_logits_row_for_sampling(
    int64_t logits_pos) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    if (metadata_.ar_len > 1 &&
        std::getenv("ET_QNN_A8_VOCAB_MAJOR_LOGITS") != nullptr) {
      std::vector<T> selected(static_cast<size_t>(metadata_.vocab_size));
      for (int32_t token = 0; token < metadata_.vocab_size; ++token) {
        selected[static_cast<size_t>(token)] =
            logits_.data[static_cast<size_t>(token) * metadata_.ar_len +
                         logits_pos];
      }
      std::memcpy(
          logits_.data +
              static_cast<size_t>(logits_pos) * metadata_.vocab_size,
          selected.data(),
          selected.size() * sizeof(T));
    }
  }
  return logits_pos;
}

template <typename T>
PromptProcessor<T>::PromptProcessor(
    DecoderRunner* decoder_runner,
    KVManager<T>* kv_manager,
    const std::string& method_name,
    Metadata metadata)
    : decoder_runner_(decoder_runner),
      kv_manager_(kv_manager),
      method_name_(method_name),
      metadata_(metadata) {
  k_cache_in_.resize(metadata_.num_layers);
  v_cache_in_.resize(metadata_.num_layers);
  k_cache_out_.resize(metadata_.num_layers);
  v_cache_out_.resize(metadata_.num_layers);
  // Calculate I/O size
  if (metadata_.use_separate_embed) {
    ET_CHECK_MSG(
        metadata_.separate_embedding != nullptr && metadata_.embedding_dim > 0 &&
            metadata_.embedding_row_bytes > 0,
        "Invalid separate embedding metadata");
    input_embedding_.size =
        static_cast<size_t>(metadata_.ar_len) * metadata_.embedding_row_bytes;
  } else {
    input_toks_.size = metadata_.ar_len * sizeof(int64_t);
  }
  if (is_bert())
    input_pos_.size = 0;
  else
    input_pos_.size = metadata_.ar_len * sizeof(int32_t);

  switch (metadata_.cache_mode) {
    case CacheMode::StaticCahce:
      attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size = 0;
      break;
    case CacheMode::HybridCache:
      attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      window_attention_mask_.size =
          metadata_.ar_len * metadata_.context_len * sizeof(uint16_t);
      break;
    default:
      ET_CHECK_MSG(false, "Unsupported llama cache mode");
      break;
  }

  if (metadata_.outputs_logits) {
    logits_.size = metadata_.ar_len * metadata_.vocab_size * sizeof(T);
  }
};
template <typename T>
void PromptProcessor<T>::init_io(
    IMemAlloc* buffer_manager,
    Result<MethodMeta> method_meta) {
  size_t idx = 0;
  input_tensors_.reserve(method_meta->num_inputs());
  output_tensors_.reserve(method_meta->num_outputs());
  // [I]: token ids or externally supplied hidden states.
  Result<TensorInfo> input_toks = method_meta->input_tensor_meta(idx++);
  if (metadata_.use_separate_embed) {
    ET_CHECK_MSG(
        input_toks->sizes().size() == 3 &&
            input_toks->sizes()[2] == metadata_.embedding_dim,
        "Separate embedding input shape does not match exported decoder");
    input_embedding_.data = reinterpret_cast<uint8_t*>(
        buffer_manager->allocate(input_embedding_.size));
    input_embedding_.tensor = std::make_unique<TensorImpl>(
        input_toks->scalar_type(),
        input_toks->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
        input_embedding_.data,
        const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
    input_tensors_.emplace_back(input_embedding_.tensor.get());
    buffer_manager->add_memory_info(
        input_embedding_.data, input_embedding_.size, input_toks.get());
  } else {
    input_toks_.data =
        reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
    input_toks_.tensor = std::make_unique<TensorImpl>(
        input_toks->scalar_type(),
        input_toks->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_toks->sizes().data()),
        input_toks_.data,
        const_cast<TensorImpl::DimOrderType*>(input_toks->dim_order().data()));
    input_tensors_.emplace_back(input_toks_.tensor.get());
    buffer_manager->add_memory_info(
        input_toks_.data, input_toks_.size, input_toks.get());
  }

  // [I]: attention_mask
  Result<TensorInfo> attention_mask = method_meta->input_tensor_meta(idx++);
  attention_mask_.data = reinterpret_cast<uint16_t*>(
      buffer_manager->allocate(attention_mask_.size));
  attention_mask_.tensor = std::make_unique<TensorImpl>(
      attention_mask->scalar_type(),
      attention_mask->sizes().size(),
      const_cast<TensorImpl::SizesType*>(attention_mask->sizes().data()),
      attention_mask_.data,
      const_cast<TensorImpl::DimOrderType*>(
          attention_mask->dim_order().data()));
  input_tensors_.emplace_back(attention_mask_.tensor.get());
  buffer_manager->add_memory_info(
      attention_mask_.data, attention_mask_.size, attention_mask.get());

  // [I]: sliding window attention_mask
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    Result<TensorInfo> window_attention_mask =
        method_meta->input_tensor_meta(idx++);
    window_attention_mask_.data = reinterpret_cast<uint16_t*>(
        buffer_manager->allocate(window_attention_mask_.size));
    window_attention_mask_.tensor = std::make_unique<TensorImpl>(
        window_attention_mask->scalar_type(),
        window_attention_mask->sizes().size(),
        const_cast<TensorImpl::SizesType*>(
            window_attention_mask->sizes().data()),
        window_attention_mask_.data,
        const_cast<TensorImpl::DimOrderType*>(
            window_attention_mask->dim_order().data()));
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
    buffer_manager->add_memory_info(
        window_attention_mask_.data,
        window_attention_mask_.size,
        window_attention_mask.get());
  }

  if (!is_bert()) {
    // [I]: input_pos
    Result<TensorInfo> input_pos = method_meta->input_tensor_meta(idx++);
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = std::make_unique<TensorImpl>(
        input_pos->scalar_type(),
        input_pos->sizes().size(),
        const_cast<TensorImpl::SizesType*>(input_pos->sizes().data()),
        input_pos_.data,
        const_cast<TensorImpl::DimOrderType*>(input_pos->dim_order().data()));
    input_tensors_.emplace_back(input_pos_.tensor.get());
    buffer_manager->add_memory_info(
        input_pos_.data, input_pos_.size, input_pos.get());

    // [I] kv_cache
    // Prepare the vector of EValue for kv cache to evict token
    cache_inputs_.reserve(2 * metadata_.num_layers);
    size_t index = idx; // bypass input_tokens, atten_mask, input_pos
    for (int cache_group = 0; cache_group < 2; ++cache_group) {
      std::vector<std::unique_ptr<TensorImpl>>& cache =
          (cache_group == 0 ? k_cache_in_ : v_cache_in_);
      std::vector<KVCache<T>> cache_ptrs = (cache_group == 0)
          ? kv_manager_->get_k_cache_()
          : kv_manager_->get_v_cache_();
      for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
        Result<TensorInfo> kv_cache = method_meta->input_tensor_meta(index);

        T* cache_ptr = cache_ptrs[layer].buffer;

        cache[layer] = std::make_unique<TensorImpl>(
            kv_cache->scalar_type(),
            kv_cache->sizes().size(),
            const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
            cache_ptr,
            const_cast<TensorImpl::DimOrderType*>(
                kv_cache->dim_order().data()));
        input_tensors_.emplace_back(cache[layer].get());
        cache_inputs_.emplace_back(input_tensors_.back());
        buffer_manager->add_memory_info(
            cache_ptr, cache[layer]->nbytes(), kv_cache.get());
      }
    }
  }

  size_t index = 0;
  if (metadata_.outputs_logits) {
    // [O]: logits
    Result<TensorInfo> logits = method_meta->output_tensor_meta(index++);
    logits_.data = reinterpret_cast<T*>(buffer_manager->allocate(logits_.size));
    logits_.tensor = std::make_unique<TensorImpl>(
        logits->scalar_type(),
        logits->sizes().size(),
        const_cast<TensorImpl::SizesType*>(logits->sizes().data()),
        logits_.data,
        const_cast<TensorImpl::DimOrderType*>(logits->dim_order().data()));
    output_tensors_.emplace_back(logits_.tensor.get());
    buffer_manager->add_memory_info(logits_.data, logits_.size, logits.get());
  }

  // [O] kv_cache
  for (int cache_group = 0; cache_group < 2; ++cache_group) {
    std::vector<std::unique_ptr<TensorImpl>>& cache =
        (cache_group == 0 ? k_cache_out_ : v_cache_out_);
    std::vector<KVCache<T>> cache_ptrs = (cache_group == 0)
        ? kv_manager_->get_k_cache_()
        : kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer, ++index) {
      Result<TensorInfo> kv_cache = method_meta->output_tensor_meta(index);
      T* cache_ptr = cache_ptrs[layer].output_buffer;
      cache[layer] = std::make_unique<TensorImpl>(
          kv_cache->scalar_type(),
          kv_cache->sizes().size(),
          const_cast<TensorImpl::SizesType*>(kv_cache->sizes().data()),
          cache_ptr,
          const_cast<TensorImpl::DimOrderType*>(kv_cache->dim_order().data()));
      output_tensors_.emplace_back(cache[layer].get());
      buffer_manager->add_memory_info(
          cache_ptr, cache[layer]->nbytes(), kv_cache.get());
    }
  }
  // Prepare the vector of EValue to run inference
  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

template <typename T>
void PromptProcessor<T>::init_io_from_metadata(IMemAlloc* buffer_manager) {
  using ScalarType = executorch::aten::ScalarType;
  using SizesType = TensorImpl::SizesType;
  using DimOrderType = TensorImpl::DimOrderType;

  const int32_t cache_len = metadata_.ar_len == metadata_.context_len
      ? metadata_.context_len
      : metadata_.context_len - metadata_.ar_len;
  const ScalarType token_type =
      metadata_.use_int64_token ? ScalarType::Long : ScalarType::Int;
  // The Qwen3 static shard MethodMeta declares Float tensors, while the QNN
  // backend consumes and produces its native quantized payload through these
  // buffers.
  // This mirrors init_io(), which adopts MethodMeta scalar types without
  // resizing the existing native client allocations.
  const ScalarType qnn_io_type = ScalarType::Float;
  const ScalarType mask_type = qnn_io_type;
  const ScalarType pos_type = ScalarType::Int;
  const ScalarType logits_type = qnn_io_type;
  const ScalarType kv_type = qnn_io_type;

  const size_t num_inputs = 3 + static_cast<size_t>(metadata_.num_layers) * 2;
  const size_t num_outputs = (metadata_.outputs_logits ? 1 : 0) +
      static_cast<size_t>(metadata_.num_layers) * 2;
  input_tensors_.reserve(num_inputs);
  output_tensors_.reserve(num_outputs);
  synthetic_sizes_.reserve(num_inputs + num_outputs);
  synthetic_dim_orders_.reserve(num_inputs + num_outputs);

  auto make_tensor = [&](ScalarType scalar_type,
                         std::vector<SizesType> sizes,
                         void* data) {
    std::vector<DimOrderType> dim_order;
    dim_order.reserve(sizes.size());
    for (size_t i = 0; i < sizes.size(); ++i) {
      dim_order.push_back(static_cast<DimOrderType>(i));
    }
    synthetic_sizes_.push_back(std::move(sizes));
    synthetic_dim_orders_.push_back(std::move(dim_order));
    return std::make_unique<TensorImpl>(
        scalar_type,
        synthetic_sizes_.back().size(),
        synthetic_sizes_.back().data(),
        data,
        synthetic_dim_orders_.back().data());
  };

  if (metadata_.use_separate_embed) {
    input_embedding_.data = reinterpret_cast<uint8_t*>(
        buffer_manager->allocate(input_embedding_.size));
    input_embedding_.tensor = make_tensor(
        metadata_.embedding_scalar_type,
        {1,
         static_cast<SizesType>(metadata_.ar_len),
         static_cast<SizesType>(metadata_.embedding_dim)},
        input_embedding_.data);
    input_tensors_.emplace_back(input_embedding_.tensor.get());
  } else {
    input_toks_.data =
        reinterpret_cast<int64_t*>(buffer_manager->allocate(input_toks_.size));
    input_toks_.tensor = make_tensor(
        token_type, {1, static_cast<SizesType>(metadata_.ar_len)}, input_toks_.data);
    input_tensors_.emplace_back(input_toks_.tensor.get());
  }

  attention_mask_.data = reinterpret_cast<uint16_t*>(
      buffer_manager->allocate(attention_mask_.size));
  attention_mask_.tensor = make_tensor(
      mask_type,
      {1,
       static_cast<SizesType>(metadata_.ar_len),
       static_cast<SizesType>(metadata_.context_len)},
      attention_mask_.data);
  input_tensors_.emplace_back(attention_mask_.tensor.get());

  if (metadata_.cache_mode == CacheMode::HybridCache) {
    window_attention_mask_.data = reinterpret_cast<uint16_t*>(
        buffer_manager->allocate(window_attention_mask_.size));
    window_attention_mask_.tensor = make_tensor(
        mask_type,
        {1,
         static_cast<SizesType>(metadata_.ar_len),
         static_cast<SizesType>(metadata_.context_len)},
        window_attention_mask_.data);
    input_tensors_.emplace_back(window_attention_mask_.tensor.get());
  }

  if (!is_bert()) {
    input_pos_.data =
        reinterpret_cast<int32_t*>(buffer_manager->allocate(input_pos_.size));
    input_pos_.tensor = make_tensor(
        pos_type, {1, static_cast<SizesType>(metadata_.ar_len)}, input_pos_.data);
    input_tensors_.emplace_back(input_pos_.tensor.get());

    cache_inputs_.reserve(2 * metadata_.num_layers);
    auto k_cache_ptrs = kv_manager_->get_k_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      T* cache_ptr = k_cache_ptrs[layer].buffer;
      k_cache_in_[layer] = make_tensor(
          kv_type,
          {1,
           static_cast<SizesType>(metadata_.num_heads),
           static_cast<SizesType>(kv_manager_->get_head_dim()),
           static_cast<SizesType>(cache_len)},
          cache_ptr);
      input_tensors_.emplace_back(k_cache_in_[layer].get());
      cache_inputs_.emplace_back(input_tensors_.back());
    }
    auto v_cache_ptrs = kv_manager_->get_v_cache_();
    for (int layer = 0; layer < metadata_.num_layers; ++layer) {
      T* cache_ptr = v_cache_ptrs[layer].buffer;
      v_cache_in_[layer] = make_tensor(
          kv_type,
          {1,
           static_cast<SizesType>(metadata_.num_heads),
           static_cast<SizesType>(cache_len),
           static_cast<SizesType>(kv_manager_->get_head_dim())},
          cache_ptr);
      input_tensors_.emplace_back(v_cache_in_[layer].get());
      cache_inputs_.emplace_back(input_tensors_.back());
    }
  }

  if (metadata_.outputs_logits) {
    logits_.data = reinterpret_cast<T*>(buffer_manager->allocate(logits_.size));
    logits_.tensor = make_tensor(
        logits_type,
        {1,
         static_cast<SizesType>(metadata_.ar_len),
         static_cast<SizesType>(metadata_.vocab_size)},
        logits_.data);
    output_tensors_.emplace_back(logits_.tensor.get());
  }

  auto k_cache_ptrs = kv_manager_->get_k_cache_();
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    T* cache_ptr = k_cache_ptrs[layer].output_buffer;
    k_cache_out_[layer] = make_tensor(
        kv_type,
        {1,
         static_cast<SizesType>(metadata_.num_heads),
         static_cast<SizesType>(kv_manager_->get_head_dim()),
         static_cast<SizesType>(metadata_.ar_len)},
        cache_ptr);
    output_tensors_.emplace_back(k_cache_out_[layer].get());
  }
  auto v_cache_ptrs = kv_manager_->get_v_cache_();
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    T* cache_ptr = v_cache_ptrs[layer].output_buffer;
    v_cache_out_[layer] = make_tensor(
        kv_type,
        {1,
         static_cast<SizesType>(metadata_.num_heads),
         static_cast<SizesType>(metadata_.ar_len),
         static_cast<SizesType>(kv_manager_->get_head_dim())},
        cache_ptr);
    output_tensors_.emplace_back(v_cache_out_[layer].get());
  }

  inputs_.reserve(input_tensors_.size());
  for (auto& input_tensor : input_tensors_) {
    inputs_.emplace_back(std::move(input_tensor));
  }
}

template <typename T>
const std::vector<uint16_t>& PromptProcessor<T>::get_all_logits() {
  return prompt_all_logits_;
}

template <typename T>
void PromptProcessor<T>::prepare_prompt_embeddings(
    const std::vector<uint64_t>& prompt_tokens) {
  clear_prompt_embeddings();
  if (!metadata_.use_separate_embed) {
    return;
  }
  ET_CHECK_MSG(
      metadata_.separate_embedding != nullptr && metadata_.embedding_dim > 0,
      "Separate embedding instance is required");
  const size_t embedding_dim = static_cast<size_t>(metadata_.embedding_dim);
  ET_CHECK_MSG(
      prompt_tokens.size() <=
          std::numeric_limits<size_t>::max() / embedding_dim,
      "Prompt embedding cache size overflow");
  prompt_embeddings_.resize(prompt_tokens.size() * embedding_dim);
  for (size_t token_index = 0; token_index < prompt_tokens.size();
       ++token_index) {
    metadata_.separate_embedding->copy_row_to_float(
        prompt_tokens[token_index],
        prompt_embeddings_.data() + token_index * embedding_dim,
        embedding_dim);
  }
}

template <typename T>
void PromptProcessor<T>::clear_prompt_embeddings() {
  std::vector<float>().swap(prompt_embeddings_);
}

template <typename T>
void PromptProcessor<T>::prepare_io(
    const std::vector<uint64_t>& prompt_tokens,
    int64_t prompt_pos,
    int64_t start_pos,
    bool prepare_embedding) {
  const size_t embedding_dim = static_cast<size_t>(metadata_.embedding_dim);
  const bool use_cached_embeddings =
      metadata_.use_separate_embed && prepare_embedding &&
      prompt_embeddings_.size() == prompt_tokens.size() * embedding_dim;
  if (metadata_.use_separate_embed && prepare_embedding) {
    ET_CHECK_MSG(
        metadata_.separate_embedding != nullptr,
        "Separate embedding instance is required");
    std::memset(input_embedding_.data, 0, input_embedding_.size);
    if (use_cached_embeddings &&
        prompt_pos < static_cast<int64_t>(prompt_tokens.size())) {
      const size_t valid_rows = std::min<size_t>(
          static_cast<size_t>(metadata_.ar_len),
          prompt_tokens.size() - static_cast<size_t>(prompt_pos));
      std::memcpy(
          input_embedding_.data,
          prompt_embeddings_.data() +
              static_cast<size_t>(prompt_pos) * embedding_dim,
          valid_rows * embedding_dim * sizeof(float));
    }
  }
  for (int i = 0; i < metadata_.ar_len; i++) {
    if (!is_bert()) {
      input_pos_.data[i] = start_pos + i;
    }
    if (prompt_pos + i >= static_cast<int64_t>(prompt_tokens.size())) {
      continue;
    }
    const uint64_t token = prompt_tokens[prompt_pos + i];
    if (metadata_.use_separate_embed) {
      if (prepare_embedding && !use_cached_embeddings) {
        metadata_.separate_embedding->copy_row_to_float(
            token,
            reinterpret_cast<float*>(input_embedding_.data) +
                static_cast<size_t>(i) * metadata_.embedding_dim,
            metadata_.embedding_dim);
      }
    } else if (metadata_.use_int64_token) {
      input_toks_.data[i] = token;
    } else {
      int32_t* input_toks_ptr = reinterpret_cast<int32_t*>(input_toks_.data);
      input_toks_ptr[i] = static_cast<int32_t>(token);
    }
  }
  if (metadata_.use_separate_embed && prepare_embedding &&
      metadata_.embedding_qnn_u16_input) {
    ET_CHECK_MSG(
        metadata_.embedding_qnn_u16_scale > 0.0f,
        "Folded QNN U16 embedding input requires a positive scale");
    const size_t elements =
        static_cast<size_t>(metadata_.ar_len) * embedding_dim;
    float* source = reinterpret_cast<float*>(input_embedding_.data);
    uint16_t* destination = reinterpret_cast<uint16_t*>(input_embedding_.data);
    const float inverse_scale = 1.0f / metadata_.embedding_qnn_u16_scale;
    for (size_t i = 0; i < elements; ++i) {
      const long code = std::lround(source[i] * inverse_scale) +
          metadata_.embedding_qnn_u16_zero_point;
      destination[i] = static_cast<uint16_t>(
          std::min<long>(65535, std::max<long>(0, code)));
    }
  }
}

template <typename T>
void PromptProcessor<T>::acquire_prefill_kv_slot_and_rebind(
    int32_t layer_begin,
    int32_t layer_end_exclusive) {
  ET_CHECK_MSG(
      kv_manager_->uses_elastic_prefill_slots(),
      "Elastic Prefill KV rebinding requested while the pool is disabled");
  kv_manager_->acquire_prefill_kv_slot(layer_begin, layer_end_exclusive);
  const auto& k_cache = kv_manager_->get_k_cache_();
  const auto& v_cache = kv_manager_->get_v_cache_();
  for (int32_t layer = layer_begin; layer < layer_end_exclusive; ++layer) {
    ET_CHECK_MSG(
        k_cache_in_.at(static_cast<size_t>(layer)) != nullptr &&
            k_cache_out_.at(static_cast<size_t>(layer)) != nullptr &&
            v_cache_in_.at(static_cast<size_t>(layer)) != nullptr &&
            v_cache_out_.at(static_cast<size_t>(layer)) != nullptr,
        "Elastic Prefill KV tensors are not initialized for layer %d",
        layer);
    k_cache_in_[static_cast<size_t>(layer)]->set_data(
        k_cache[static_cast<size_t>(layer)].buffer);
    k_cache_out_[static_cast<size_t>(layer)]->set_data(
        k_cache[static_cast<size_t>(layer)].output_buffer);
    v_cache_in_[static_cast<size_t>(layer)]->set_data(
        v_cache[static_cast<size_t>(layer)].buffer);
    v_cache_out_[static_cast<size_t>(layer)]->set_data(
        v_cache[static_cast<size_t>(layer)].output_buffer);
  }
}

template <typename T>
void PromptProcessor<T>::clear_all_logits() {
  prompt_all_logits_.clear();
}

template <typename T>
void PromptProcessor<T>::reserve_all_logits(size_t elements) {
  prompt_all_logits_.reserve(elements);
}

template <typename T>
Result<uint64_t> PromptProcessor<T>::prefill(
    std::vector<uint64_t> prompt_tokens,
    int64_t start_pos,
    bool dump_logits,
    AttentionSinkRopeRunner* attention_sink_rope_runner,
    bool force_greedy_argmax) {
  ET_CHECK_MSG(!prompt_tokens.empty(), "Prompt cannot be null");

  if (const char* token_dump_path = std::getenv("ET_PROMPT_TOKEN_DUMP_PATH")) {
    std::ofstream token_dump(token_dump_path, std::ios::trunc);
    if (!token_dump.is_open()) {
      ET_LOG(Error, "Unable to open prompt token dump path=%s", token_dump_path);
    } else {
      for (uint64_t token : prompt_tokens) {
        token_dump << token << static_cast<char>(10);
      }
      ET_LOG(
          Info,
          "Wrote %zu QNN prompt tokens to %s",
          prompt_tokens.size(),
          token_dump_path);
    }
  }

  int64_t shifted_pos = start_pos;
  bool enable_attention_sink = attention_sink_rope_runner != nullptr;

  // Calculate number of blocks
  int32_t num_prompt_tokens = prompt_tokens.size();
  if (is_bert()) {
    ET_CHECK_MSG(
        start_pos == 0, "Bert model doesn't support multi-turn conversation.");
  } else if (!enable_attention_sink) {
    ET_CHECK_MSG(
        (start_pos + num_prompt_tokens) <=
            (metadata_.context_len - metadata_.ar_len),
        "The sequence length exceeds the maximum limit that the prompt processor can handle.");
  }

  // store the token
  int64_t cur_token;
  int64_t prompt_pos = 0;
  int32_t n_update = metadata_.ar_len;
  int num_iters = 1 + ((num_prompt_tokens - 1) / metadata_.ar_len);
  ET_LOG(
      Info,
      "Prompt Processor: total %d prompt tokens (AR-%d * %d iters)",
      num_prompt_tokens,
      metadata_.ar_len,
      num_iters);

  // Initialize attention sink rope runner if given and update position
  // accordingly
  if (enable_attention_sink) {
    ET_CHECK_MSG(
        attention_sink_rope_runner->set_outputs(method_name_, cache_inputs_) ==
            executorch::runtime::Error::Ok,
        "Failed to set output tensor for module %s",
        method_name_.c_str());
    shifted_pos =
        shifted_pos - attention_sink_rope_runner->get_position_shift();
  }

  // Rearrange KV cache first
  kv_manager_->rearrange_cache(metadata_.ar_len);
  std::vector<int32_t> attention_map(metadata_.ar_len);
  std::iota(attention_map.begin(), attention_map.end(), -1);
  // Initialize attention mask with current position
  kv_manager_->init_attention_mask(
      attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
  // Initialize window attention mask with current position
  if (metadata_.cache_mode == CacheMode::HybridCache) {
    kv_manager_->init_attention_mask(
        window_attention_mask_.data,
        attention_map,
        metadata_.ar_len,
        shifted_pos,
        metadata_.sliding_window);
  }

  // Initialize the output of the module
  ET_CHECK_MSG(
      decoder_runner_->set_outputs(method_name_, output_tensors_) ==
          executorch::runtime::Error::Ok,
      "Failed to set output tensor for module %s",
      method_name_.c_str());


  if (decoder_runner_->uses_prefill_shard_stage_major()) {
    ET_CHECK_MSG(
        !enable_attention_sink,
        "Stage-major sharded prefill does not support attention-sink eviction");
    const size_t shard_count = decoder_runner_->prefill_shard_count();
    ET_CHECK_MSG(shard_count > 0, "Stage-major prefill has no shards");
    std::vector<DecoderRunner::PrefillShardStageState> stage_states(
        static_cast<size_t>(num_iters));
    ET_LOG(
        Info,
        "Prompt Processor: stage-major sharded prefill shards=%zu iters=%d",
        shard_count,
        num_iters);

    bool final_overlap_prepared = false;
    for (size_t shard_index = 0; shard_index < shard_count; ++shard_index) {
      if (shard_index + 1 == shard_count) {
        final_overlap_prepared =
            decoder_runner_->prepare_final_prefill_shard_overlap();
      }
      const int32_t layer_begin = static_cast<int32_t>(
          decoder_runner_->prefill_shard_layer_offset(shard_index));
      const int32_t layer_end_exclusive = layer_begin + static_cast<int32_t>(
          decoder_runner_->prefill_shard_layer_count(shard_index));
      if (kv_manager_->uses_elastic_prefill_slots()) {
        acquire_prefill_kv_slot_and_rebind(
            layer_begin, layer_end_exclusive);
      }
      ET_CHECK_OK_OR_RETURN_ERROR(
          decoder_runner_->begin_prefill_shard_stage(shard_index));

      for (int i = 0; i < num_iters; ++i) {
        const int64_t stage_prompt_pos = static_cast<int64_t>(i) * metadata_.ar_len;
        const int64_t stage_pos = shifted_pos + stage_prompt_pos;
        kv_manager_->init_attention_mask(
            attention_mask_.data, attention_map, metadata_.ar_len, stage_pos);
        if (metadata_.cache_mode == CacheMode::HybridCache) {
          kv_manager_->init_attention_mask(
              window_attention_mask_.data,
              attention_map,
              metadata_.ar_len,
              stage_pos,
              metadata_.sliding_window);
        }
        prepare_io(
            prompt_tokens,
            stage_prompt_pos,
            stage_pos,
            shard_index == 0);

        const DecoderRunner::PrefillShardStageState* previous_stage =
            shard_index == 0 ? nullptr : &stage_states[static_cast<size_t>(i)];
        auto stage_result = decoder_runner_->step_prefill_shard_stage(
            shard_index, inputs_, previous_stage);
        ET_CHECK_OK_OR_RETURN_ERROR(stage_result.error());
        stage_states[static_cast<size_t>(i)] = stage_result.get();

        const int32_t stage_n_update = i == num_iters - 1
            ? 1 + ((num_prompt_tokens - 1) % metadata_.ar_len)
            : metadata_.ar_len;
        kv_manager_->update_cache_range(
            metadata_.ar_len,
            static_cast<int32_t>(stage_pos),
            stage_n_update,
            {},
            layer_begin,
            layer_end_exclusive);

        if (metadata_.outputs_logits && dump_logits && shard_index + 1 == shard_count) {
          prompt_all_logits_.insert(
              prompt_all_logits_.end(),
              logits_.data,
              logits_.data + metadata_.ar_len * metadata_.vocab_size);
        }
      }
      ET_CHECK_OK_OR_RETURN_ERROR(
          decoder_runner_->end_prefill_shard_stage(shard_index));
    }
    if (!final_overlap_prepared) {
      ET_CHECK_MSG(
          decoder_runner_->prepare_final_prefill_shard_overlap(),
          "Unable to release final Prefill rebuild resources");
    }

    if (!metadata_.outputs_logits) {
      return 0;
    }
    const int64_t logits_pos =
        (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len;
    const int64_t sampling_logits_pos =
        prepare_logits_row_for_sampling(logits_pos);
    cur_token = force_greedy_argmax
        ? decoder_runner_->logits_to_argmax_token(
              output_tensors_[0], sampling_logits_pos)
        : decoder_runner_->logits_to_token(
              output_tensors_[0], sampling_logits_pos);
    return cur_token;
  }

  for (int i = 0; i < num_iters; ++i) {
    // The current position plus the future generated cache exceeds the cache
    // size, which means we need to remove eviction_batch_size key-value cache
    // entries to make room for new tokens.
    if (enable_attention_sink &&
        shifted_pos + metadata_.ar_len >
            metadata_.context_len - metadata_.ar_len) {
      attention_sink_rope_runner->evict_token(method_name_, cache_inputs_);
      shifted_pos =
          shifted_pos - attention_sink_rope_runner->get_eviction_batch_size();
      // Initialize attention mask with current position
      kv_manager_->init_attention_mask(
          attention_mask_.data, attention_map, metadata_.ar_len, shifted_pos);
      // Initialize window attention mask with current position
      if (metadata_.cache_mode == CacheMode::HybridCache) {
        kv_manager_->init_attention_mask(
            window_attention_mask_.data,
            attention_map,
            metadata_.ar_len,
            shifted_pos,
            metadata_.sliding_window);
      }
    }

    // Fill in the token and position data
    prepare_io(prompt_tokens, prompt_pos, shifted_pos);

    // Run inference
    auto step_result = decoder_runner_->step(method_name_, inputs_);
    ET_CHECK_OK_OR_RETURN_ERROR(step_result.error());
    if (metadata_.outputs_logits && dump_logits) {
      prompt_all_logits_.insert(
          prompt_all_logits_.end(),
          logits_.data,
          logits_.data + metadata_.ar_len * metadata_.vocab_size);
    }
    // In the last run, offset to the meaningful logits.
    if (i == num_iters - 1) {
      n_update = 1 + ((num_prompt_tokens - 1) % metadata_.ar_len);
    }
    // Update KV Cache with the output results
    kv_manager_->update_cache(metadata_.ar_len, shifted_pos, n_update, {});

    // Update attention mask with current position
    kv_manager_->update_attention_mask(
        attention_mask_.data, metadata_.ar_len, shifted_pos, n_update);
    if (metadata_.cache_mode == CacheMode::HybridCache) {
      kv_manager_->update_attention_mask(
          window_attention_mask_.data,
          metadata_.ar_len,
          shifted_pos,
          n_update,
          metadata_.sliding_window);
    }
    prompt_pos += metadata_.ar_len;
    shifted_pos += metadata_.ar_len;
  }

  if (!metadata_.outputs_logits) {
    return 0;
  }
  const int64_t logits_pos =
      (num_prompt_tokens + metadata_.ar_len - 1) % metadata_.ar_len;
  if (std::getenv("ET_A8_IO_DIAG") != nullptr) {
    const T* row = logits_.data + logits_pos * metadata_.vocab_size;
    T raw_min = std::numeric_limits<T>::max();
    T raw_max = std::numeric_limits<T>::min();
    std::array<bool, 1u << (sizeof(T) * 8)> seen{};
    size_t distinct = 0;
    int32_t raw_top1 = 0;
    for (int32_t i = 0; i < metadata_.vocab_size; ++i) {
      const T raw = row[i];
      raw_min = std::min(raw_min, raw);
      raw_max = std::max(raw_max, raw);
      if (!seen[static_cast<size_t>(raw)]) {
        seen[static_cast<size_t>(raw)] = true;
        ++distinct;
      }
      if (raw > row[raw_top1]) raw_top1 = i;
    }
    size_t mask_nonzero = 0;
    const size_t mask_numel =
        static_cast<size_t>(metadata_.ar_len) * metadata_.context_len;
    for (size_t i = 0; i < mask_numel; ++i)
      mask_nonzero += attention_mask_.data[i] != 0;
    ET_LOG(
        Info,
        "A8_IO_DIAG prefill logits_pos=%ld mask_nonzero=%zu mask_max=%u "
        "logits_min=%u logits_max=%u logits_distinct=%zu raw_top1=%d",
        logits_pos,
        mask_nonzero,
        static_cast<unsigned>(std::numeric_limits<T>::max()),
        static_cast<unsigned>(raw_min),
        static_cast<unsigned>(raw_max),
        distinct,
        raw_top1);
    if (metadata_.ar_len > 1) {
      T strided_min = std::numeric_limits<T>::max();
      T strided_max = std::numeric_limits<T>::min();
      std::array<bool, 1u << (sizeof(T) * 8)> strided_seen{};
      size_t strided_distinct = 0;
      int32_t strided_top1 = 0;
      for (int32_t i = 0; i < metadata_.vocab_size; ++i) {
        const T raw = logits_.data[
            static_cast<size_t>(i) * metadata_.ar_len + logits_pos];
        strided_min = std::min(strided_min, raw);
        strided_max = std::max(strided_max, raw);
        if (!strided_seen[static_cast<size_t>(raw)]) {
          strided_seen[static_cast<size_t>(raw)] = true;
          ++strided_distinct;
        }
        if (raw > logits_.data[
                      static_cast<size_t>(strided_top1) * metadata_.ar_len +
                      logits_pos]) {
          strided_top1 = i;
        }
      }
      ET_LOG(
          Info,
          "A8_IO_DIAG prefill_strided logits_pos=%ld logits_min=%u "
          "logits_max=%u logits_distinct=%zu raw_top1=%d",
          logits_pos,
          static_cast<unsigned>(strided_min),
          static_cast<unsigned>(strided_max),
          strided_distinct,
          strided_top1);
    }
  }
  const int64_t sampling_logits_pos =
      prepare_logits_row_for_sampling(logits_pos);
  cur_token = force_greedy_argmax
      ? decoder_runner_->logits_to_argmax_token(
            output_tensors_[0], sampling_logits_pos)
      : decoder_runner_->logits_to_token(
            output_tensors_[0], sampling_logits_pos);
  return cur_token;
}

// Explicit instantiations
template class PromptProcessor<uint16_t>;
template class PromptProcessor<uint8_t>;

} // namespace example
