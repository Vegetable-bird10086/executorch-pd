/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/kv_manager.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorch.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/assert.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <type_traits>
namespace example {
template <typename T>
KVManager<T>::KVManager(Metadata metadata) : metadata_(metadata) {
  k_cache_.resize(metadata_.num_layers);
  v_cache_.resize(metadata_.num_layers);

  // Calculate cache size. The elastic stage-major path owns its KV slots
  // outside the monolithic RpcMem arena so it can grow only when handoff lags.
  elastic_prefill_cache_in_bytes_ = static_cast<size_t>(metadata_.num_heads) *
      metadata_.head_dim * metadata_.max_cache_len * sizeof(T);
  elastic_prefill_cache_out_bytes_ = static_cast<size_t>(metadata_.num_heads) *
      metadata_.head_dim * metadata_.max_ar_len * sizeof(T);
  elastic_prefill_per_layer_bytes_ =
      2 * (elastic_prefill_cache_in_bytes_ + elastic_prefill_cache_out_bytes_);
  elastic_prefill_enabled_ = std::is_same_v<T, uint8_t> &&
      metadata_.elastic_prefill_slot_layers > 0 &&
      metadata_.elastic_prefill_initial_slots > 0;
  if (elastic_prefill_enabled_) {
    ET_CHECK_MSG(
        metadata_.num_layers % metadata_.elastic_prefill_slot_layers == 0,
        "Elastic Prefill KV slot layers must divide model layers: layers=%lld slot_layers=%d",
        static_cast<long long>(metadata_.num_layers),
        metadata_.elastic_prefill_slot_layers);
    elastic_prefill_slot_bytes_ = elastic_prefill_per_layer_bytes_ *
        static_cast<size_t>(metadata_.elastic_prefill_slot_layers);
    elastic_prefill_layer_slots_.assign(
        static_cast<size_t>(metadata_.num_layers), -1);
    total_cache_size_ = 0;
  } else {
    total_cache_size_ = elastic_prefill_per_layer_bytes_ *
        static_cast<size_t>(metadata_.num_layers);
  }

  if constexpr (std::is_same_v<T, uint8_t>) {
    const char* bridge_path = std::getenv("ET_A8_TOKEN_AXIS_KV_BRIDGE_PATH");
    if (bridge_path != nullptr && bridge_path[0] != '\0') {
      std::ifstream input(bridge_path, std::ios::binary);
      ET_CHECK_MSG(input.is_open(), "Unable to open A8 KV bridge: %s", bridge_path);
      auto read_exact = [&](void* dst, size_t size) {
        input.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(size));
        ET_CHECK_MSG(input.good(), "Truncated A8 KV bridge: %s", bridge_path);
      };
      char magic[8];
      uint32_t version = 0, layers = 0, period = 0;
      read_exact(magic, sizeof(magic));
      read_exact(&version, sizeof(version));
      read_exact(&layers, sizeof(layers));
      read_exact(&period, sizeof(period));
      ET_CHECK_MSG(
          std::memcmp(magic, "KVA8BR1", 7) == 0 && version == 1 &&
              layers == static_cast<uint32_t>(metadata_.num_layers) && period > 0,
          "Invalid A8 KV bridge header: %s",
          bridge_path);
      a8_axis_period_ = static_cast<int32_t>(period);
      a8_axis_bridge_.resize(layers);
      for (auto& layer : a8_axis_bridge_) {
        read_exact(&layer.target_k_scale, sizeof(float));
        read_exact(&layer.target_k_zero_point, sizeof(int32_t));
        layer.target_v_scales.resize(period);
        layer.target_v_zero_points.resize(period);
        read_exact(layer.target_v_scales.data(), period * sizeof(float));
        read_exact(layer.target_v_zero_points.data(), period * sizeof(int32_t));
        read_exact(&layer.decode_k_output_scale, sizeof(float));
        read_exact(&layer.decode_k_output_zero_point, sizeof(int32_t));
        read_exact(&layer.decode_v_output_scale, sizeof(float));
        read_exact(&layer.decode_v_output_zero_point, sizeof(int32_t));
      }
      ET_LOG(
          Info,
          "Loaded A8 token-axis KV bridge: path=%s layers=%u period=%u",
          bridge_path,
          layers,
          period);
    }
  }
};

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  init_attention_mask_impl(attention_mask, attention_map, ar_len, n_past);
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint8_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  init_attention_mask_impl(attention_mask, attention_map, ar_len, n_past);
}

template <typename T>
template <typename MaskT>
void KVManager<T>::init_attention_mask_impl(
    MaskT* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  constexpr MaskT neg_val = 0;
  constexpr MaskT pos_val = std::numeric_limits<MaskT>::max();
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  MaskT* past_ptr = attention_mask;
  MaskT* new_ptr = attention_mask + (metadata_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      MaskT* parent_ptr = attention_mask + pidx * metadata_.context_len;
      std::memcpy(
          past_ptr, parent_ptr, metadata_.context_len * sizeof(MaskT));
    }
    // Attend to itself
    new_ptr[i] = pos_val;
    past_ptr += metadata_.context_len;
    new_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint16_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  init_attention_mask_impl(
      attention_mask,
      attention_map,
      ar_len,
      n_past,
      sliding_window,
      position_offset);
}

template <typename T>
void KVManager<T>::init_attention_mask(
    uint8_t* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  init_attention_mask_impl(
      attention_mask,
      attention_map,
      ar_len,
      n_past,
      sliding_window,
      position_offset);
}

template <typename T>
template <typename MaskT>
void KVManager<T>::init_attention_mask_impl(
    MaskT* attention_mask,
    const std::vector<int32_t>& attention_map,
    int32_t ar_len,
    int32_t n_past,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  ET_CHECK_MSG(
      attention_map.size() <= ar_len,
      "The size of attention_map (%zu) doesn't match with ar_len (%d)",
      attention_map.size(),
      ar_len);
  constexpr MaskT neg_val = 0;
  constexpr MaskT pos_val = std::numeric_limits<MaskT>::max();
  // Clear the attention mask
  std::fill_n(attention_mask, ar_len * metadata_.context_len, neg_val);

  // SMART_MASK requires special handling of attention mask
  MaskT* past_ptr = attention_mask;
  MaskT* new_ptr = attention_mask + (metadata_.context_len - ar_len);
  // All inputs will necessarily attend to n_past and itself
  for (int i = 0; i < ar_len; i++) {
    // Iterate across ar_len
    if (attention_map[i] < 0) {
      // If negative, attend to only past tokens
      std::fill_n(past_ptr, n_past, pos_val);
    } else {
      // If positive, copy attention map from (relative to 0th input) parent
      // Parent token index
      const int32_t pidx = attention_map[i];
      MaskT* parent_ptr = attention_mask + pidx * metadata_.context_len;
      std::memcpy(
          past_ptr, parent_ptr, metadata_.context_len * sizeof(MaskT));
    }
    // Attend to itself
    new_ptr[i] = pos_val;

    // mask by limitation of sliding_window
    int32_t available_context_len = position_offset.empty()
        ? sliding_window - (i + 1) - n_past
        : sliding_window - (position_offset[i] + 1) - n_past;
    // if available_context_len is less than 0, it means we need to mask some
    // tokens in the past to avoid exceeding the sliding window
    if (available_context_len < 0) {
      std::fill_n(past_ptr, -available_context_len, neg_val);
    }

    past_ptr += metadata_.context_len;
    new_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  update_attention_mask_impl(attention_mask, ar_len, n_past, n_update);
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint8_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  update_attention_mask_impl(attention_mask, ar_len, n_past, n_update);
}

template <typename T>
template <typename MaskT>
void KVManager<T>::update_attention_mask_impl(
    MaskT* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update) {
  constexpr MaskT pos_val = std::numeric_limits<MaskT>::max();
  MaskT* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint16_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  update_attention_mask_impl(
      attention_mask,
      ar_len,
      n_past,
      n_update,
      sliding_window,
      position_offset);
}

template <typename T>
void KVManager<T>::update_attention_mask(
    uint8_t* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  update_attention_mask_impl(
      attention_mask,
      ar_len,
      n_past,
      n_update,
      sliding_window,
      position_offset);
}

template <typename T>
template <typename MaskT>
void KVManager<T>::update_attention_mask_impl(
    MaskT* attention_mask,
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    int32_t sliding_window,
    const std::vector<int32_t>& position_offset) {
  constexpr MaskT pos_val = std::numeric_limits<MaskT>::max();
  constexpr MaskT neg_val = 0;
  MaskT* cur_ptr = attention_mask;
  cur_ptr += n_past;

  for (int i = 0; i < ar_len; i++) {
    std::fill_n(cur_ptr, n_update, pos_val);
    int32_t available_cache_len = position_offset.empty()
        ? sliding_window - (i + 1)
        : sliding_window - (position_offset[i] + 1);
    if (n_past + n_update > available_cache_len) {
      std::fill_n(
          cur_ptr - n_past, n_past + n_update - available_cache_len, neg_val);
    }
    cur_ptr += metadata_.context_len;
  }
}

template <typename T>
KVManager<T>::~KVManager() {
  for (auto& slot : elastic_prefill_slots_) {
    if (slot.custom_mem != nullptr) {
      QnnExecuTorchFreeCustomMem(slot.custom_mem);
      slot.custom_mem = nullptr;
    }
  }
}

template <typename T>
size_t KVManager<T>::allocate_elastic_prefill_slot_locked() {
  ET_CHECK_MSG(elastic_prefill_enabled_, "Elastic Prefill KV pool is disabled");
  void* custom_mem = QnnExecuTorchAllocCustomMem(
      elastic_prefill_slot_bytes_,
      executorch::runtime::MemoryAllocator::kDefaultAlignment);
  ET_CHECK_MSG(
      custom_mem != nullptr,
      "Failed to allocate elastic Prefill KV slot: bytes=%zu",
      elastic_prefill_slot_bytes_);
  const size_t slot_index = elastic_prefill_slots_.size();
  elastic_prefill_slots_.push_back(
      ElasticPrefillSlot{custom_mem, false, -1, 0});
  elastic_prefill_peak_slots_ =
      std::max(elastic_prefill_peak_slots_, elastic_prefill_slots_.size());

  auto* base = static_cast<std::byte*>(custom_mem);
  for (int32_t local_layer = 0;
       local_layer < metadata_.elastic_prefill_slot_layers;
       ++local_layer) {
    std::byte* layer_base = base +
        static_cast<size_t>(local_layer) * elastic_prefill_per_layer_bytes_;
    void* k_in = layer_base;
    void* k_out = layer_base + elastic_prefill_cache_in_bytes_;
    void* v_in = layer_base + elastic_prefill_cache_in_bytes_ +
        elastic_prefill_cache_out_bytes_;
    void* v_out = layer_base + 2 * elastic_prefill_cache_in_bytes_ +
        elastic_prefill_cache_out_bytes_;
    QnnExecuTorchAddCustomMemTensorAddr(k_in, custom_mem);
    QnnExecuTorchAddCustomMemTensorAddr(k_out, custom_mem);
    QnnExecuTorchAddCustomMemTensorAddr(v_in, custom_mem);
    QnnExecuTorchAddCustomMemTensorAddr(v_out, custom_mem);
  }
  ET_LOG(
      Info,
      "elastic Prefill KV slot allocated: index=%zu bytes=%zu slots=%zu",
      slot_index,
      elastic_prefill_slot_bytes_,
      elastic_prefill_slots_.size());
  return slot_index;
}

template <typename T>
void KVManager<T>::bind_elastic_prefill_layer_locked(
    int32_t logical_layer,
    size_t slot_index,
    int32_t local_layer) {
  ET_CHECK_MSG(
      logical_layer >= 0 && logical_layer < metadata_.num_layers &&
          slot_index < elastic_prefill_slots_.size() && local_layer >= 0 &&
          local_layer < metadata_.elastic_prefill_slot_layers,
      "Invalid elastic Prefill KV binding: layer=%d slot=%zu local=%d",
      logical_layer,
      slot_index,
      local_layer);
  auto* layer_base = static_cast<std::byte*>(
      elastic_prefill_slots_[slot_index].custom_mem) +
      static_cast<size_t>(local_layer) * elastic_prefill_per_layer_bytes_;
  k_cache_[static_cast<size_t>(logical_layer)].buffer =
      reinterpret_cast<T*>(layer_base);
  k_cache_[static_cast<size_t>(logical_layer)].output_buffer =
      reinterpret_cast<T*>(layer_base + elastic_prefill_cache_in_bytes_);
  v_cache_[static_cast<size_t>(logical_layer)].buffer = reinterpret_cast<T*>(
      layer_base + elastic_prefill_cache_in_bytes_ +
      elastic_prefill_cache_out_bytes_);
  v_cache_[static_cast<size_t>(logical_layer)].output_buffer =
      reinterpret_cast<T*>(
          layer_base + 2 * elastic_prefill_cache_in_bytes_ +
          elastic_prefill_cache_out_bytes_);
  elastic_prefill_layer_slots_[static_cast<size_t>(logical_layer)] =
      static_cast<int32_t>(slot_index);
}

template <typename T>
void KVManager<T>::acquire_prefill_kv_slot(
    int32_t layer_begin,
    int32_t layer_end_exclusive) {
  ET_CHECK_MSG(elastic_prefill_enabled_, "Elastic Prefill KV pool is disabled");
  const int32_t layer_count = layer_end_exclusive - layer_begin;
  ET_CHECK_MSG(
      layer_begin >= 0 && layer_end_exclusive <= metadata_.num_layers &&
          layer_count > 0 &&
          layer_count <= metadata_.elastic_prefill_slot_layers,
      "Invalid elastic Prefill KV acquire range [%d,%d)",
      layer_begin,
      layer_end_exclusive);
  std::lock_guard<std::mutex> lock(elastic_prefill_mutex_);
  size_t slot_index = elastic_prefill_slots_.size();
  for (size_t i = 0; i < elastic_prefill_slots_.size(); ++i) {
    if (!elastic_prefill_slots_[i].in_use) {
      slot_index = i;
      break;
    }
  }
  const bool grew = slot_index == elastic_prefill_slots_.size();
  if (grew) {
    slot_index = allocate_elastic_prefill_slot_locked();
  }
  auto& slot = elastic_prefill_slots_[slot_index];
  slot.in_use = true;
  slot.layer_begin = layer_begin;
  slot.layer_count = layer_count;
  for (int32_t layer = layer_begin; layer < layer_end_exclusive; ++layer) {
    bind_elastic_prefill_layer_locked(layer, slot_index, layer - layer_begin);
  }
  ET_LOG(
      Info,
      "elastic Prefill KV slot acquired: slot=%zu layers=[%d,%d) grew=%d slots=%zu",
      slot_index,
      layer_begin,
      layer_end_exclusive,
      static_cast<int>(grew),
      elastic_prefill_slots_.size());
}

template <typename T>
void KVManager<T>::release_prefill_kv_slot(
    int32_t layer_begin,
    int32_t layer_end_exclusive) {
  ET_CHECK_MSG(elastic_prefill_enabled_, "Elastic Prefill KV pool is disabled");
  std::lock_guard<std::mutex> lock(elastic_prefill_mutex_);
  ET_CHECK_MSG(
      layer_begin >= 0 && layer_begin < layer_end_exclusive &&
          layer_end_exclusive <= metadata_.num_layers,
      "Invalid elastic Prefill KV release range [%d,%d)",
      layer_begin,
      layer_end_exclusive);
  const int32_t slot_index =
      elastic_prefill_layer_slots_.at(static_cast<size_t>(layer_begin));
  ET_CHECK_MSG(
      slot_index >= 0 &&
          static_cast<size_t>(slot_index) < elastic_prefill_slots_.size(),
      "Missing elastic Prefill KV slot for layer %d",
      layer_begin);
  auto& slot = elastic_prefill_slots_[static_cast<size_t>(slot_index)];
  ET_CHECK_MSG(
      slot.in_use && slot.layer_begin == layer_begin &&
          slot.layer_count == layer_end_exclusive - layer_begin,
      "Elastic Prefill KV slot release mismatch: slot=%d owner=[%d,%d) release=[%d,%d)",
      slot_index,
      slot.layer_begin,
      slot.layer_begin + slot.layer_count,
      layer_begin,
      layer_end_exclusive);
  slot.in_use = false;
  slot.layer_begin = -1;
  slot.layer_count = 0;
  ET_LOG(
      Info,
      "elastic Prefill KV slot released: slot=%d layers=[%d,%d) slots=%zu",
      slot_index,
      layer_begin,
      layer_end_exclusive,
      elastic_prefill_slots_.size());
}

template <typename T>
size_t KVManager<T>::elastic_prefill_slot_count() const {
  std::lock_guard<std::mutex> lock(elastic_prefill_mutex_);
  return elastic_prefill_slots_.size();
}

template <typename T>
size_t KVManager<T>::elastic_prefill_peak_slot_count() const {
  std::lock_guard<std::mutex> lock(elastic_prefill_mutex_);
  return elastic_prefill_peak_slots_;
}

template <typename T>
void KVManager<T>::init_cache(IMemAlloc* buffer_manager, int32_t ar_len) {
  cur_ar_len_ = ar_len;
  if (elastic_prefill_enabled_) {
    std::lock_guard<std::mutex> lock(elastic_prefill_mutex_);
    for (int32_t i = 0; i < metadata_.elastic_prefill_initial_slots; ++i) {
      allocate_elastic_prefill_slot_locked();
    }
    for (int32_t layer = 0; layer < metadata_.num_layers; ++layer) {
      const size_t slot_index = static_cast<size_t>(
          (layer / metadata_.elastic_prefill_slot_layers) %
          metadata_.elastic_prefill_initial_slots);
      bind_elastic_prefill_layer_locked(
          layer, slot_index, layer % metadata_.elastic_prefill_slot_layers);
    }
    ET_LOG(
        Info,
        "elastic Prefill KV pool ready: initial_slots=%d slot_layers=%d "
        "slot_bytes=%zu total_bytes=%zu legacy_bytes=%zu",
        metadata_.elastic_prefill_initial_slots,
        metadata_.elastic_prefill_slot_layers,
        elastic_prefill_slot_bytes_,
        elastic_prefill_slots_.size() * elastic_prefill_slot_bytes_,
        elastic_prefill_per_layer_bytes_ *
            static_cast<size_t>(metadata_.num_layers));
    return;
  }
  const size_t max_in_cache_block_in_bytes =
      metadata_.max_cache_len * sizeof(T);
  const size_t max_out_cache_block_in_bytes = metadata_.max_ar_len * sizeof(T);

  const size_t cache_in_bytes =
      metadata_.num_heads * metadata_.head_dim * max_in_cache_block_in_bytes;
  const size_t cache_out_bytes =
      metadata_.num_heads * metadata_.head_dim * max_out_cache_block_in_bytes;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    // Allocate buffer for key cache and value cache
    T* single_layer_k_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_k_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));
    T* single_layer_v_cache_in =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_in_bytes));
    T* single_layer_v_cache_out =
        reinterpret_cast<T*>(buffer_manager->allocate(cache_out_bytes));

    k_cache_[layer].buffer = single_layer_k_cache_in;
    k_cache_[layer].output_buffer = single_layer_k_cache_out;
    v_cache_[layer].buffer = single_layer_v_cache_in;
    v_cache_[layer].output_buffer = single_layer_v_cache_out;
  }
}

template <typename T>
void KVManager<T>::rearrange_cache(int32_t ar_len_dst) {
  // Don't need to rearrange if cur_ar_len_ is equal to target ar_len
  if (cur_ar_len_ == ar_len_dst)
    return;
  for (int layer = 0; layer < metadata_.num_layers; ++layer) {
    rearrange_key(k_cache_[layer], ar_len_dst);
    rearrange_value(v_cache_[layer], ar_len_dst);
  }
  // rearrange done.
  cur_ar_len_ = ar_len_dst;
}

template <typename T>
void KVManager<T>::rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* k_cache_in_read_ptr = k_cache.buffer;
  T* k_cache_in_write_ptr = k_cache.buffer;

  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, dst_cache_num * sizeof(T));
      k_cache_in_read_ptr += src_cache_num;
      k_cache_in_write_ptr += dst_cache_num;
    }
  } else {
    k_cache_in_read_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * src_cache_num;
    k_cache_in_write_ptr +=
        (metadata_.head_dim * metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.head_dim * metadata_.num_heads; i++) {
      std::memmove(
          k_cache_in_write_ptr, k_cache_in_read_ptr, src_cache_num * sizeof(T));
      k_cache_in_read_ptr -= src_cache_num;
      k_cache_in_write_ptr -= dst_cache_num;
    }
  }
}

template <typename T>
void KVManager<T>::rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst) {
  const int32_t src_cache_num = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t dst_cache_num = metadata_.context_len - ar_len_dst;
  T* v_cache_in_read_ptr = v_cache.buffer;
  T* v_cache_in_write_ptr = v_cache.buffer;
  if (src_cache_num > dst_cache_num) {
    // copy from first dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          dst_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr += src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr += dst_cache_num * metadata_.head_dim;
    }
  } else {
    v_cache_in_read_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * src_cache_num;
    v_cache_in_write_ptr +=
        metadata_.head_dim * (metadata_.num_heads - 1) * dst_cache_num;
    // copy from last dimension
    for (int i = 0; i < metadata_.num_heads; i++) {
      std::memmove(
          v_cache_in_write_ptr,
          v_cache_in_read_ptr,
          src_cache_num * metadata_.head_dim * sizeof(T));
      v_cache_in_read_ptr -= src_cache_num * metadata_.head_dim;
      v_cache_in_write_ptr -= dst_cache_num * metadata_.head_dim;
    }
  }
}

template <typename T>
void KVManager<T>::update_cache(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  update_cache_range(
      ar_len, n_past, n_update, selected, 0, metadata_.num_layers);
}

template <typename T>
void KVManager<T>::update_cache_range(
    int32_t ar_len,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected,
    int32_t layer_begin,
    int32_t layer_end_exclusive) {
  ET_CHECK_MSG(
      cur_ar_len_ == ar_len,
      "Current AR length (%d) is not matched with target AR length (%d). Please rearrange cache first.",
      cur_ar_len_,
      ar_len);
  ET_CHECK_MSG(
      layer_begin >= 0 && layer_begin <= layer_end_exclusive &&
          layer_end_exclusive <= metadata_.num_layers,
      "Invalid KV layer range [%d, %d) for %lld layers",
      layer_begin,
      layer_end_exclusive,
      static_cast<long long>(metadata_.num_layers));
  for (int layer = layer_begin; layer < layer_end_exclusive; ++layer) {
    update_key(k_cache_[layer], layer, n_past, n_update, selected);
    update_value(v_cache_[layer], layer, n_past, n_update, selected);
  }
}

template <typename T>
void KVManager<T>::update_key(
    KVCache<T>& k_cache,
    int32_t layer,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = k_cache.buffer;
  T* read_ptr = k_cache.output_buffer;
  const int32_t copy_size = n_update * sizeof(T);
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len
      : metadata_.context_len - cur_ar_len_;
  const int32_t out_size = cur_ar_len_;
  const int32_t past_size = n_past;
  const int32_t n_iter = metadata_.head_dim * metadata_.num_heads;

  write_ptr += past_size;
  const bool requant = !a8_axis_bridge_.empty() && cur_ar_len_ == 1;
  if (requant) {
    ET_CHECK_MSG(selected.empty(), "A8 KV bridge does not support selected decode");
    const auto& q = a8_axis_bridge_.at(static_cast<size_t>(layer));
    for (int i = 0; i < n_iter; ++i) {
      for (int j = 0; j < n_update; ++j) {
        const float real =
            (static_cast<int32_t>(read_ptr[j]) - q.decode_k_output_zero_point) *
            q.decode_k_output_scale;
        const int32_t code = static_cast<int32_t>(std::nearbyint(
            real / q.target_k_scale + q.target_k_zero_point));
        write_ptr[j] = static_cast<T>(std::clamp(code, 0, 255));
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
    return;
  }
  if (selected.empty()) {
    for (int i = 0; i < n_iter; ++i) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    std::vector<int32_t> true_indices(n_update);
    for (int i = 0, j = 0; i < selected.size() && j < n_update; ++i) {
      if (selected[i]) {
        true_indices[j++] = i;
      }
    }
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto ind : true_indices) {
        *wp++ = rp[ind];
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

template <typename T>
void KVManager<T>::update_value(
    KVCache<T>& v_cache,
    int32_t layer,
    int32_t n_past,
    int32_t n_update,
    const std::vector<bool>& selected) {
  T* write_ptr = v_cache.buffer;
  T* read_ptr = v_cache.output_buffer;
  const int32_t copy_size = n_update * metadata_.head_dim * sizeof(T);
  const int32_t past_size = n_past * metadata_.head_dim;
  const int32_t n_iter = metadata_.num_heads;
  const int32_t iter_size = (cur_ar_len_ == metadata_.context_len)
      ? metadata_.context_len * metadata_.head_dim
      : (metadata_.context_len - cur_ar_len_) * metadata_.head_dim;
  const int32_t out_size = cur_ar_len_ * metadata_.head_dim;

  write_ptr += past_size;
  const bool requant = !a8_axis_bridge_.empty() && cur_ar_len_ == 1;
  if (requant) {
    ET_CHECK_MSG(selected.empty(), "A8 KV bridge does not support selected decode");
    const auto& q = a8_axis_bridge_.at(static_cast<size_t>(layer));
    for (int i = 0; i < n_iter; ++i) {
      for (int token = 0; token < n_update; ++token) {
        const int32_t target_slot = (n_past + token) % a8_axis_period_;
        const float target_scale = q.target_v_scales[target_slot];
        const int32_t target_zp = q.target_v_zero_points[target_slot];
        for (int dim = 0; dim < metadata_.head_dim; ++dim) {
          const int32_t offset = token * metadata_.head_dim + dim;
          const float real =
              (static_cast<int32_t>(read_ptr[offset]) -
               q.decode_v_output_zero_point) *
              q.decode_v_output_scale;
          const int32_t code = static_cast<int32_t>(std::nearbyint(
              real / target_scale + target_zp));
          write_ptr[offset] = static_cast<T>(std::clamp(code, 0, 255));
        }
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
    return;
  }

  if (selected.empty()) {
    for (int i = 0; i < n_iter; i++) {
      std::memcpy(write_ptr, read_ptr, copy_size);
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  } else {
    int32_t update_times = n_update;
    for (int i = 0; i < n_iter; ++i) {
      auto wp = write_ptr, rp = read_ptr;
      for (auto sel : selected) {
        if (sel) {
          std::memcpy(wp, rp, metadata_.head_dim * sizeof(T));
          wp += metadata_.head_dim;
          update_times--;
          if (update_times == 0)
            break;
        }
        rp += metadata_.head_dim;
      }
      write_ptr += iter_size;
      read_ptr += out_size;
    }
  }
}

// Explicit instantiations
template class KVManager<uint16_t>;
template class KVManager<uint8_t>;

} // namespace example
