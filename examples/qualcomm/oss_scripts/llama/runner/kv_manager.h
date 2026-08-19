/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/imem_alloc.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace example {

// Structure to hold key-value cache buffers
template <typename T>
struct KVCache {
  T* buffer;
  T* output_buffer;
};

/**
 * @class KVManager
 * @brief Class for kv cache update, rearrangement, and buffer allocatation.
 */
template <typename T>
class KVManager {
 public:
  struct Metadata {
    int32_t context_len;
    int64_t head_dim;
    int32_t max_ar_len;
    int32_t max_cache_len;
    int64_t num_heads;
    int64_t num_layers;
    // Default-off, stage-major Prefill-only KV pool. Each slot stores only the
    // layers in one shard; slots are leased until asynchronous PD handoff has
    // copied those layers and can grow when both initial slots are busy.
    int32_t elastic_prefill_slot_layers{0};
    int32_t elastic_prefill_initial_slots{0};
  };
  KVManager(Metadata metadata);
  ~KVManager();

  /**
   * @brief Allocate buffer for KV cache and set the cur_ar_len_.
   * @param buffer_manager Pointer to IMemAlloc instance; by default, it uses a
   * shared buffer with RPC memory.
   * @param ar_len Length of input tokens.
   */
  void init_cache(IMemAlloc* buffer_manager, int32_t ar_len);

  /**
   * @brief Switch key and value cache from AR-cur to AR-dst.
   * @param ar_len_dst Target length of input tokens.
   */
  void rearrange_cache(int32_t ar_len_dst);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);
  void init_attention_mask(
      uint8_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);

  /**
   * @brief Initialize attention mask based on kv manager mode, and attention
   * map.
   * For example,
   * ar_len = 4, CL = 6, n_past = 0,
   * attention map: {-1, 0, 1, 2} and SMART_MASK.
   * Attention_mask will be:
   * [     0     0 65535     0     0     0 ]
   * [     0     0 65535 65535     0     0 ]
   * [     0     0 65535 65535 65535     0 ]
   * [     0     0 65535 65535 65535 65535 ]
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param attention_map Vector containing the attention map values. The shape
   * of attention map should be [ar_len].
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   */
  void init_attention_mask(
      uint16_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});
  void init_attention_mask(
      uint8_t* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);
  void update_attention_mask(
      uint8_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);

  /**
   * @brief Update attention mask based on kv manager mode, and n_update.
   * @param attention_mask Pointer to the attention mask array to be
   * initialized.
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param sliding_window Length of sliding window for sliding window attention
   * mask
   * @param position_offset (optional) attention mask position offset of
   * lookahead decoder
   */
  void update_attention_mask(
      uint16_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});
  void update_attention_mask(
      uint8_t* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset = {});

  /**
   * @brief Based on cur_ar_len_ to update cache
   * @param ar_len Length of input tokens.
   * @param n_past Number of past elements in the cache.
   * @param n_update Number of elements to be updated.
   * @param selected Indicate which position to be updated
   */
  void update_cache(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);
  void update_cache_range(
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected,
      int32_t layer_begin,
      int32_t layer_end_exclusive);

  const std::vector<KVCache<T>>& get_k_cache_() const {
    return k_cache_;
  }
  const std::vector<KVCache<T>>& get_v_cache_() const {
    return v_cache_;
  }

  inline const size_t total_cache_size_in_bytes() const {
    return total_cache_size_;
  }

  int64_t get_head_dim() const {
    return metadata_.head_dim;
  }

  bool uses_elastic_prefill_slots() const {
    return elastic_prefill_enabled_;
  }
  void acquire_prefill_kv_slot(int32_t layer_begin, int32_t layer_end_exclusive);
  void release_prefill_kv_slot(int32_t layer_begin, int32_t layer_end_exclusive);
  size_t elastic_prefill_slot_count() const;
  size_t elastic_prefill_peak_slot_count() const;
  size_t elastic_prefill_slot_bytes() const {
    return elastic_prefill_slot_bytes_;
  }

 private:
  template <typename MaskT>
  void init_attention_mask_impl(
      MaskT* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past);
  template <typename MaskT>
  void init_attention_mask_impl(
      MaskT* attention_mask,
      const std::vector<int32_t>& attention_map,
      int32_t ar_len,
      int32_t n_past,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset);
  template <typename MaskT>
  void update_attention_mask_impl(
      MaskT* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update);
  template <typename MaskT>
  void update_attention_mask_impl(
      MaskT* attention_mask,
      int32_t ar_len,
      int32_t n_past,
      int32_t n_update,
      int32_t sliding_window,
      const std::vector<int32_t>& position_offset);

  // Helper functions to rearrange and update key and value caches
  void rearrange_key(KVCache<T>& k_cache, int32_t ar_len_dst);
  void rearrange_value(KVCache<T>& v_cache, int32_t ar_len_dst);
  void update_key(
      KVCache<T>& k_cache,
      int32_t layer,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);
  void update_value(
      KVCache<T>& v_cache,
      int32_t layer,
      int32_t n_past,
      int32_t n_update,
      const std::vector<bool>& selected);

  // metadata
  Metadata metadata_;
  size_t total_cache_size_;
  int32_t cur_ar_len_;
  // Store start pointer of k and v cache for input and output
  // input: layer -> head * head_dim * max_cache_len
  // output: layer -> head * head_dim * max_ar_len
  std::vector<KVCache<T>> k_cache_;
  std::vector<KVCache<T>> v_cache_;
  struct ElasticPrefillSlot {
    void* custom_mem{nullptr};
    bool in_use{false};
    int32_t layer_begin{-1};
    int32_t layer_count{0};
  };
  bool elastic_prefill_enabled_{false};
  size_t elastic_prefill_cache_in_bytes_{0};
  size_t elastic_prefill_cache_out_bytes_{0};
  size_t elastic_prefill_per_layer_bytes_{0};
  size_t elastic_prefill_slot_bytes_{0};
  size_t elastic_prefill_peak_slots_{0};
  mutable std::mutex elastic_prefill_mutex_;
  std::vector<ElasticPrefillSlot> elastic_prefill_slots_;
  std::vector<int32_t> elastic_prefill_layer_slots_;
  size_t allocate_elastic_prefill_slot_locked();
  void bind_elastic_prefill_layer_locked(
      int32_t logical_layer,
      size_t slot_index,
      int32_t local_layer);
  struct A8AxisBridgeLayer {
    float target_k_scale{1.0f};
    int32_t target_k_zero_point{0};
    std::vector<float> target_v_scales;
    std::vector<int32_t> target_v_zero_points;
    float decode_k_output_scale{1.0f};
    int32_t decode_k_output_zero_point{0};
    float decode_v_output_scale{1.0f};
    int32_t decode_v_output_zero_point{0};
  };
  std::vector<A8AxisBridgeLayer> a8_axis_bridge_;
  int32_t a8_axis_period_{0};
};
} // namespace example
