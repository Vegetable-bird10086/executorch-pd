/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace example {

struct PteRebuildResult {
  std::shared_ptr<std::vector<uint8_t>> rebuilt_pte;
  double rebuild_time_ms{0.0};
  size_t rebuilt_records{0};
  size_t materialized_weight_bytes{0};
  bool specialized_fast_path_used{false};
};

struct PteSplitMaterializationStats {
  size_t num_splits{1};
  size_t full_materialized_weight_bytes{0};
  size_t peak_split_materialized_weight_bytes{0};
  std::vector<size_t> split_materialized_weight_bytes;
  std::vector<size_t> split_record_counts;
};

PteRebuildResult rebuild_pte_from_stripped_checkpoint(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& checkpoint_bytes,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode);

PteRebuildResult rebuild_pte_from_stripped_tmac_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes);

PteRebuildResult rebuild_pte_from_stripped_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes);

PteRebuildResult rebuild_pte_split_from_stripped_checkpoint(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& checkpoint_bytes,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode,
    int split_id);

PteRebuildResult rebuild_pte_split_from_stripped_tmac_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes,
    int split_id);

PteRebuildResult rebuild_pte_split_from_stripped_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes,
    int split_id);

PteSplitMaterializationStats analyze_split_materialization(
    const std::vector<uint8_t>& index_bytes);

} // namespace example
