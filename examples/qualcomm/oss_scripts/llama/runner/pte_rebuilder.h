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
  bool specialized_fast_path_used{false};
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

} // namespace example
