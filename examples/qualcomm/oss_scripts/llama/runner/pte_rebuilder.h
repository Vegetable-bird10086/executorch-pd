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

class ReadOnlyMappedFile {
 public:
  static std::shared_ptr<ReadOnlyMappedFile> open(const std::string& path);
  ~ReadOnlyMappedFile();

  ReadOnlyMappedFile(const ReadOnlyMappedFile&) = delete;
  ReadOnlyMappedFile& operator=(const ReadOnlyMappedFile&) = delete;

  const uint8_t* data() const;
  size_t size() const;
  bool empty() const;
  void discard_resident_pages() const;

 private:
  ReadOnlyMappedFile(int fd, const uint8_t* data, size_t size);

  int fd_{-1};
  const uint8_t* data_{nullptr};
  size_t size_{0};
};

class PteRebuildBuffer {
 public:
  explicit PteRebuildBuffer(size_t capacity);

  PteRebuildBuffer(const PteRebuildBuffer&) = delete;
  PteRebuildBuffer& operator=(const PteRebuildBuffer&) = delete;

  void resize_uninitialized(size_t size);
  uint8_t* data();
  const uint8_t* data() const;
  size_t size() const;
  size_t capacity() const;
  bool empty() const;

 private:
  std::unique_ptr<uint8_t[]> bytes_;
  size_t size_{0};
  size_t capacity_{0};
};

struct PteRebuildResult {
  std::shared_ptr<std::vector<uint8_t>> rebuilt_pte;
  double rebuild_time_ms{0.0};
  size_t rebuilt_records{0};
  size_t materialized_weight_bytes{0};
  bool specialized_fast_path_used{false};
  double allocation_ms{0.0};
  double static_copy_ms{0.0};
  double weight_materialization_ms{0.0};
  std::shared_ptr<PteRebuildBuffer> rebuilt_pte_buffer;
};

struct PteSplitMaterializationStats {
  size_t num_splits{1};
  size_t full_materialized_weight_bytes{0};
  size_t peak_split_materialized_weight_bytes{0};
  std::vector<size_t> split_materialized_weight_bytes;
  std::vector<size_t> split_record_counts;
};

// Immutable QAT metadata shared by all stripped PTE shard rebuilds that use
// one safetensors checkpoint.
struct PteQatRebuildContext {
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

// Immutable, per-PTE rebuild plan. Preparing it resolves the index and
// checkpoint blocks once; rebuild then only copies and writes weight bytes.
struct PteQatShardRecipe {
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

struct PteGgufRebuildContext {
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

struct PteGgufShardRecipe {
  struct Impl;
  std::shared_ptr<const Impl> impl_;
};

enum class PteGgufRecipeRelayoutKind {
  None,
  RawBlocks,
  Gs32Source,
};

struct PteGgufRecipeRelayoutStats {
  PteGgufRecipeRelayoutKind kind{PteGgufRecipeRelayoutKind::None};
  bool enabled{false};
  double relayout_ms{0.0};
  size_t relayout_bytes{0};
};

std::shared_ptr<PteQatRebuildContext> create_pte_qat_rebuild_context(
    const std::shared_ptr<std::vector<uint8_t>>& checkpoint_bytes);

std::shared_ptr<PteQatShardRecipe> prepare_pte_qat_shard_recipe(
    const std::shared_ptr<PteQatRebuildContext>& context,
    const std::shared_ptr<std::vector<uint8_t>>& index_bytes,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode);

PteRebuildResult rebuild_pte_from_stripped_checkpoint_recipe(
    const std::vector<uint8_t>& stripped_pte,
    const PteQatShardRecipe& recipe,
    std::shared_ptr<PteRebuildBuffer> output_buffer);

size_t pte_rebuild_output_size(const PteQatShardRecipe& recipe);

std::shared_ptr<PteGgufRebuildContext> create_pte_gguf_rebuild_context(
    const std::shared_ptr<ReadOnlyMappedFile>& gguf_bytes);

void discard_pte_gguf_rebuild_source_pages(
    const std::shared_ptr<PteGgufRebuildContext>& context);

std::shared_ptr<PteGgufShardRecipe> prepare_pte_gguf_shard_recipe(
    const std::shared_ptr<PteGgufRebuildContext>& context,
    const std::shared_ptr<std::vector<uint8_t>>& index_bytes,
    int source_group_size,
    PteGgufRecipeRelayoutKind relayout_kind =
        PteGgufRecipeRelayoutKind::None);

PteGgufRecipeRelayoutStats pte_gguf_recipe_relayout_stats(
    const PteGgufShardRecipe& recipe);

PteRebuildResult rebuild_pte_from_stripped_gguf_recipe(
    const std::vector<uint8_t>& stripped_pte,
    const PteGgufShardRecipe& recipe,
    std::shared_ptr<PteRebuildBuffer> output_buffer);

size_t pte_rebuild_output_size(const PteGgufShardRecipe& recipe);

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
