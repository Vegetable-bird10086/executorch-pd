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
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/exec_aten/util/scalar_type_util.h>
#include <executorch/runtime/core/portable_type/half.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <numeric>
#include <limits>
#include <sstream>
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

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(
    SteadyClock::time_point start,
    SteadyClock::time_point end = SteadyClock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

const char* gguf_relayout_kind_name(PteGgufRecipeRelayoutKind kind) {
  switch (kind) {
    case PteGgufRecipeRelayoutKind::None:
      return "none";
    case PteGgufRecipeRelayoutKind::RawBlocks:
      return "raw_blocks";
    case PteGgufRecipeRelayoutKind::Gs32Source:
      return "gs32_source";
  }
  return "unknown";
}

struct ProcessMemorySnapshot {
  uint64_t rss_bytes{0};
  uint64_t hwm_bytes{0};
};

uint64_t read_proc_status_bytes(const char* field) {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(field, 0) != 0) {
      continue;
    }
    std::istringstream value(line.substr(std::strlen(field)));
    uint64_t kib = 0;
    value >> kib;
    return kib * 1024;
  }
  return 0;
}

ProcessMemorySnapshot process_memory_snapshot() {
  return {
      read_proc_status_bytes("VmRSS:"),
      read_proc_status_bytes("VmHWM:"),
  };
}

void record_memory_peak(uint64_t* destination, uint64_t value) {
  *destination = std::max(*destination, value);
}

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

TensorPtr clone_tensor(const Tensor& source) {
  std::vector<executorch::aten::SizesType> sizes(
      source.sizes().begin(), source.sizes().end());
  TensorPtr copy = make_tensor_ptr_from_sizes(sizes, source.scalar_type());
  ET_CHECK_MSG(copy->nbytes() == source.nbytes(), "Tensor clone size mismatch");
  std::memcpy(
      copy->mutable_data_ptr<void>(),
      source.const_data_ptr<void>(),
      source.nbytes());
  return copy;
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

// Diagnostic-only raw dump. The tensor bytes are written exactly as supplied by
// the runtime; the adjacent JSON records their shape and scalar type so a host
// tool can compare the QNN boundary with llama.cpp without changing values.
bool shard_tensor_dump_enabled() {
  const char* dir = std::getenv("ET_SHARD_TENSOR_DUMP_DIR");
  return dir != nullptr && dir[0] != static_cast<char>(0);
}

bool shard_tensor_dump_kv_enabled() {
  return std::getenv("ET_SHARD_TENSOR_DUMP_KV") != nullptr;
}

bool shard_tensor_dump_all_inputs_enabled() {
  return std::getenv("ET_SHARD_TENSOR_DUMP_ALL_INPUTS") != nullptr;
}

bool shard_tensor_dump_matches_input_binding(size_t binding_index) {
  const char* requested =
      std::getenv("ET_SHARD_TENSOR_DUMP_INPUT_BINDING_INDEX");
  if (requested == nullptr || requested[0] == static_cast<char>(0)) {
    return true;
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(requested, &end, 10);
  return end != requested && end != nullptr && end[0] == static_cast<char>(0) &&
      value == binding_index;
}

bool shard_tensor_dump_matches_layer_offset(size_t layer_offset) {
  const char* requested = std::getenv("ET_SHARD_TENSOR_DUMP_LAYER_OFFSET");
  if (requested == nullptr || requested[0] == static_cast<char>(0)) {
    return true;
  }
  char* end = nullptr;
  const unsigned long value = std::strtoul(requested, &end, 10);
  return end != requested && end != nullptr && end[0] == static_cast<char>(0) &&
      value == layer_offset;
}

std::string prefill_shard_tensor_dump_stem(
    size_t shard_index,
    size_t execution_count,
    const char* direction,
    const char* kind,
    size_t binding_index) {
  const char* dir = std::getenv("ET_SHARD_TENSOR_DUMP_DIR");
  std::ostringstream stem;
  stem << dir << "/qnn_prefill_shard" << shard_index << "_run"
       << execution_count << "_" << direction << "_" << kind << "_"
       << binding_index;
  return stem.str();
}

void dump_prefill_shard_tensor_bytes(
    size_t shard_index,
    size_t execution_count,
    const char* direction,
    const char* kind,
    size_t binding_index,
    const Tensor& tensor) {
  const std::string binary_path = prefill_shard_tensor_dump_stem(
      shard_index, execution_count, direction, kind, binding_index) + ".bin";
  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  if (!binary.is_open()) {
    ET_LOG(Error, "Unable to open QNN tensor dump path=%s", binary_path.c_str());
    return;
  }
  binary.write(
      reinterpret_cast<const char*>(tensor.const_data_ptr()),
      static_cast<std::streamsize>(tensor.nbytes()));
  binary.close();
  if (!binary) {
    ET_LOG(Error, "Unable to write QNN tensor dump path=%s", binary_path.c_str());
    return;
  }
  ET_LOG(
      Info,
      "wrote raw QNN prefill tensor dump shard=%zu run=%zu direction=%s kind=%s "
      "binding=%zu bytes=%zu dtype=%s",
      shard_index,
      execution_count,
      direction,
      kind,
      binding_index,
      tensor.nbytes(),
      executorch::runtime::toString(tensor.scalar_type()));
}

void dump_prefill_shard_tensor(
    size_t shard_index,
    size_t execution_count,
    const char* direction,
    const char* kind,
    size_t binding_index,
    const Tensor& tensor) {
  const char* dir = std::getenv("ET_SHARD_TENSOR_DUMP_DIR");
  if (dir == nullptr || dir[0] == static_cast<char>(0)) {
    return;
  }

  const std::string stem = prefill_shard_tensor_dump_stem(
      shard_index, execution_count, direction, kind, binding_index);
  const std::string binary_path = stem + ".bin";
  const std::string metadata_path = stem + ".json";

  std::ofstream binary(binary_path, std::ios::binary | std::ios::trunc);
  if (!binary.is_open()) {
    ET_LOG(
        Error,
        "Unable to open QNN tensor dump path=%s; create ET_SHARD_TENSOR_DUMP_DIR first",
        binary_path.c_str());
    return;
  }
  binary.write(
      reinterpret_cast<const char*>(tensor.const_data_ptr()),
      static_cast<std::streamsize>(tensor.nbytes()));
  binary.close();
  if (!binary) {
    ET_LOG(Error, "Unable to write QNN tensor dump path=%s", binary_path.c_str());
    return;
  }

  std::ofstream metadata(metadata_path, std::ios::trunc);
  if (!metadata.is_open()) {
    ET_LOG(Error, "Unable to write QNN tensor dump metadata=%s", metadata_path.c_str());
    return;
  }
  metadata << "{\n"
           << "  \"shard_index\": " << shard_index << ",\n"
           << "  \"execution_count\": " << execution_count << ",\n"
           << "  \"direction\": \"" << direction << "\",\n"
           << "  \"kind\": \"" << kind << "\",\n"
           << "  \"binding_index\": " << binding_index << ",\n"
           << "  \"scalar_type\": \""
           << executorch::runtime::toString(tensor.scalar_type()) << "\",\n"
           << "  \"scalar_type_id\": " << static_cast<int>(tensor.scalar_type())
           << ",\n"
           << "  \"element_size\": "
           << executorch::runtime::elementSize(tensor.scalar_type()) << ",\n"
           << "  \"nbytes\": " << tensor.nbytes() << ",\n"
           << "  \"sizes\": [";
  for (size_t i = 0; i < tensor.sizes().size(); ++i) {
    if (i != 0) {
      metadata << ", ";
    }
    metadata << tensor.sizes()[i];
  }
  metadata << "],\n  \"dim_order\": [";
  for (size_t i = 0; i < tensor.dim_order().size(); ++i) {
    if (i != 0) {
      metadata << ", ";
    }
    metadata << static_cast<uint32_t>(tensor.dim_order()[i]);
  }
  metadata << "]\n}\n";
  ET_LOG(
      Info,
      "wrote QNN prefill tensor dump shard=%zu run=%zu direction=%s kind=%s binding=%zu bytes=%zu dtype=%s",
      shard_index,
      execution_count,
      direction,
      kind,
      binding_index,
      tensor.nbytes(),
      executorch::runtime::toString(tensor.scalar_type()));
}

void log_raw_fp16_carrier_stats(
    size_t shard_offset,
    const char* kind,
    const Tensor& carrier,
    size_t logical_numel) {
  ET_CHECK_MSG(
      carrier.nbytes() >= logical_numel * sizeof(uint16_t),
      "Raw FP16 carrier is too small: kind=%s carrier_bytes=%zu required=%zu",
      kind,
      carrier.nbytes(),
      logical_numel * sizeof(uint16_t));
  const uint16_t* values = carrier.const_data_ptr<uint16_t>();
  size_t nan_count = 0;
  size_t inf_count = 0;
  size_t zero_count = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < logical_numel; ++i) {
    executorch::aten::Half fp16;
    std::memcpy(&fp16, &values[i], sizeof(values[i]));
    const float value = static_cast<float>(fp16);
    if (std::isnan(value)) {
      ++nan_count;
      continue;
    }
    if (std::isinf(value)) {
      ++inf_count;
      continue;
    }
    if (value == 0.0f) {
      ++zero_count;
    }
    max_abs = std::max(max_abs, std::fabs(value));
  }
  ET_LOG(
      Info,
      "prefill raw fp16 carrier stats: shard_offset=%zu kind=%s numel=%zu nan=%zu inf=%zu zero=%zu max_abs=%g",
      shard_offset,
      kind,
      logical_numel,
      nan_count,
      inf_count,
      zero_count,
      max_abs);
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  const std::streamsize size = input.tellg();
  ET_CHECK_MSG(size >= 0, "Unable to determine file size: %s", path.c_str());
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.seekg(0, std::ios::beg);
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char*>(bytes.data()), size);
    ET_CHECK_MSG(input.good(), "Unable to read file: %s", path.c_str());
  }
  return bytes;
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

DecoderRunner::~DecoderRunner() {
  stop_prefill_shard_three_stage_pipeline();
}

void DecoderRunner::use_qwen3_prefill_static_plan(
    bool enabled,
    int32_t aux_size,
    int32_t hidden_size) {
  prefill_qwen3_static_plan_ = enabled;
  prefill_static_aux_size_ = aux_size;
  prefill_static_hidden_size_ = hidden_size;
}

void DecoderRunner::set_prefill_outputs_logits(bool enabled) {
  prefill_outputs_logits_ = enabled;
}

void DecoderRunner::set_prefill_separate_embed(bool enabled) {
  prefill_separate_embed_ = enabled;
}

void DecoderRunner::set_prefill_etdump_config(PrefillEtDumpConfig config) {
  ET_CHECK_MSG(
      prefill_shards_.empty(),
      "Configure prefill ETDump before prefill shard setup");
  prefill_etdump_config_ = std::move(config);
  if (prefill_etdump_config_.enabled()) {
    ET_LOG(
        Info,
        "prefill ETDump configured: shard=%d dir=%s debug_buffer_bytes=%zu; "
        "the selected PTE must have dump_intermediate_outputs enabled",
        prefill_etdump_config_.shard_index,
        prefill_etdump_config_.output_dir.c_str(),
        prefill_etdump_config_.debug_buffer_bytes);
  }
}

std::shared_ptr<PteRebuildBuffer> DecoderRunner::acquire_prefill_rebuild_buffer(
    size_t required_size) {
  std::shared_ptr<PteRebuildBuffer> buffer;
  {
    std::lock_guard<std::mutex> lock(prefill_rebuild_buffer_pool_mutex_);
    auto it = std::find_if(
        prefill_rebuild_buffer_pool_.begin(),
        prefill_rebuild_buffer_pool_.end(),
        [required_size](const std::shared_ptr<PteRebuildBuffer>& candidate) {
          return candidate->capacity() >= required_size;
        });
    if (it != prefill_rebuild_buffer_pool_.end()) {
      buffer = std::move(*it);
      prefill_rebuild_buffer_pool_.erase(it);
    }
  }

  const bool reused = buffer != nullptr;
  if (!reused) {
    buffer = std::make_shared<PteRebuildBuffer>(required_size);
  } else {
    buffer->resize_uninitialized(required_size);
  }
  ET_LOG(
      Info,
      "prefill rebuild buffer: required_bytes=%zu capacity_bytes=%zu reused=%d",
      required_size,
      buffer->capacity(),
      static_cast<int>(reused));
  return buffer;
}

void DecoderRunner::release_prefill_rebuild_buffer(
    std::shared_ptr<PteRebuildBuffer> buffer) {
  if (!buffer) {
    return;
  }
  buffer->resize_uninitialized(0);
  const size_t retained_buffer_count =
      prefill_shard_rebuild_.pipeline_qnn_load ? 3 : 1;
  std::shared_ptr<PteRebuildBuffer> discarded;
  {
    std::lock_guard<std::mutex> lock(prefill_rebuild_buffer_pool_mutex_);
    if (prefill_rebuild_buffer_pool_.size() < retained_buffer_count) {
      prefill_rebuild_buffer_pool_.push_back(std::move(buffer));
    } else {
      auto smallest = std::min_element(
          prefill_rebuild_buffer_pool_.begin(),
          prefill_rebuild_buffer_pool_.end(),
          [](const std::shared_ptr<PteRebuildBuffer>& lhs,
             const std::shared_ptr<PteRebuildBuffer>& rhs) {
            return lhs->capacity() < rhs->capacity();
          });
      if (smallest != prefill_rebuild_buffer_pool_.end() &&
          (*smallest)->capacity() < buffer->capacity()) {
        discarded = std::move(*smallest);
        *smallest = std::move(buffer);
      } else {
        discarded = std::move(buffer);
      }
    }
  }
}

void DecoderRunner::preload_prefill_shard(PrefillShardPlan& shard) {
  if (!shard.rebuild_on_execute) {
    return;
  }

  if (shard.stripped_pte_bytes == nullptr || shard.index_bytes == nullptr) {
    const auto preload_start = SteadyClock::now();
    shard.stripped_pte_bytes =
        std::make_shared<std::vector<uint8_t>>(read_binary_file(shard.pte_path));
    shard.index_bytes =
        std::make_shared<std::vector<uint8_t>>(read_binary_file(shard.index_path));
    shard.runtime_stats.preload_ms += elapsed_ms(preload_start);
    ET_LOG(
        Info,
        "preloaded prefill shard: stripped=%s index=%s stripped_bytes=%zu "
        "index_bytes=%zu preload_ms=%f",
        shard.pte_path.c_str(),
        shard.index_path.c_str(),
        shard.stripped_pte_bytes->size(),
        shard.index_bytes->size(),
        shard.runtime_stats.preload_ms);
  }

  if (prefill_shard_rebuild_.source_kind ==
          PrefillShardRebuildConfig::SourceKind::Gguf &&
      shard.gguf_rebuild_recipe == nullptr) {
    if (prefill_gguf_rebuild_context_ == nullptr) {
      const auto context_start = SteadyClock::now();
      prefill_gguf_rebuild_context_ = create_pte_gguf_rebuild_context(
          prefill_shard_rebuild_.source_bytes);
      shard.runtime_stats.gguf_checkpoint_context_ms += elapsed_ms(context_start);
      ET_LOG(
          Info,
          "prepared shared GGUF rebuild context: gguf_bytes=%zu context_ms=%f",
          prefill_shard_rebuild_.source_bytes->size(),
          shard.runtime_stats.gguf_checkpoint_context_ms);
    }

    const auto recipe_start = SteadyClock::now();
    shard.gguf_rebuild_recipe = prepare_pte_gguf_shard_recipe(
        prefill_gguf_rebuild_context_,
        shard.index_bytes,
        prefill_shard_rebuild_.group_size,
        prefill_shard_rebuild_.gguf_relayout_kind);
    shard.runtime_stats.gguf_recipe_ms += elapsed_ms(recipe_start);
    const auto relayout_stats =
        pte_gguf_recipe_relayout_stats(*shard.gguf_rebuild_recipe);
    shard.runtime_stats.gguf_relayout_ms += relayout_stats.relayout_ms;
    shard.runtime_stats.gguf_relayout_bytes += relayout_stats.relayout_bytes;
    ET_LOG(
        Info,
        "prepared GGUF prefill shard recipe: stripped=%s index=%s recipe_ms=%f",
        shard.pte_path.c_str(),
        shard.index_path.c_str(),
        shard.runtime_stats.gguf_recipe_ms);
    if (relayout_stats.enabled) {
      ET_LOG(
          Info,
          "prepared GGUF source relayout: kind=%s stripped=%s relayout_ms=%f "
          "relayout_bytes=%zu",
          gguf_relayout_kind_name(relayout_stats.kind),
          shard.pte_path.c_str(),
          relayout_stats.relayout_ms,
          relayout_stats.relayout_bytes);
    }
    return;
  }

  if (prefill_shard_rebuild_.source_kind !=
          PrefillShardRebuildConfig::SourceKind::QatCheckpoint ||
      shard.qat_rebuild_recipe != nullptr) {
    return;
  }

  if (prefill_qat_rebuild_context_ == nullptr) {
    const auto context_start = SteadyClock::now();
    prefill_qat_rebuild_context_ = create_pte_qat_rebuild_context(
        prefill_shard_rebuild_.source_bytes);
    shard.runtime_stats.qat_checkpoint_context_ms += elapsed_ms(context_start);
    ET_LOG(
        Info,
        "prepared shared QAT rebuild checkpoint context: checkpoint_bytes=%zu "
        "context_ms=%f",
        prefill_shard_rebuild_.source_bytes->size(),
        shard.runtime_stats.qat_checkpoint_context_ms);
  }

  const auto recipe_start = SteadyClock::now();
  shard.qat_rebuild_recipe = prepare_pte_qat_shard_recipe(
      prefill_qat_rebuild_context_,
      shard.index_bytes,
      prefill_shard_rebuild_.bits_hint,
      prefill_shard_rebuild_.group_size,
      prefill_shard_rebuild_.qweight_mode);
  shard.runtime_stats.qat_recipe_ms += elapsed_ms(recipe_start);
  ET_LOG(
      Info,
      "prepared QAT prefill shard recipe: stripped=%s index=%s recipe_ms=%f",
      shard.pte_path.c_str(),
      shard.index_path.c_str(),
      shard.runtime_stats.qat_recipe_ms);
}

PteRebuildResult DecoderRunner::rebuild_prefill_shard(
    PrefillShardPlan& shard) {
  ET_CHECK_MSG(
      prefill_shard_rebuild_.source_kind !=
          PrefillShardRebuildConfig::SourceKind::None,
      "Prefill shard index was provided but no rebuild source is configured");
  ET_CHECK_MSG(
      prefill_shard_rebuild_.source_bytes &&
          !prefill_shard_rebuild_.source_bytes->empty(),
      "Prefill shard rebuild source bytes are empty");
  ET_CHECK_MSG(
      shard.stripped_pte_bytes && shard.index_bytes,
      "Prefill shard inputs were not preloaded: %s",
      shard.pte_path.c_str());

  switch (prefill_shard_rebuild_.source_kind) {
    case PrefillShardRebuildConfig::SourceKind::QatCheckpoint:
      ET_CHECK_MSG(
          shard.qat_rebuild_recipe != nullptr,
          "QAT rebuild recipe was not prepared: %s",
          shard.pte_path.c_str());
      return rebuild_pte_from_stripped_checkpoint_recipe(
          *shard.stripped_pte_bytes,
          *shard.qat_rebuild_recipe,
          acquire_prefill_rebuild_buffer(
              pte_rebuild_output_size(*shard.qat_rebuild_recipe)));
    case PrefillShardRebuildConfig::SourceKind::TmacGguf:
      return rebuild_pte_from_stripped_tmac_gguf(
          *shard.stripped_pte_bytes,
          *shard.index_bytes,
          *prefill_shard_rebuild_.source_bytes);
    case PrefillShardRebuildConfig::SourceKind::Gguf:
      ET_CHECK_MSG(
          shard.gguf_rebuild_recipe != nullptr,
          "GGUF rebuild recipe was not prepared: %s",
          shard.pte_path.c_str());
      return rebuild_pte_from_stripped_gguf_recipe(
          *shard.stripped_pte_bytes,
          *shard.gguf_rebuild_recipe,
          acquire_prefill_rebuild_buffer(
              pte_rebuild_output_size(*shard.gguf_rebuild_recipe)));
    case PrefillShardRebuildConfig::SourceKind::None:
      ET_CHECK_MSG(false, "Invalid prefill shard rebuild source");
  }
  ET_CHECK_MSG(false, "Unhandled prefill shard rebuild source");
  return {};
}

void DecoderRunner::attach_rebuilt_prefill_shard(
    PrefillShardPlan& shard,
    PteRebuildResult rebuild_result) {
  ET_CHECK_MSG(
      (rebuild_result.rebuilt_pte_buffer != nullptr &&
       !rebuild_result.rebuilt_pte_buffer->empty()) ||
          (rebuild_result.rebuilt_pte != nullptr &&
           !rebuild_result.rebuilt_pte->empty()),
      "Rebuilt prefill shard is empty: %s",
      shard.pte_path.c_str());
  shard.rebuilt_pte_buffer = std::move(rebuild_result.rebuilt_pte_buffer);
  shard.rebuilt_pte_bytes = std::move(rebuild_result.rebuilt_pte);
  const uint8_t* rebuilt_pte_data = shard.rebuilt_pte_buffer
      ? shard.rebuilt_pte_buffer->data()
      : shard.rebuilt_pte_bytes->data();
  const size_t rebuilt_pte_size = shard.rebuilt_pte_buffer
      ? shard.rebuilt_pte_buffer->size()
      : shard.rebuilt_pte_bytes->size();
  shard.runtime_stats.rebuild_ms += rebuild_result.rebuild_time_ms;
  shard.runtime_stats.rebuild_allocation_ms += rebuild_result.allocation_ms;
  shard.runtime_stats.rebuild_static_copy_ms += rebuild_result.static_copy_ms;
  shard.runtime_stats.rebuild_weight_materialization_ms +=
      rebuild_result.weight_materialization_ms;
  auto data_loader = std::make_unique<executorch::extension::BufferDataLoader>(
      rebuilt_pte_data, rebuilt_pte_size);
  shard.module = std::make_unique<Module>(std::move(data_loader));
  ET_LOG(
      Info,
      "rebuilt prefill shard: stripped=%s index=%s "
      "materialized_weight_bytes=%zu rebuilt_records=%zu rebuild_ms=%f "
      "allocation_ms=%f static_copy_ms=%f weight_materialization_ms=%f",
      shard.pte_path.c_str(),
      shard.index_path.c_str(),
      rebuild_result.materialized_weight_bytes,
      rebuild_result.rebuilt_records,
      rebuild_result.rebuild_time_ms,
      rebuild_result.allocation_ms,
      rebuild_result.static_copy_ms,
      rebuild_result.weight_materialization_ms);
}

void DecoderRunner::materialize_prefill_shard(PrefillShardPlan& shard) {
  if (shard.module != nullptr) {
    return;
  }

  preload_prefill_shard(shard);
  const auto materialize_start = SteadyClock::now();
  if (shard.rebuild_on_execute) {
    attach_rebuilt_prefill_shard(shard, rebuild_prefill_shard(shard));
  } else {
    shard.module = std::make_unique<Module>(
        shard.pte_path.c_str(), Module::LoadMode::MmapUseMlockIgnoreErrors);
  }
  shard.runtime_stats.materialize_ms += elapsed_ms(materialize_start);
}

void DecoderRunner::release_prefill_shard(PrefillShardPlan& shard) {
  if (!shard.rebuild_on_execute) {
    return;
  }
  shard.module.reset();
  shard.rebuilt_pte_bytes.reset();
  auto rebuild_buffer = std::move(shard.rebuilt_pte_buffer);
  release_prefill_rebuild_buffer(std::move(rebuild_buffer));
}

Error DecoderRunner::load_prefill_shard_method(PrefillShardPlan& shard) {
  ET_CHECK_MSG(shard.module != nullptr, "Prefill shard is not materialized");
  const bool arm_etdump = should_arm_prefill_etdump(shard);
  if (arm_etdump) {
    if (shard.module->is_method_loaded(shard.method_name)) {
      ET_CHECK_MSG(
          shard.module->unload_method(shard.method_name),
          "Unable to unload already-loaded ETDump shard %zu",
          shard.shard_index);
    }
    if (shard.etdump_gen == nullptr) {
      shard.etdump_gen = std::make_unique<executorch::etdump::ETDumpGen>();
    }
  }
  const auto qnn_load_start = SteadyClock::now();
  ET_CHECK_OK_OR_RETURN_ERROR(shard.module->load_method(
      shard.method_name,
      nullptr,
      arm_etdump ? shard.etdump_gen.get() : nullptr));
  shard.runtime_stats.qnn_load_method_ms += elapsed_ms(qnn_load_start);
  if (arm_etdump && !shard.etdump_armed) {
    // Match qnn_executor_runner: the delegate receives the ETDump tracer at
    // method load, but tensor recording is enabled only after QNN setup.
    shard.etdump_debug_buffer.assign(
        prefill_etdump_config_.debug_buffer_bytes, 0);
    const auto set_buffer_result = shard.etdump_gen->set_debug_buffer(
        executorch::runtime::Span<uint8_t>(
            shard.etdump_debug_buffer.data(),
            shard.etdump_debug_buffer.size()));
    ET_CHECK_MSG(
        set_buffer_result.ok(),
        "Unable to configure ETDump debug buffer for shard %zu: error=%d",
        shard.shard_index,
        static_cast<int>(set_buffer_result.error()));
    shard.etdump_gen->set_event_tracer_debug_level(
        executorch::runtime::EventTracerDebugLogLevel::kIntermediateOutputs);
    shard.etdump_armed = true;
    ET_LOG(
        Info,
        "armed prefill ETDump: shard=%zu bytes=%zu",
        shard.shard_index,
        shard.etdump_debug_buffer.size());
  }

  const auto method_meta = shard.module->method_meta(shard.method_name);
  ET_CHECK_OK_OR_RETURN_ERROR(method_meta.error());
  ET_CHECK_MSG(
      method_meta->num_inputs() == shard.input_bindings.size(),
      "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu inputs but the "
      "configured static plan provides %zu. Check prefill_outputs_logits in the "
      "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
      shard.layer_offset,
      shard.method_name.c_str(),
      method_meta->num_inputs(),
      shard.input_bindings.size());
  ET_CHECK_MSG(
      method_meta->num_outputs() == shard.output_bindings.size(),
      "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu outputs but the "
      "configured static plan provides %zu. Check prefill_outputs_logits in the "
      "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
      shard.layer_offset,
      shard.method_name.c_str(),
      method_meta->num_outputs(),
      shard.output_bindings.size());

  const auto memory_after_load = process_memory_snapshot();
  record_memory_peak(
      &shard.runtime_stats.rss_after_load_bytes, memory_after_load.rss_bytes);
  record_memory_peak(
      &shard.runtime_stats.hwm_after_load_bytes, memory_after_load.hwm_bytes);

  const auto output_binding_start = SteadyClock::now();
  for (size_t i = 0; i < shard.output_tensors.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        shard.module->set_output(shard.method_name, shard.output_tensors[i], i));
  }
  shard.runtime_stats.output_binding_ms += elapsed_ms(output_binding_start);
  return Error::Ok;
}

bool DecoderRunner::should_arm_prefill_etdump(
    const PrefillShardPlan& shard) const {
  return prefill_etdump_config_.enabled() && !shard.etdump_written &&
      shard.shard_index ==
      static_cast<size_t>(prefill_etdump_config_.shard_index);
}

Error DecoderRunner::write_prefill_etdump(PrefillShardPlan& shard) {
  if (!shard.etdump_armed || shard.etdump_written) {
    return Error::Ok;
  }
  ET_CHECK_MSG(
      shard.etdump_gen != nullptr,
      "ETDump shard %zu has no event tracer",
      shard.shard_index);

  std::error_code create_dir_error;
  std::filesystem::create_directories(
      prefill_etdump_config_.output_dir, create_dir_error);
  ET_CHECK_MSG(
      !create_dir_error,
      "Unable to create prefill ETDump directory %s: %s",
      prefill_etdump_config_.output_dir.c_str(),
      create_dir_error.message().c_str());

  const std::string stem =
      prefill_etdump_config_.output_dir + "/prefill_shard" +
      std::to_string(shard.shard_index) + "_run" +
      std::to_string(shard.runtime_stats.execution_count);
  const std::string etdump_path = stem + ".etdp";
  const std::string debug_output_path = stem + ".debug_output.bin";
  const auto etdump = shard.etdump_gen->get_etdump_data();
  ET_CHECK_MSG(
      etdump.buf != nullptr && etdump.size > 0,
      "ETDump contains no events for shard %zu. Re-export its PTE with "
      "--dump_intermediate_outputs before running this diagnostic.",
      shard.shard_index);

  {
    std::ofstream output(etdump_path, std::ios::binary | std::ios::trunc);
    ET_CHECK_MSG(output.is_open(), "Unable to write ETDump: %s", etdump_path.c_str());
    output.write(reinterpret_cast<const char*>(etdump.buf), etdump.size);
    output.close();
    ET_CHECK_MSG(output.good(), "Unable to finish ETDump: %s", etdump_path.c_str());
  }
  std::free(etdump.buf);
  {
    std::ofstream output(
        debug_output_path, std::ios::binary | std::ios::trunc);
    ET_CHECK_MSG(
        output.is_open(), "Unable to write ETDump tensor buffer: %s", debug_output_path.c_str());
    output.write(
        reinterpret_cast<const char*>(shard.etdump_debug_buffer.data()),
        static_cast<std::streamsize>(shard.etdump_debug_buffer.size()));
    output.close();
    ET_CHECK_MSG(
        output.good(), "Unable to finish ETDump tensor buffer: %s", debug_output_path.c_str());
  }

  shard.etdump_written = true;
  shard.etdump_armed = false;
  ET_LOG(
      Info,
      "wrote prefill ETDump: shard=%zu run=%zu etdump=%s bytes=%zu "
      "debug_output=%s bytes=%zu",
      shard.shard_index,
      shard.runtime_stats.execution_count,
      etdump_path.c_str(),
      etdump.size,
      debug_output_path.c_str(),
      shard.etdump_debug_buffer.size());

  // ETDumpGen is finalized by get_etdump_data(). Unload and load normally so
  // later AR blocks cannot append to a finalized tracer.
  ET_CHECK_MSG(
      shard.module->unload_method(shard.method_name),
      "Unable to unload ETDump shard %zu after capture",
      shard.shard_index);
  shard.etdump_gen.reset();
  shard.etdump_debug_buffer.clear();
  return load_prefill_shard_method(shard);
}

bool DecoderRunner::uses_prefill_shard_three_stage_pipeline() const {
  return prefill_shard_rebuild_.pipeline_qnn_load &&
      uses_prefill_shard_stage_major() && !prefill_shards_.empty();
}

void DecoderRunner::start_prefill_shard_three_stage_pipeline() {
  if (prefill_shard_three_stage_pipeline_ != nullptr) {
    return;
  }
  ET_CHECK_MSG(
      uses_prefill_shard_three_stage_pipeline(),
      "Three-stage pipeline requires stage-major sharded prefill");
  for (const auto& shard : prefill_shards_) {
    ET_CHECK_MSG(
        shard.rebuild_on_execute,
        "Three-stage pipeline requires every prefill shard to rebuild on execute");
  }

  const size_t shard_count = prefill_shards_.size();
  auto pipeline = std::make_unique<PrefillShardThreeStagePipeline>();
  pipeline->rebuild_results.resize(shard_count);
  pipeline->rebuilt.assign(shard_count, false);
  pipeline->loaded.assign(shard_count, false);
  pipeline->rebuild_permit_index = std::min<size_t>(1, shard_count - 1);
  pipeline->load_permit_index = 0;
  auto* state = pipeline.get();
  prefill_shard_three_stage_pipeline_ = std::move(pipeline);

  state->rebuild_worker = std::thread([this, state, shard_count]() {
    for (size_t shard_index = 0; shard_index < shard_count; ++shard_index) {
      {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait(lock, [state, shard_index]() {
          return state->stop || shard_index <= state->rebuild_permit_index;
        });
        if (state->stop) {
          return;
        }
      }

      PteRebuildResult rebuild_result =
          rebuild_prefill_shard(prefill_shards_[shard_index]);
      {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->stop) {
          return;
        }
        state->rebuild_results[shard_index].emplace(std::move(rebuild_result));
        state->rebuilt[shard_index] = true;
      }
      state->cv.notify_all();
    }
  });

  state->load_worker = std::thread([this, state, shard_count]() {
    for (size_t shard_index = 0; shard_index < shard_count; ++shard_index) {
      PteRebuildResult rebuild_result;
      {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait(lock, [state, shard_index]() {
          return state->stop ||
              (state->rebuilt[shard_index] &&
               shard_index <= state->load_permit_index);
        });
        if (state->stop) {
          return;
        }
        rebuild_result = std::move(*state->rebuild_results[shard_index]);
        state->rebuild_results[shard_index].reset();
      }

      auto& shard = prefill_shards_[shard_index];
      const auto materialize_start = SteadyClock::now();
      attach_rebuilt_prefill_shard(shard, std::move(rebuild_result));
      shard.runtime_stats.materialize_ms += elapsed_ms(materialize_start);
      const auto memory_after_materialize = process_memory_snapshot();
      record_memory_peak(
          &shard.runtime_stats.rss_after_materialize_bytes,
          memory_after_materialize.rss_bytes);
      record_memory_peak(
          &shard.runtime_stats.hwm_after_materialize_bytes,
          memory_after_materialize.hwm_bytes);
      ET_CHECK_MSG(
          load_prefill_shard_method(shard) == Error::Ok,
          "Failed to load prefill shard method: index=%zu",
          shard_index);

      {
        std::lock_guard<std::mutex> lock(state->mutex);
        state->loaded[shard_index] = true;
      }
      state->cv.notify_all();
    }
  });
  ET_LOG(
      Info,
      "three-stage prefill pipeline started: rebuild_worker=1 load_worker=1 shards=%zu",
      shard_count);
}

void DecoderRunner::stop_prefill_shard_three_stage_pipeline() {
  auto pipeline = std::move(prefill_shard_three_stage_pipeline_);
  if (pipeline == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(pipeline->mutex);
    pipeline->stop = true;
  }
  pipeline->cv.notify_all();
  if (pipeline->rebuild_worker.joinable()) {
    pipeline->rebuild_worker.join();
  }
  if (pipeline->load_worker.joinable()) {
    pipeline->load_worker.join();
  }
}

void DecoderRunner::permit_prefill_shard_three_stage_pipeline(
    size_t rebuild_index,
    size_t load_index) {
  auto* pipeline = prefill_shard_three_stage_pipeline_.get();
  ET_CHECK_MSG(pipeline != nullptr, "Three-stage pipeline is not active");
  const size_t last_index = prefill_shards_.size() - 1;
  {
    std::lock_guard<std::mutex> lock(pipeline->mutex);
    pipeline->rebuild_permit_index = std::max(
        pipeline->rebuild_permit_index, std::min(rebuild_index, last_index));
    pipeline->load_permit_index = std::max(
        pipeline->load_permit_index, std::min(load_index, last_index));
  }
  pipeline->cv.notify_all();
}

double DecoderRunner::wait_for_prefill_shard_three_stage_load(
    size_t shard_index) {
  auto* pipeline = prefill_shard_three_stage_pipeline_.get();
  ET_CHECK_MSG(pipeline != nullptr, "Three-stage pipeline is not active");
  const auto wait_start = SteadyClock::now();
  {
    std::unique_lock<std::mutex> lock(pipeline->mutex);
    pipeline->cv.wait(lock, [pipeline, shard_index]() {
      return pipeline->stop || pipeline->loaded[shard_index];
    });
    ET_CHECK_MSG(!pipeline->stop, "Three-stage pipeline stopped before shard load");
  }
  return elapsed_ms(wait_start);
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
    shard.shard_index = prefill_shards_.size();
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
          (i == 0 && input_meta->sizes().size() == 3) ||
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
            {PrefillShardPlan::OutputKind::FinalLogits, 0, static_cast<size_t>(-1)});
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
    } else if (should_arm_prefill_etdump(shard)) {
      ET_CHECK_MSG(
          shard.module->unload_method(shard.method_name),
          "Unable to unload ETDump shard %zu during configuration",
          shard.shard_index);
      ET_CHECK_MSG(
          load_prefill_shard_method(shard) == Error::Ok,
          "Unable to load ETDump shard %zu during configuration",
          shard.shard_index);
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
    shard.shard_index = shard_index;
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
      if (prefill_separate_embed_) {
        // Separate-embedding exports use input_pos, atten_mask, then hidden_states.
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::Position, 0});
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::AttentionMask, 0});
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::Tokens, 0});
      } else {
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::Tokens, 0});
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::Position, 0});
        shard.input_bindings.push_back({PrefillShardPlan::InputKind::AttentionMask, 0});
      }
    } else {
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousAux, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousAux, 1});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::PreviousHidden, 0});
      shard.input_bindings.push_back({PrefillShardPlan::InputKind::AttentionMask, 0});
    }
    // Without logits, the final transformer layer only emits K/V. Its
    // attention/MLP path is pruned, so its old KV cache is not a graph input.
    const bool final_no_logits_shard =
        !prefill_outputs_logits_ && shard_index + 1 == prefill_shard_paths_.size();
    const size_t cache_input_layers =
        final_no_logits_shard ? shard.layer_count - 1 : shard.layer_count;
    for (size_t local_layer = 0; local_layer < cache_input_layers; ++local_layer) {
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
          {PrefillShardPlan::OutputKind::FinalVCache, local_layer, static_cast<size_t>(-1)});
      shard.output_bindings.push_back(
          {PrefillShardPlan::OutputKind::FinalKCache, local_layer, static_cast<size_t>(-1)});
    }
    if (shard_index + 1 == prefill_shard_paths_.size()) {
      if (prefill_outputs_logits_) {
        shard.output_bindings.push_back(
            {PrefillShardPlan::OutputKind::FinalLogits, 0, static_cast<size_t>(-1)});
      }
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
    shard.runtime_stats.layer_offset = shard.layer_offset;
    shard.runtime_stats.layer_count = shard.layer_count;
    prefill_shards_.push_back(std::move(shard));
    layer_offset += layers_per_shard;
  }

  ET_CHECK_MSG(
      layer_offset == static_cast<size_t>(prefill_num_layers_),
      "Qwen3 static prefill shard layers mismatch: expected=%lld got=%zu",
      static_cast<long long>(prefill_num_layers_),
      layer_offset);
  size_t preloaded_bytes = 0;
  const auto qnn_compile_spec_bytes =
      prefill_shard_rebuild_.qnn_compile_spec_bytes;
  if (prefill_shard_rebuild_.prewarm_qnn_backend &&
      (qnn_compile_spec_bytes == nullptr || qnn_compile_spec_bytes->empty())) {
    ET_LOG(
        Error,
        "QNN backend prewarm skipped: shard manifest has no "
        "qnn_compile_spec_hex from the complete PTE");
  }
  if (prefill_shard_rebuild_.prewarm_qnn_backend &&
      qnn_compile_spec_bytes != nullptr && !qnn_compile_spec_bytes->empty()) {
    ET_LOG(
        Info,
        "QNN backend prewarm begin on the prefill main thread: "
        "compile_spec_bytes=%zu",
        qnn_compile_spec_bytes->size());
    const auto prewarm_start = SteadyClock::now();
    const Error prewarm_error = executorch::backends::qnn::PrewarmQnnBackend(
        qnn_compile_spec_bytes->data(), qnn_compile_spec_bytes->size());
    prefill_qnn_backend_prewarm_ms_ = elapsed_ms(prewarm_start);
    if (prewarm_error == Error::Ok) {
      prefill_qnn_backend_prewarmed_ = true;
      ET_LOG(
          Info,
          "prewarmed QNN backend/device from manifest compile spec: "
          "compile_spec_bytes=%zu prewarm_ms=%f",
          qnn_compile_spec_bytes->size(),
          prefill_qnn_backend_prewarm_ms_);
    } else {
      ET_LOG(
          Error,
          "QNN backend prewarm failed with error=%d; falling back to normal "
          "shard 0 initialization",
          static_cast<int>(prewarm_error));
    }
  }
  for (size_t shard_index = 0; shard_index < prefill_shards_.size();
       ++shard_index) {
    auto& shard = prefill_shards_[shard_index];
    preload_prefill_shard(shard);
    if (shard.stripped_pte_bytes) {
      preloaded_bytes += shard.stripped_pte_bytes->size();
    }
    if (shard.index_bytes) {
      preloaded_bytes += shard.index_bytes->size();
    }
  }
  ET_LOG(
      Info,
      "preloaded prefill shard inputs: shards=%zu total_bytes=%zu",
      prefill_shards_.size(),
      preloaded_bytes);
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
          ET_CHECK_OK_OR_RETURN_ERROR(load_prefill_shard_method(shard));
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
  const size_t k_out_base = prefill_outputs_logits_ ? 1 : 0;
  const size_t v_out_base = k_out_base + static_cast<size_t>(prefill_num_layers_);

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


bool DecoderRunner::uses_prefill_shard_stage_major() const {
  return prefill_shard_rebuild_.stage_major_execution &&
      prefill_qwen3_static_plan_ && !prefill_shards_.empty();
}

size_t DecoderRunner::prefill_shard_count() const {
  return prefill_shards_.size();
}

size_t DecoderRunner::prefill_shard_layer_offset(size_t shard_index) const {
  ET_CHECK_MSG(shard_index < prefill_shards_.size(), "Invalid prefill shard index %zu", shard_index);
  return prefill_shards_[shard_index].layer_offset;
}

size_t DecoderRunner::prefill_shard_layer_count(size_t shard_index) const {
  ET_CHECK_MSG(shard_index < prefill_shards_.size(), "Invalid prefill shard index %zu", shard_index);
  return prefill_shards_[shard_index].layer_count;
}

Error DecoderRunner::begin_prefill_shard_stage(size_t shard_index) {
  ET_CHECK_MSG(
      uses_prefill_shard_stage_major(),
      "Stage-major prefill is not enabled");
  ET_CHECK_MSG(shard_index < prefill_shards_.size(), "Invalid prefill shard index %zu", shard_index);
  if (prefill_shard_stage_starts_.size() != prefill_shards_.size()) {
    prefill_shard_stage_starts_.resize(prefill_shards_.size());
  }

  auto& shard = prefill_shards_[shard_index];
  if (!uses_prefill_shard_three_stage_pipeline()) {
    ET_CHECK_MSG(shard.module == nullptr, "Prefill shard %zu is already active", shard_index);
  }
  prefill_shard_stage_starts_[shard_index] = SteadyClock::now();
  const auto memory_before = process_memory_snapshot();
  if (shard.runtime_stats.execution_count == 0) {
    shard.runtime_stats.rss_before_bytes = memory_before.rss_bytes;
    shard.runtime_stats.hwm_before_bytes = memory_before.hwm_bytes;
  }

  if (uses_prefill_shard_three_stage_pipeline()) {
    start_prefill_shard_three_stage_pipeline();
    shard.runtime_stats.pipeline_wait_ms +=
        wait_for_prefill_shard_three_stage_load(shard_index);
    ET_CHECK_MSG(
        shard.module != nullptr,
        "Three-stage pipeline did not load prefill shard %zu",
        shard_index);
    permit_prefill_shard_three_stage_pipeline(
        shard_index + 2, shard_index + 1);
    ET_LOG(
        Info,
        "stage-major three-stage pipeline: execute index=%zu while loading index=%zu and rebuilding index=%zu",
        shard_index,
        std::min(shard_index + 1, prefill_shards_.size() - 1),
        std::min(shard_index + 2, prefill_shards_.size() - 1));
    ET_LOG(
        Info,
        "prefill shard stage begin: index=%zu layers=[%zu,%zu)",
        shard_index,
        shard.layer_offset,
        shard.layer_offset + shard.layer_count);
    return Error::Ok;
  }

  if (prefill_shard_stage_pending_rebuild_.valid()) {
    ET_CHECK_MSG(
        prefill_shard_stage_pending_rebuild_index_ == shard_index,
        "Unexpected stage-major pending rebuild: expected=%zu got=%zu",
        shard_index,
        prefill_shard_stage_pending_rebuild_index_);
    const auto pipeline_wait_start = SteadyClock::now();
    PteRebuildResult rebuild_result =
        prefill_shard_stage_pending_rebuild_.get();
    shard.runtime_stats.pipeline_wait_ms += elapsed_ms(pipeline_wait_start);
    const auto materialize_start = SteadyClock::now();
    attach_rebuilt_prefill_shard(shard, std::move(rebuild_result));
    shard.runtime_stats.materialize_ms += elapsed_ms(materialize_start);
    prefill_shard_stage_pending_rebuild_index_ =
        std::numeric_limits<size_t>::max();
  } else {
    materialize_prefill_shard(shard);
  }
  const auto memory_after_materialize = process_memory_snapshot();
  record_memory_peak(
      &shard.runtime_stats.rss_after_materialize_bytes,
      memory_after_materialize.rss_bytes);
  record_memory_peak(
      &shard.runtime_stats.hwm_after_materialize_bytes,
      memory_after_materialize.hwm_bytes);

  if (prefill_shard_rebuild_.pipeline_rebuild &&
      shard_index + 1 < prefill_shards_.size() &&
      !prefill_shard_stage_pending_rebuild_.valid()) {
    const size_t next_shard_index = shard_index + 1;
    auto& next_shard = prefill_shards_[next_shard_index];
    if (next_shard.rebuild_on_execute && next_shard.module == nullptr) {
      prefill_shard_stage_pending_rebuild_index_ = next_shard_index;
      prefill_shard_stage_pending_rebuild_ = std::async(
          std::launch::async,
          [this, next_shard_index]() {
            return rebuild_prefill_shard(prefill_shards_[next_shard_index]);
          });
      ET_LOG(
          Info,
          "stage-major shard pipeline: loading/executing index=%zu while rebuilding index=%zu",
          shard_index,
          next_shard_index);
    }
  }

  ET_CHECK_OK_OR_RETURN_ERROR(load_prefill_shard_method(shard));
  const auto method_meta = shard.module->method_meta(shard.method_name);
  ET_CHECK_OK_OR_RETURN_ERROR(method_meta.error());
  ET_CHECK_MSG(
      method_meta->num_inputs() == shard.input_bindings.size(),
      "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu inputs but the "
      "configured static plan provides %zu. Check prefill_outputs_logits in the "
      "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
      shard_index,
      shard.method_name.c_str(),
      method_meta->num_inputs(),
      shard.input_bindings.size());
  ET_CHECK_MSG(
      method_meta->num_outputs() == shard.output_bindings.size(),
      "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu outputs but the "
      "configured static plan provides %zu. Check prefill_outputs_logits in the "
      "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
      shard_index,
      shard.method_name.c_str(),
      method_meta->num_outputs(),
      shard.output_bindings.size());
  const auto memory_after_load = process_memory_snapshot();
  record_memory_peak(
      &shard.runtime_stats.rss_after_load_bytes,
      memory_after_load.rss_bytes);
  record_memory_peak(
      &shard.runtime_stats.hwm_after_load_bytes,
      memory_after_load.hwm_bytes);

  const auto output_binding_start = SteadyClock::now();
  for (size_t i = 0; i < shard.output_tensors.size(); ++i) {
    ET_CHECK_OK_OR_RETURN_ERROR(
        shard.module->set_output(shard.method_name, shard.output_tensors[i], i));
  }
  shard.runtime_stats.output_binding_ms += elapsed_ms(output_binding_start);
  ET_LOG(
      Info,
      "prefill shard stage begin: index=%zu layers=[%zu,%zu)",
      shard_index,
      shard.layer_offset,
      shard.layer_offset + shard.layer_count);
  return Error::Ok;
}

Result<DecoderRunner::PrefillShardStageState>
DecoderRunner::execute_prefill_shard(
    PrefillShardPlan& shard,
    std::vector<EValue>& inputs,
    const PrefillShardStageState* previous_stage) {
  const size_t full_mask_index = 1;
  const size_t full_window_mask_index =
      prefill_has_window_attention_mask_ ? 2 : static_cast<size_t>(-1);
  const size_t full_pos_index = prefill_has_window_attention_mask_ ? 3 : 2;
  const size_t full_k_base = full_pos_index + 1;
  const size_t full_v_base = full_k_base + static_cast<size_t>(prefill_num_layers_);

  const auto input_binding_start = SteadyClock::now();
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
            full_window_mask_index != static_cast<size_t>(-1),
            "Missing window attention mask");
        shard_inputs.emplace_back(inputs[full_window_mask_index]);
        break;
      case PrefillShardPlan::InputKind::Position:
        shard_inputs.emplace_back(inputs[full_pos_index]);
        break;
      case PrefillShardPlan::InputKind::KCache:
        shard_inputs.emplace_back(
            inputs[full_k_base + shard.layer_offset + binding.index]);
        break;
      case PrefillShardPlan::InputKind::VCache:
        shard_inputs.emplace_back(
            inputs[full_v_base + shard.layer_offset + binding.index]);
        break;
      case PrefillShardPlan::InputKind::PreviousAux:
        ET_CHECK_MSG(
            previous_stage != nullptr && binding.index < previous_stage->aux.size(),
            "Missing stage-major auxiliary output %zu",
            binding.index);
        shard_inputs.emplace_back(*previous_stage->aux[binding.index]);
        break;
      case PrefillShardPlan::InputKind::PreviousHidden:
        ET_CHECK_MSG(
            previous_stage != nullptr && previous_stage->hidden != nullptr,
            "Missing stage-major hidden output");
        shard_inputs.emplace_back(*previous_stage->hidden);
        break;
    }
  }
  shard.runtime_stats.input_binding_ms += elapsed_ms(input_binding_start);

  if (shard_tensor_dump_enabled() &&
      shard_tensor_dump_matches_layer_offset(shard.layer_offset)) {
    for (size_t i = 0; i < shard.input_bindings.size(); ++i) {
      if (!shard_tensor_dump_matches_input_binding(i)) {
        continue;
      }
      const auto& binding = shard.input_bindings[i];
      const char* kind = nullptr;
      switch (binding.kind) {
        case PrefillShardPlan::InputKind::Tokens:
          // With the separate embedding export, this is the embedded hidden state.
          kind = "tokens_or_embedded_hidden";
          break;
        case PrefillShardPlan::InputKind::Position:
          kind = "position";
          break;
        case PrefillShardPlan::InputKind::PreviousAux:
          kind = "previous_aux";
          break;
        case PrefillShardPlan::InputKind::PreviousHidden:
          kind = "previous_hidden";
          break;
        case PrefillShardPlan::InputKind::AttentionMask:
          if (shard_tensor_dump_all_inputs_enabled()) {
            kind = "attention_mask";
          }
          break;
        case PrefillShardPlan::InputKind::WindowAttentionMask:
          if (shard_tensor_dump_all_inputs_enabled()) {
            kind = "window_attention_mask";
          }
          break;
        case PrefillShardPlan::InputKind::KCache:
          if (shard_tensor_dump_all_inputs_enabled()) {
            kind = "k_cache";
          }
          break;
        case PrefillShardPlan::InputKind::VCache:
          if (shard_tensor_dump_all_inputs_enabled()) {
            kind = "v_cache";
          }
          break;
      }
      if (kind != nullptr) {
        ET_LOG(
            Info,
            "prefill raw input dump begin: shard=%zu binding=%zu kind=%s",
            shard.layer_offset,
            i,
            kind);
        const bool raw_only = shard_tensor_dump_all_inputs_enabled();
        // Dump the exact EValue passed to Method::execute(). Redirecting this
        // diagnostic read to the top-level input vector is unsafe because its
        // TensorImpl metadata may already have been rebound for shard execution.
        const Tensor& tensor = shard_inputs[i].toTensor();
        if (raw_only) {
          dump_prefill_shard_tensor_bytes(
              shard.layer_offset,
              shard.runtime_stats.execution_count,
              "input",
              kind,
              i,
              tensor);
        } else {
          dump_prefill_shard_tensor(
              shard.layer_offset,
              shard.runtime_stats.execution_count,
              "input",
              kind,
              i,
              tensor);
        }
        ET_LOG(
            Info,
            "prefill raw input dump complete: shard=%zu binding=%zu kind=%s bytes=%zu",
            shard.layer_offset,
            i,
            kind,
            tensor.nbytes());
      }
    }
  }

  const auto execute_start = SteadyClock::now();
  auto outputs_res = shard.module->execute(shard.method_name, shard_inputs);
  shard.runtime_stats.execute_ms += elapsed_ms(execute_start);
  const auto memory_after_execute = process_memory_snapshot();
  record_memory_peak(
      &shard.runtime_stats.rss_after_execute_bytes,
      memory_after_execute.rss_bytes);
  record_memory_peak(
      &shard.runtime_stats.hwm_after_execute_bytes,
      memory_after_execute.hwm_bytes);
  ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
  ET_CHECK_MSG(
      outputs_res->size() == shard.output_tensors.size(),
      "Shard output count mismatch: returned=%zu bound=%zu",
      outputs_res->size(),
      shard.output_tensors.size());
  const auto output_copy_start = SteadyClock::now();
  for (size_t i = 0; i < outputs_res->size(); ++i) {
    ET_CHECK_MSG(
        outputs_res.get()[i].isTensor(),
        "Non Tensor Output returned from prefill shard");
    if (shard_tensor_dump_enabled() &&
        shard.output_bindings[i].kind !=
            PrefillShardPlan::OutputKind::FinalKCache &&
        shard.output_bindings[i].kind !=
            PrefillShardPlan::OutputKind::FinalVCache) {
      dump_prefill_shard_tensor(
          shard.layer_offset,
          shard.runtime_stats.execution_count,
          "runtime_output",
          "pre_copy",
          i,
          outputs_res.get()[i].toTensor());
    }
    copy_tensor_data(outputs_res.get()[i].toTensor(), shard.output_tensors[i]);
  }
  shard.runtime_stats.output_copy_ms += elapsed_ms(output_copy_start);
  ET_CHECK_OK_OR_RETURN_ERROR(write_prefill_etdump(shard));

  if (shard_tensor_dump_enabled()) {
    for (size_t i = 0; i < shard.output_bindings.size(); ++i) {
      const auto& binding = shard.output_bindings[i];
      const char* kind = nullptr;
      switch (binding.kind) {
        case PrefillShardPlan::OutputKind::IntermediateAux:
          kind = "intermediate_aux";
          break;
        case PrefillShardPlan::OutputKind::IntermediateHidden:
          kind = "intermediate_hidden";
          break;
        case PrefillShardPlan::OutputKind::FinalLogits:
          kind = "final_logits";
          break;
        case PrefillShardPlan::OutputKind::FinalKCache:
          if (shard_tensor_dump_kv_enabled()) {
            kind = "final_k_cache";
          }
          break;
        case PrefillShardPlan::OutputKind::FinalVCache:
          if (shard_tensor_dump_kv_enabled()) {
            kind = "final_v_cache";
          }
          break;
      }
      if (kind != nullptr) {
        dump_prefill_shard_tensor(
            shard.layer_offset,
            shard.runtime_stats.execution_count,
            "output",
            kind,
            i,
            shard.output_tensors[i]);
      }
    }
  }

  PrefillShardStageState next_stage;
  if (previous_stage != nullptr) {
    next_stage.aux = previous_stage->aux;
  }
  for (const auto& binding : shard.output_bindings) {
    if (binding.kind == PrefillShardPlan::OutputKind::IntermediateAux) {
      next_stage.aux.push_back(
          clone_tensor(*shard.owned_outputs[binding.owned_index]));
    } else if (binding.kind == PrefillShardPlan::OutputKind::IntermediateHidden) {
      next_stage.hidden = clone_tensor(*shard.owned_outputs[binding.owned_index]);
    }
  }
  if (previous_stage == nullptr && prefill_shard_swap_aux() &&
      next_stage.aux.size() == 2) {
    std::swap(next_stage.aux[0], next_stage.aux[1]);
  }
  return next_stage;
}

Result<DecoderRunner::PrefillShardStageState>
DecoderRunner::step_prefill_shard_stage(
    size_t shard_index,
    std::vector<EValue>& inputs,
    const PrefillShardStageState* previous_stage) {
  ET_CHECK_MSG(
      uses_prefill_shard_stage_major(),
      "Stage-major prefill is not enabled");
  ET_CHECK_MSG(shard_index < prefill_shards_.size(), "Invalid prefill shard index %zu", shard_index);
  auto& shard = prefill_shards_[shard_index];
  ET_CHECK_MSG(shard.module != nullptr, "Prefill shard %zu was not started", shard_index);
  ++shard.runtime_stats.execution_count;
  return execute_prefill_shard(shard, inputs, previous_stage);
}

Error DecoderRunner::end_prefill_shard_stage(size_t shard_index) {
  ET_CHECK_MSG(shard_index < prefill_shards_.size(), "Invalid prefill shard index %zu", shard_index);
  ET_CHECK_MSG(
      prefill_shard_stage_starts_.size() == prefill_shards_.size(),
      "Prefill shard stage timing is not initialized");
  auto& shard = prefill_shards_[shard_index];
  ET_CHECK_MSG(shard.module != nullptr, "Prefill shard %zu was not started", shard_index);
  if (uses_prefill_shard_three_stage_pipeline() &&
      shard_index + 1 < prefill_shards_.size()) {
    auto& next_shard = prefill_shards_[shard_index + 1];
    next_shard.runtime_stats.pipeline_wait_ms +=
        wait_for_prefill_shard_three_stage_load(shard_index + 1);
  }
  const auto release_start = SteadyClock::now();
  release_prefill_shard(shard);
  shard.runtime_stats.release_ms += elapsed_ms(release_start);
  const auto memory_after_release = process_memory_snapshot();
  record_memory_peak(
      &shard.runtime_stats.rss_after_release_bytes,
      memory_after_release.rss_bytes);
  record_memory_peak(
      &shard.runtime_stats.hwm_after_release_bytes,
      memory_after_release.hwm_bytes);
  shard.runtime_stats.total_ms += elapsed_ms(prefill_shard_stage_starts_[shard_index]);
  ET_LOG(
      Info,
      "prefill shard stage end: index=%zu runs=%zu",
      shard_index,
      shard.runtime_stats.execution_count);
  if (uses_prefill_shard_three_stage_pipeline() &&
      shard_index + 1 == prefill_shards_.size()) {
    stop_prefill_shard_three_stage_pipeline();
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
  if (prefill_qwen3_static_plan_) {
    ET_LOG(
        Info,
        "qwen3 static prefill shard IO mode=float_tensor_metadata_raw_fp16_storage");
  }
  std::future<PteRebuildResult> pending_rebuild;
  size_t pending_rebuild_index = std::numeric_limits<size_t>::max();
  for (size_t shard_index = 0; shard_index < prefill_shards_.size(); ++shard_index) {
    auto& shard = prefill_shards_[shard_index];
    const auto shard_total_start = SteadyClock::now();
    ++shard.runtime_stats.execution_count;
    const auto memory_before = process_memory_snapshot();
    if (shard.runtime_stats.execution_count == 1) {
      shard.runtime_stats.rss_before_bytes = memory_before.rss_bytes;
      shard.runtime_stats.hwm_before_bytes = memory_before.hwm_bytes;
    }
    if (pending_rebuild.valid() && pending_rebuild_index == shard_index) {
      const auto pipeline_wait_start = SteadyClock::now();
      PteRebuildResult rebuild_result = pending_rebuild.get();
      shard.runtime_stats.pipeline_wait_ms += elapsed_ms(pipeline_wait_start);
      const auto materialize_start = SteadyClock::now();
      attach_rebuilt_prefill_shard(shard, std::move(rebuild_result));
      shard.runtime_stats.materialize_ms += elapsed_ms(materialize_start);
      pending_rebuild_index = std::numeric_limits<size_t>::max();
    } else {
      materialize_prefill_shard(shard);
    }
    const auto memory_after_materialize = process_memory_snapshot();
    record_memory_peak(
        &shard.runtime_stats.rss_after_materialize_bytes,
        memory_after_materialize.rss_bytes);
    record_memory_peak(
        &shard.runtime_stats.hwm_after_materialize_bytes,
        memory_after_materialize.hwm_bytes);
    if (prefill_shard_rebuild_.pipeline_rebuild &&
        shard_index + 1 < prefill_shards_.size() &&
        !pending_rebuild.valid()) {
      const size_t next_shard_index = shard_index + 1;
      auto& next_shard = prefill_shards_[next_shard_index];
      if (next_shard.rebuild_on_execute && next_shard.module == nullptr) {
        pending_rebuild_index = next_shard_index;
        pending_rebuild = std::async(
            std::launch::async,
            [this, next_shard_index]() {
              return rebuild_prefill_shard(prefill_shards_[next_shard_index]);
            });
        ET_LOG(
            Info,
            "prefill shard pipeline: loading/executing index=%zu while rebuilding index=%zu",
            shard_index,
            next_shard_index);
      }
    }
    ET_CHECK_OK_OR_RETURN_ERROR(load_prefill_shard_method(shard));
    const auto method_meta = shard.module->method_meta(shard.method_name);
    ET_CHECK_OK_OR_RETURN_ERROR(method_meta.error());
    ET_CHECK_MSG(
        method_meta->num_inputs() == shard.input_bindings.size(),
        "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu inputs but the "
        "configured static plan provides %zu. Check prefill_outputs_logits in the "
        "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
        shard_index,
        shard.method_name.c_str(),
        method_meta->num_inputs(),
        shard.input_bindings.size());
    ET_CHECK_MSG(
        method_meta->num_outputs() == shard.output_bindings.size(),
        "Prefill shard ABI mismatch: shard=%zu method=%s expects %zu outputs but the "
        "configured static plan provides %zu. Check prefill_outputs_logits in the "
        "shard manifest or use --prefill_force_logits only with logits-retaining PTEs.",
        shard_index,
        shard.method_name.c_str(),
        method_meta->num_outputs(),
        shard.output_bindings.size());
    const auto memory_after_load = process_memory_snapshot();
    record_memory_peak(
        &shard.runtime_stats.rss_after_load_bytes,
        memory_after_load.rss_bytes);
    record_memory_peak(
        &shard.runtime_stats.hwm_after_load_bytes,
        memory_after_load.hwm_bytes);

    // PromptProcessor provides Float tensor metadata backed by the native FP16
    // QNN buffers, matching the non-PD runner. No conversion bridge is needed.

    // Populate shard inputs (after owned inputs have been set up).
    const auto input_binding_start = SteadyClock::now();
    std::vector<EValue> shard_inputs;
    shard_inputs.reserve(shard.input_bindings.size());
    for (const auto& binding : shard.input_bindings) {
      switch (binding.kind) {
        case PrefillShardPlan::InputKind::Tokens:
          shard_inputs.emplace_back(inputs[0]);
          break;
        case PrefillShardPlan::InputKind::AttentionMask:
          if (binding.owned_index != static_cast<size_t>(-1)) {
            const Tensor& src = inputs[full_mask_index].toTensor();
            Tensor& dst = *shard.owned_inputs[binding.owned_index];
            const uint16_t* sd = src.const_data_ptr<uint16_t>();
            float* dd = dst.mutable_data_ptr<float>();
            for (size_t j = 0; j < static_cast<size_t>(src.numel()); ++j)
              dd[j] = sd[j] == 0 ? -100.0f : 0.0f;
            shard_inputs.emplace_back(dst);
          } else {
            shard_inputs.emplace_back(inputs[full_mask_index]);
          }
          break;
        case PrefillShardPlan::InputKind::WindowAttentionMask:
          if (binding.owned_index != static_cast<size_t>(-1)) {
            const Tensor& src = inputs[full_window_mask_index].toTensor();
            Tensor& dst = *shard.owned_inputs[binding.owned_index];
            const uint16_t* sd = src.const_data_ptr<uint16_t>();
            float* dd = dst.mutable_data_ptr<float>();
            for (size_t j = 0; j < static_cast<size_t>(src.numel()); ++j)
              dd[j] = sd[j] == 0 ? -100.0f : 0.0f;
            shard_inputs.emplace_back(dst);
          } else {
            shard_inputs.emplace_back(inputs[full_window_mask_index]);
          }
          break;
        case PrefillShardPlan::InputKind::Position:
          shard_inputs.emplace_back(inputs[full_pos_index]);
          break;
        case PrefillShardPlan::InputKind::KCache:
          if (binding.owned_index != static_cast<size_t>(-1)) {
            const Tensor& src =
                inputs[full_k_base + shard.layer_offset + binding.index].toTensor();
            Tensor& dst = *shard.owned_inputs[binding.owned_index];
            float* dd = dst.mutable_data_ptr<float>();
            const uint16_t* sd = src.const_data_ptr<uint16_t>();
            for (size_t j = 0; j < static_cast<size_t>(dst.numel()); ++j) {
              uint16_t b = sd[j];
              uint32_t sign = (b & 0x8000u) << 16;
              uint32_t e = (b >> 10) & 0x1Fu;
              uint32_t m = b & 0x3FFu;
              uint32_t v;
              if (e == 0) v = sign;
              else if (e == 0x1F) v = sign | 0x7F800000u | (m << 13);
              else v = sign | ((e - 15 + 127) << 23) | (m << 13);
              std::memcpy(&dd[j], &v, 4);
            }
            shard_inputs.emplace_back(dst);
          } else {
            shard_inputs.emplace_back(
                inputs[full_k_base + shard.layer_offset + binding.index]);
          }
          break;
        case PrefillShardPlan::InputKind::VCache:
          if (binding.owned_index != static_cast<size_t>(-1)) {
            const Tensor& src =
                inputs[full_v_base + shard.layer_offset + binding.index].toTensor();
            Tensor& dst = *shard.owned_inputs[binding.owned_index];
            float* dd = dst.mutable_data_ptr<float>();
            const uint16_t* sd = src.const_data_ptr<uint16_t>();
            for (size_t j = 0; j < static_cast<size_t>(dst.numel()); ++j) {
              uint16_t b = sd[j];
              uint32_t sign = (b & 0x8000u) << 16;
              uint32_t e = (b >> 10) & 0x1Fu;
              uint32_t m = b & 0x3FFu;
              uint32_t v;
              if (e == 0) v = sign;
              else if (e == 0x1F) v = sign | 0x7F800000u | (m << 13);
              else v = sign | ((e - 15 + 127) << 23) | (m << 13);
              std::memcpy(&dd[j], &v, 4);
            }
            shard_inputs.emplace_back(dst);
          } else {
            shard_inputs.emplace_back(
                inputs[full_v_base + shard.layer_offset + binding.index]);
          }
          break;
        case PrefillShardPlan::InputKind::PreviousAux:
          ET_CHECK_MSG(
              binding.index < previous_aux.size(),
              "Missing previous shard aux tensor %zu", binding.index);
          shard_inputs.emplace_back(previous_aux[binding.index]);
          break;
        case PrefillShardPlan::InputKind::PreviousHidden:
          ET_CHECK_MSG(previous_hidden != nullptr, "Missing previous shard hidden state");
          shard_inputs.emplace_back(*previous_hidden);
          break;
      }
    }
    shard.runtime_stats.input_binding_ms += elapsed_ms(input_binding_start);

    const auto output_binding_start = SteadyClock::now();
    for (size_t i = 0; i < shard.output_tensors.size(); ++i) {
      ET_CHECK_OK_OR_RETURN_ERROR(
          shard.module->set_output(shard.method_name, shard.output_tensors[i], i));
    }
    shard.runtime_stats.output_binding_ms += elapsed_ms(output_binding_start);

    const auto shard_execute_start = SteadyClock::now();
    auto outputs_res = shard.module->execute(shard.method_name, shard_inputs);
    const double shard_execute_ms = elapsed_ms(shard_execute_start);
    shard.runtime_stats.execute_ms += shard_execute_ms;
    const auto memory_after_execute = process_memory_snapshot();
    record_memory_peak(
        &shard.runtime_stats.rss_after_execute_bytes,
        memory_after_execute.rss_bytes);
    record_memory_peak(
        &shard.runtime_stats.hwm_after_execute_bytes,
        memory_after_execute.hwm_bytes);
    ET_CHECK_OK_OR_RETURN_ERROR(outputs_res.error());
    ET_CHECK_MSG(
        outputs_res->size() == shard.output_tensors.size(),
        "Shard output count mismatch: returned=%zu bound=%zu",
        outputs_res->size(),
        shard.output_tensors.size());
    const auto output_copy_start = SteadyClock::now();
    for (size_t i = 0; i < outputs_res->size(); ++i) {
      ET_CHECK_MSG(
          outputs_res.get()[i].isTensor(),
          "Non Tensor Output returned from prefill shard");
      copy_tensor_data(outputs_res.get()[i].toTensor(), shard.output_tensors[i]);
    }
    shard.runtime_stats.output_copy_ms += elapsed_ms(output_copy_start);
    ET_CHECK_OK_OR_RETURN_ERROR(write_prefill_etdump(shard));

    if (std::getenv("ET_SHARD_OUTPUT_DIAG") != nullptr) {

      for (size_t oi = 0; oi < shard.output_bindings.size(); ++oi) {
        const auto& b = shard.output_bindings[oi];
        if (b.kind != PrefillShardPlan::OutputKind::FinalKCache &&
            b.kind != PrefillShardPlan::OutputKind::FinalVCache)
          continue;
        const Tensor& t = shard.output_tensors[oi];
        const size_t n = std::min<size_t>(8, static_cast<size_t>(t.numel()));
        std::ostringstream oss;
        if (t.scalar_type() == executorch::aten::ScalarType::Float ||
            t.scalar_type() == executorch::aten::ScalarType::Half) {
          const uint16_t* p = t.const_data_ptr<uint16_t>();
          for (size_t v = 0; v < n; ++v) {
            if (v) oss << " ";
            executorch::aten::Half fp16;
            std::memcpy(&fp16, &p[v], sizeof(p[v]));
            oss << static_cast<float>(fp16);
          }
        } else {
          oss << "(scalar=" << static_cast<int>(t.scalar_type()) << ")";
        }
        ET_LOG(Info,
            "shard_output_diag layer_offset=%zu kind=%s sample=[%s]",
            shard.layer_offset,
            b.kind == PrefillShardPlan::OutputKind::FinalKCache ? "K" : "V",
            oss.str().c_str());
      }
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
    const auto release_start = SteadyClock::now();
    release_prefill_shard(shard);
    shard.runtime_stats.release_ms += elapsed_ms(release_start);
    const auto memory_after_release = process_memory_snapshot();
    record_memory_peak(
        &shard.runtime_stats.rss_after_release_bytes,
        memory_after_release.rss_bytes);
    record_memory_peak(
        &shard.runtime_stats.hwm_after_release_bytes,
        memory_after_release.hwm_bytes);
    shard.runtime_stats.total_ms += elapsed_ms(shard_total_start);
  }
  return prefill_output_values_[0];
}

std::vector<DecoderRunner::PrefillShardRuntimeStats>
DecoderRunner::prefill_shard_runtime_stats() const {
  std::vector<PrefillShardRuntimeStats> stats;
  stats.reserve(prefill_shards_.size());
  for (const auto& shard : prefill_shards_) {
    stats.push_back(shard.runtime_stats);
  }
  return stats;
}

double DecoderRunner::prefill_qnn_backend_prewarm_ms() const {
  return prefill_qnn_backend_prewarm_ms_;
}

bool DecoderRunner::prefill_qnn_backend_prewarmed() const {
  return prefill_qnn_backend_prewarmed_;
}

bool DecoderRunner::uses_prefill_shards() const {
  return !prefill_shards_.empty();
}

} // namespace example
