#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/client_mem.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pd_runner.h>
#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/extension/llm/runner/llm_runner_helper.h>
#include <executorch/extension/llm/runner/util.h>
#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/schema/program_generated.h>
#include <nlohmann/json.hpp>
#include <pytorch/tokenizers/hf_tokenizer.h>
#include <pytorch/tokenizers/llama2c_tokenizer.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using executorch::extension::llm::time_in_ms;
using executorch::runtime::Error;
using executorch::runtime::MethodMeta;
using executorch::runtime::Result;
namespace llm = ::executorch::extension::llm;
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace example {
namespace {

double nll_from_quantized_logits(
    const uint16_t* logits,
    int32_t vocab_size,
    uint64_t target_token,
    float logits_scale,
    int32_t logits_zero_point) {
  ET_CHECK_MSG(logits != nullptr, "logits cannot be null");
  ET_CHECK_MSG(logits_scale > 0.0f, "logits_scale must be positive");
  ET_CHECK_MSG(
      target_token < static_cast<uint64_t>(vocab_size),
      "target token is outside vocabulary");
  double max_logit = -std::numeric_limits<double>::infinity();
  for (int32_t i = 0; i < vocab_size; ++i) {
    const double value =
        (static_cast<double>(logits[i]) - logits_zero_point) * logits_scale;
    max_logit = std::max(max_logit, value);
  }
  double exp_sum = 0.0;
  for (int32_t i = 0; i < vocab_size; ++i) {
    const double value =
        (static_cast<double>(logits[i]) - logits_zero_point) * logits_scale;
    exp_sum += std::exp(value - max_logit);
  }
  const double target =
      (static_cast<double>(logits[target_token]) - logits_zero_point) *
      logits_scale;
  return max_logit + std::log(exp_sum) - target;
}

int32_t argmax_u16_logits(const uint16_t* logits, int32_t vocab_size) {
  return static_cast<int32_t>(
      std::max_element(logits, logits + vocab_size) - logits);
}

struct KvQuantAttr {
  double scale{1.0};
  int32_t zero_point{0};
  std::string dtype;
  bool valid{false};
};

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(
    SteadyClock::time_point start,
    SteadyClock::time_point end = SteadyClock::now()) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool pd_debug_kv_enabled() {
  const char* value = std::getenv("ET_PD_DEBUG_KV");
  if (value == nullptr) {
    return false;
  }
  const std::string flag(value);
  return !flag.empty() && flag != "0" && flag != "false" && flag != "FALSE";
}

float dequantize_u8_value(uint8_t value, const KvQuantAttr& quant_attr) {
  return (static_cast<int32_t>(value) - quant_attr.zero_point) *
      static_cast<float>(quant_attr.scale);
}

float fp16_bits_to_float(uint16_t bits) {
  executorch::aten::Half fp16;
  static_assert(sizeof(fp16) == sizeof(bits), "Unexpected half size");
  std::memcpy(&fp16, &bits, sizeof(bits));
  return static_cast<float>(fp16);
}

bool is_power_of_two(int64_t value) {
  return value > 0 && (value & (value - 1)) == 0;
}

void apply_normalized_hadamard_inplace(std::vector<float>* values) {
  ET_CHECK_MSG(values != nullptr, "Hadamard input cannot be null");
  const size_t n = values->size();
  ET_CHECK_MSG(n > 0 && ((n & (n - 1)) == 0), "Hadamard size must be a power of two");
  for (size_t len = 1; len < n; len <<= 1) {
    for (size_t start = 0; start < n; start += (len << 1)) {
      for (size_t i = 0; i < len; ++i) {
        const float u = values->at(start + i);
        const float v = values->at(start + len + i);
        values->at(start + i) = u + v;
        values->at(start + len + i) = u - v;
      }
    }
  }
  const float scale = 1.0f / std::sqrt(static_cast<float>(n));
  for (float& value : *values) {
    value *= scale;
  }
}

bool should_undo_r3_on_export(
    DecoderModelVersion decoder_model_version,
    const std::string& model_path,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim) {
  if (decoder_model_version != DecoderModelVersion::kQwen3) {
    return false;
  }

  const bool looks_like_qwen3_1_7b_shape =
      num_layers == 28 && num_heads == 8 && head_dim == 128;
  const bool looks_like_qwen3_1_7b_path =
      model_path.find("1_7b") != std::string::npos ||
      model_path.find("1.7b") != std::string::npos ||
      model_path.find("1-7b") != std::string::npos;

  return looks_like_qwen3_1_7b_shape || looks_like_qwen3_1_7b_path;
}

std::vector<uint8_t> read_binary_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  ET_CHECK_MSG(input.is_open(), "Unable to read file: %s", path.c_str());
  return std::vector<uint8_t>(
      std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string hex_u64(uint64_t value) {
  std::ostringstream oss;
  oss << std::hex << std::setw(16) << std::setfill('0') << value;
  return oss.str();
}

uint64_t fnv1a64(const std::vector<uint8_t>& bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : bytes) {
    hash ^= static_cast<uint64_t>(byte);
    hash *= 1099511628211ull;
  }
  return hash;
}

json make_file_fingerprint(const std::string& path) {
  json out;
  out["path"] = path;
  if (path.empty() || !fs::exists(path)) {
    out["exists"] = false;
    return out;
  }
  out["exists"] = true;
  out["basename"] = fs::path(path).filename().string();
  out["size_bytes"] = static_cast<uint64_t>(fs::file_size(path));

  const char* full_hash = std::getenv("ET_PD_FULL_FILE_FINGERPRINT");
  const bool hash_contents = full_hash != nullptr && full_hash[0] != '\0' &&
      std::string(full_hash) != "0";
  out["fingerprint_mode"] = hash_contents ? "full_fnv1a64" : "metadata_only";
  if (hash_contents) {
    const auto bytes = read_binary_file(path);
    out["fnv1a64"] = hex_u64(fnv1a64(bytes));
  }
  return out;
}

std::string decoder_model_to_string(DecoderModelVersion version) {
  switch (version) {
    case DecoderModelVersion::kLlama2:
      return "llama2";
    case DecoderModelVersion::kLlama3:
      return "llama3";
    case DecoderModelVersion::kGemma:
      return "gemma";
    case DecoderModelVersion::kGemma3:
      return "gemma3";
    case DecoderModelVersion::kGranite:
      return "granite";
    case DecoderModelVersion::kPhi4:
      return "phi_4_mini";
    case DecoderModelVersion::kQwen2_5:
      return "qwen2_5";
    case DecoderModelVersion::kQwen3:
      return "qwen3";
    case DecoderModelVersion::kSmollm2_135m:
      return "smollm2_135m";
    case DecoderModelVersion::kSmollm3:
      return "smollm3";
    case DecoderModelVersion::kCodegen:
      return "codegen";
    case DecoderModelVersion::kGlm:
      return "glm";
    case DecoderModelVersion::kGemma2:
      return "gemma2";
  }
  return "unknown";
}

std::string cache_mode_to_string(CacheMode mode) {
  switch (mode) {
    case CacheMode::StaticCahce:
      return "static";
    case CacheMode::HybridCache:
      return "hybrid";
  }
  return "unknown";
}

template <typename T>
uint16_t fp16_bits_from_value(T value) {
  if constexpr (std::is_same_v<T, uint16_t>) {
    return value;
  } else {
    executorch::aten::Half fp16(static_cast<float>(value));
    uint16_t bits = 0;
    static_assert(sizeof(fp16) == sizeof(bits), "Unexpected half size");
    std::memcpy(&bits, &fp16, sizeof(bits));
    return bits;
  }
}

uint16_t fp16_bits_from_float(float value) {
  executorch::aten::Half fp16(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &fp16, sizeof(bits));
  return bits;
}

template <typename T>
void log_cache_sample(
    const char* label,
    const KVCache<T>& cache,
    int32_t layer,
    int64_t head_dim,
    int32_t cache_stride,
    int32_t prompt_len,
    const KvQuantAttr* quant_attr,
    bool is_key) {
  if (!pd_debug_kv_enabled()) {
    return;
  }
  const int32_t sample_seq = std::min<int32_t>(prompt_len, 2);
  const int64_t sample_dim = std::min<int64_t>(head_dim, 8);
  std::ostringstream header;
  header << "PD debug " << label << " layer=" << layer
         << " stride=" << cache_stride
         << " prompt_len=" << prompt_len;
  if constexpr (std::is_same_v<T, uint8_t>) {
    ET_CHECK_MSG(quant_attr != nullptr && quant_attr->valid, "Missing quant attr");
    header << " scale=" << quant_attr->scale
           << " zp=" << quant_attr->zero_point
           << " dtype=" << quant_attr->dtype;
  }
  ET_LOG(Info, "%s", header.str().c_str());

  for (int32_t seq = 0; seq < sample_seq; ++seq) {
    std::ostringstream line;
    line << "  seq=" << seq << " values=";
    for (int64_t dim = 0; dim < sample_dim; ++dim) {
      const size_t src_index = is_key
          ? static_cast<size_t>(dim) * cache_stride + seq
          : static_cast<size_t>(seq) * head_dim + dim;
      if constexpr (std::is_same_v<T, uint16_t>) {
        const uint16_t bits = cache.buffer[src_index];
        line << "[" << dim << ":fp16=0x" << std::hex << bits << std::dec
             << ",f=" << fp16_bits_to_float(bits) << "]";
      } else {
        const uint8_t raw = cache.buffer[src_index];
        const float dequantized = dequantize_u8_value(raw, *quant_attr);
        line << "[" << dim << ":q=" << static_cast<int32_t>(raw)
             << ",f=" << dequantized << "]";
      }
      if (dim + 1 < sample_dim) {
        line << " ";
      }
    }
    ET_LOG(Info, "%s", line.str().c_str());
  }
}

void log_canonical_sample(
    const std::vector<uint16_t>& canonical,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim,
    int32_t prompt_len) {
  if (!pd_debug_kv_enabled()) {
    return;
  }
  const size_t per_kind_count =
      static_cast<size_t>(num_layers) * num_heads * head_dim * prompt_len;
  const int64_t sample_dim = std::min<int64_t>(head_dim, 8);
  std::ostringstream k_line;
  k_line << "PD debug canonical K layer=0 head=0 seq=0:";
  for (int64_t dim = 0; dim < sample_dim; ++dim) {
    const uint16_t bits = canonical.at(static_cast<size_t>(dim));
    k_line << " [" << dim << ":0x" << std::hex << bits << std::dec
           << ",f=" << fp16_bits_to_float(bits) << "]";
  }
  ET_LOG(Info, "%s", k_line.str().c_str());

  std::ostringstream v_line;
  v_line << "PD debug canonical V layer=0 head=0 seq=0:";
  for (int64_t dim = 0; dim < sample_dim; ++dim) {
    const uint16_t bits = canonical.at(per_kind_count + static_cast<size_t>(dim));
    v_line << " [" << dim << ":0x" << std::hex << bits << std::dec
           << ",f=" << fp16_bits_to_float(bits) << "]";
  }
  ET_LOG(Info, "%s", v_line.str().c_str());
}

std::vector<KvQuantAttr> read_kv_quant_attrs_from_pte(
    const std::vector<uint8_t>& pte_bytes,
    size_t expected_count) {
  std::vector<KvQuantAttr> attrs(expected_count);
  const auto* program =
      executorch_flatbuffer::GetProgram(pte_bytes.data());
  ET_CHECK_MSG(program != nullptr, "Failed to parse PTE flatbuffer");
  const auto* plans = program->execution_plan();
  ET_CHECK_MSG(plans != nullptr, "PTE has no execution plan");
  const std::regex pattern("^get_kv_output_(\\d+)_quant_attr$");
  for (flatbuffers::uoffset_t i = 0; i < plans->size(); ++i) {
    const auto* plan = plans->Get(i);
    if (plan == nullptr || plan->name() == nullptr) {
      continue;
    }
    std::cmatch match;
    const std::string name = plan->name()->str();
    if (!std::regex_match(name.c_str(), match, pattern)) {
      continue;
    }
    const size_t index = static_cast<size_t>(std::stoul(match[1].str()));
    if (index >= attrs.size()) {
      continue;
    }
    const auto* values = plan->values();
    ET_CHECK_MSG(
        values != nullptr && values->size() >= 5,
        "Invalid kv quant attr method in PTE: %s",
        name.c_str());
    const auto* scale = values->Get(0)->val_as_Double();
    const auto* zero_point = values->Get(1)->val_as_Int();
    const auto* dtype = values->Get(4)->val_as_String();
    ET_CHECK_MSG(scale != nullptr, "Missing scale in %s", name.c_str());
    ET_CHECK_MSG(zero_point != nullptr, "Missing zero point in %s", name.c_str());
    attrs[index].scale = scale->double_val();
    attrs[index].zero_point = static_cast<int32_t>(zero_point->int_val());
    attrs[index].dtype = dtype != nullptr ? dtype->string_val()->str() : "";
    attrs[index].valid = true;
  }
  return attrs;
}

std::vector<KvQuantAttr> read_kv_quant_attrs_from_json(
    const std::string& path,
    size_t expected_count) {
  std::vector<KvQuantAttr> attrs(expected_count);
  if (path.empty()) {
    return attrs;
  }
  std::ifstream input(path);
  ET_CHECK_MSG(input.is_open(), "Unable to read kv quant attrs: %s", path.c_str());
  json payload = json::parse(input);
  const json* attr_entries = &payload;
  if (payload.is_object() && payload.contains("output")) {
    if (payload.contains("mode")) {
      const std::string mode = payload.at("mode").get<std::string>();
      ET_CHECK_MSG(
          mode == "prefill" || mode == "decode",
          "PD handoff export requires prefill/decode KV quant attrs, but %s reports mode=%s",
          path.c_str(),
          mode.c_str());
    }
    attr_entries = &payload.at("output");
  }
  auto assign_attr = [&](size_t index, const json& entry) {
    ET_CHECK_MSG(index < attrs.size(), "kv quant attr index out of range: %zu", index);
    attrs[index].scale = entry.at("scale").get<double>();
    attrs[index].zero_point = entry.at("zero_point").get<int32_t>();
    attrs[index].dtype = entry.value("dtype", "");
    attrs[index].valid = true;
  };
  auto classify_combined_entry_is_k = [&](const json& entry) -> bool {
    const std::string stack_trace = entry.value("stack_trace", "");
    if (stack_trace.find("k = k.transpose(2, 3)") != std::string::npos) {
      return true;
    }
    if (stack_trace.find("v = v.view") != std::string::npos &&
        stack_trace.find("transpose(1, 2)") != std::string::npos) {
      return false;
    }
    const std::string node_name = entry.value("node_name", "");
    ET_CHECK_MSG(
        !node_name.empty(),
        "Unable to classify combined kv quant attr entry without stack_trace/node_name");
    ET_CHECK_MSG(
        false,
        "Unable to classify combined kv quant attr entry %s as K or V",
        node_name.c_str());
    return false;
  };
  if (attr_entries->is_object() && attr_entries->contains("k") &&
      attr_entries->contains("v")) {
    const auto& k_entries = attr_entries->at("k");
    const auto& v_entries = attr_entries->at("v");
    ET_CHECK_MSG(
        k_entries.is_array() && v_entries.is_array(),
        "kv quant attrs explicit schema requires array-valued output.k and output.v");
    ET_CHECK_MSG(
        expected_count % 2 == 0,
        "kv quant attrs explicit schema expects even entry count, got %zu",
        expected_count);
    const size_t num_layers = expected_count / 2;
    ET_CHECK_MSG(
        k_entries.size() == num_layers && v_entries.size() == num_layers,
        "kv quant attrs explicit schema mismatch: expected %zu K and %zu V entries, got %zu K and %zu V",
        num_layers,
        num_layers,
        k_entries.size(),
        v_entries.size());
    for (size_t i = 0; i < num_layers; ++i) {
      const auto& k_entry = k_entries.at(i);
      const auto& v_entry = v_entries.at(i);
      ET_CHECK_MSG(
          !k_entry.contains("layer_index") ||
              k_entry.at("layer_index").get<size_t>() == i,
          "KV K quant attr layer_index mismatch at %zu",
          i);
      ET_CHECK_MSG(
          !v_entry.contains("layer_index") ||
              v_entry.at("layer_index").get<size_t>() == i,
          "KV V quant attr layer_index mismatch at %zu",
          i);
      assign_attr(i, k_entry);
      assign_attr(num_layers + i, v_entry);
    }
  } else if (attr_entries->is_object() && attr_entries->contains("combined")) {
    const auto& combined_entries = attr_entries->at("combined");
    ET_CHECK_MSG(
        combined_entries.is_array(),
        "kv quant attrs combined schema requires array-valued output.combined");
    ET_CHECK_MSG(
        expected_count % 2 == 0,
        "kv quant attrs combined schema expects even entry count, got %zu",
        expected_count);
    const size_t num_layers = expected_count / 2;
    ET_CHECK_MSG(
        combined_entries.size() == expected_count,
        "kv quant attrs combined schema mismatch: expected %zu entries, got %zu",
        expected_count,
        combined_entries.size());
    std::vector<bool> seen_k(num_layers, false);
    std::vector<bool> seen_v(num_layers, false);
    for (const auto& entry : combined_entries) {
      ET_CHECK_MSG(
          entry.contains("layer_index"),
          "kv quant attrs combined schema requires layer_index");
      const size_t layer_index = entry.at("layer_index").get<size_t>();
      ET_CHECK_MSG(
          layer_index < num_layers,
          "kv quant attrs combined schema layer_index out of range: %zu",
          layer_index);
      if (classify_combined_entry_is_k(entry)) {
        ET_CHECK_MSG(!seen_k[layer_index], "Duplicate K quant attr for layer %zu", layer_index);
        assign_attr(layer_index, entry);
        seen_k[layer_index] = true;
      } else {
        ET_CHECK_MSG(!seen_v[layer_index], "Duplicate V quant attr for layer %zu", layer_index);
        assign_attr(num_layers + layer_index, entry);
        seen_v[layer_index] = true;
      }
    }
    for (size_t i = 0; i < num_layers; ++i) {
      ET_CHECK_MSG(seen_k[i], "Missing K quant attr for layer %zu", i);
      ET_CHECK_MSG(seen_v[i], "Missing V quant attr for layer %zu", i);
    }
  } else if (attr_entries->is_array()) {
    for (size_t i = 0; i < attr_entries->size() && i < attrs.size(); ++i) {
      assign_attr(i, attr_entries->at(i));
    }
  } else if (attr_entries->is_object()) {
    for (const auto& item : attr_entries->items()) {
      size_t index = static_cast<size_t>(std::stoul(item.key()));
      assign_attr(index, item.value());
    }
  } else {
    ET_CHECK_MSG(false, "Unsupported kv quant attrs JSON schema");
  }
  ET_LOG(
      Info,
      "Loaded prefill KV quant attrs JSON from %s (%zu expected entries)",
      path.c_str(),
      expected_count);
  return attrs;
}

template <typename T>
void write_canonical_k_layer(
    const KVCache<T>& cache,
    int64_t num_heads,
    int64_t head_dim,
    int32_t max_cache_len,
    int32_t prompt_len,
    const KvQuantAttr* quant_attr,
    bool undo_r3,
    uint16_t* out) {
  ET_CHECK_MSG(out != nullptr, "Canonical K output cannot be null");
  std::vector<float> head_values;
  if (undo_r3) {
    ET_CHECK_MSG(
        is_power_of_two(head_dim),
        "R3 undo requires power-of-two head_dim, got %ld",
        static_cast<long>(head_dim));
    head_values.resize(static_cast<size_t>(head_dim));
  }

  for (int64_t head = 0; head < num_heads; ++head) {
    for (int32_t seq = 0; seq < prompt_len; ++seq) {
      const size_t dst_base =
          (static_cast<size_t>(head) * prompt_len + seq) * head_dim;
      if (undo_r3) {
        for (int64_t dim = 0; dim < head_dim; ++dim) {
          const size_t src_index =
              (static_cast<size_t>(head) * head_dim + dim) * max_cache_len + seq;
          if constexpr (std::is_same_v<T, uint16_t>) {
            head_values[static_cast<size_t>(dim)] =
                fp16_bits_to_float(cache.buffer[src_index]);
          } else {
            ET_CHECK_MSG(
                quant_attr != nullptr && quant_attr->valid,
                "Missing K quant attr");
            head_values[static_cast<size_t>(dim)] =
                dequantize_u8_value(cache.buffer[src_index], *quant_attr);
          }
        }
        apply_normalized_hadamard_inplace(&head_values);
        for (int64_t dim = 0; dim < head_dim; ++dim) {
          out[dst_base + dim] =
              fp16_bits_from_float(head_values[static_cast<size_t>(dim)]);
        }
      } else {
        for (int64_t dim = 0; dim < head_dim; ++dim) {
          const size_t src_index =
              (static_cast<size_t>(head) * head_dim + dim) * max_cache_len + seq;
          if constexpr (std::is_same_v<T, uint16_t>) {
            out[dst_base + dim] = cache.buffer[src_index];
          } else {
            ET_CHECK_MSG(
                quant_attr != nullptr && quant_attr->valid,
                "Missing K quant attr");
            out[dst_base + dim] = fp16_bits_from_float(
                dequantize_u8_value(cache.buffer[src_index], *quant_attr));
          }
        }
      }
    }
  }
}

template <typename T>
void write_canonical_v_layer(
    const KVCache<T>& cache,
    int64_t num_heads,
    int64_t head_dim,
    int32_t max_cache_len,
    int32_t prompt_len,
    const KvQuantAttr* quant_attr,
    uint16_t* out) {
  ET_CHECK_MSG(out != nullptr, "Canonical V output cannot be null");
  const size_t prompt_head_values =
      static_cast<size_t>(prompt_len) * head_dim;
  const size_t cache_head_values =
      static_cast<size_t>(max_cache_len) * head_dim;

  for (int64_t head = 0; head < num_heads; ++head) {
    const size_t dst_base = static_cast<size_t>(head) * prompt_head_values;
    const size_t src_base = static_cast<size_t>(head) * cache_head_values;
    if constexpr (std::is_same_v<T, uint16_t>) {
      std::memcpy(
          out + dst_base,
          cache.buffer + src_base,
          prompt_head_values * sizeof(uint16_t));
    } else {
      ET_CHECK_MSG(
          quant_attr != nullptr && quant_attr->valid,
          "Missing V quant attr");
      for (size_t i = 0; i < prompt_head_values; ++i) {
        out[dst_base + i] = fp16_bits_from_float(
            dequantize_u8_value(cache.buffer[src_base + i], *quant_attr));
      }
    }
  }
}

template <typename T>
std::vector<uint16_t> build_canonical_kv(
    KVManager<T>* kv_manager,
    DecoderModelVersion decoder_model_version,
    const std::string& model_path,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim,
    int32_t prompt_len,
    int32_t max_cache_len,
    const std::vector<KvQuantAttr>& quant_attrs) {
  const auto& k_cache = kv_manager->get_k_cache_();
  const auto& v_cache = kv_manager->get_v_cache_();
  const bool undo_r3 =
      should_undo_r3_on_export(decoder_model_version, model_path, num_layers, num_heads, head_dim);
  if (undo_r3) {
    ET_LOG(
        Info,
        "PD export will undo SpinQuant R3 on K cache before writing canonical KV");
  }

  const size_t per_layer_count =
      static_cast<size_t>(num_heads) * head_dim * prompt_len;
  const size_t per_kind_count = static_cast<size_t>(num_layers) * per_layer_count;
  std::vector<uint16_t> canonical(per_kind_count * 2);

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const KvQuantAttr* attr = nullptr;
    if constexpr (std::is_same_v<T, uint8_t>) {
      ET_CHECK_MSG(
          quant_attrs.size() >= static_cast<size_t>(num_layers) * 2,
          "Not enough KV quant attrs for %ld layers",
          num_layers);
      attr = &quant_attrs.at(static_cast<size_t>(layer));
    }
    if (layer == 0) {
      log_cache_sample(
          "export K",
          k_cache.at(static_cast<size_t>(layer)),
          static_cast<int32_t>(layer),
          head_dim,
          max_cache_len,
          prompt_len,
          attr,
          true);
    }
    write_canonical_k_layer(
        k_cache.at(static_cast<size_t>(layer)),
        num_heads,
        head_dim,
        max_cache_len,
        prompt_len,
        attr,
        undo_r3,
        canonical.data() + static_cast<size_t>(layer) * per_layer_count);
  }

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const KvQuantAttr* attr = nullptr;
    if constexpr (std::is_same_v<T, uint8_t>) {
      attr = &quant_attrs.at(static_cast<size_t>(num_layers + layer));
    }
    if (layer == 0) {
      log_cache_sample(
          "export V",
          v_cache.at(static_cast<size_t>(layer)),
          static_cast<int32_t>(layer),
          head_dim,
          max_cache_len,
          prompt_len,
          attr,
          false);
    }
    write_canonical_v_layer(
        v_cache.at(static_cast<size_t>(layer)),
        num_heads,
        head_dim,
        max_cache_len,
        prompt_len,
        attr,
        canonical.data() + per_kind_count +
            static_cast<size_t>(layer) * per_layer_count);
  }
  log_canonical_sample(canonical, num_layers, num_heads, head_dim, prompt_len);
  return canonical;
}

std::vector<uint8_t> build_qnn_u8_kv_handoff(
    KVManager<uint8_t>* kv_manager,
    int64_t num_layers,
    int64_t num_heads,
    int64_t head_dim,
    int32_t prompt_len,
    int32_t max_cache_len) {
  ET_CHECK_MSG(kv_manager != nullptr, "QNN U8 KV manager cannot be null");
  const auto& k_cache = kv_manager->get_k_cache_();
  const auto& v_cache = kv_manager->get_v_cache_();
  const size_t per_layer_count =
      static_cast<size_t>(num_heads) * prompt_len * head_dim;
  const size_t per_kind_count = static_cast<size_t>(num_layers) * per_layer_count;
  std::vector<uint8_t> direct(per_kind_count * 2);
  uint8_t* direct_k = direct.data();
  uint8_t* direct_v = direct.data() + per_kind_count;

  for (int64_t layer = 0; layer < num_layers; ++layer) {
    const auto& layer_k = k_cache.at(static_cast<size_t>(layer));
    const auto& layer_v = v_cache.at(static_cast<size_t>(layer));
    for (int64_t head = 0; head < num_heads; ++head) {
      for (int32_t seq = 0; seq < prompt_len; ++seq) {
        const size_t dst_base =
            ((static_cast<size_t>(layer) * num_heads + head) * prompt_len + seq) *
            head_dim;
        const size_t v_src_base =
            (static_cast<size_t>(head) * max_cache_len + seq) * head_dim;
        for (int64_t dim = 0; dim < head_dim; ++dim) {
          const size_t k_src =
              (static_cast<size_t>(head) * head_dim + dim) * max_cache_len + seq;
          direct_k[dst_base + dim] = layer_k.buffer[k_src];
          direct_v[dst_base + dim] = layer_v.buffer[v_src_base + dim];
        }
      }
    }
  }
  return direct;
}

void write_binary(
    const fs::path& path,
    const void* data,
    size_t size_bytes) {
  std::ofstream output(path, std::ios::binary);
  ET_CHECK_MSG(output.is_open(), "Unable to write file: %s", path.c_str());
  output.write(reinterpret_cast<const char*>(data), size_bytes);
  ET_CHECK_MSG(output.good(), "Failed to write file: %s", path.c_str());
}

} // namespace

template <typename T>
PDPrefillRunner<T>::PDPrefillRunner(
    std::unique_ptr<executorch::extension::Module> module,
    std::vector<std::string> prefill_shard_paths,
    std::vector<std::string> prefill_shard_index_paths,
    DecoderRunner::PrefillShardRebuildConfig prefill_shard_rebuild,
    bool prefill_qwen3_static_plan,
    int32_t prefill_static_aux_size,
    int32_t prefill_static_hidden_size,
    bool prefill_outputs_logits,
    bool separate_embed,
    const std::string& embedding_matrix_path,
    bool resident_embedding,
    StaticMetadata static_metadata,
    const std::string& decoder_model_version,
    const std::string& model_path,
    const std::string& tokenizer_path,
    std::shared_ptr<std::vector<uint8_t>> pte_bytes,
    const int eval_mode,
    const bool shared_buffer,
    std::unique_ptr<tokenizers::Tokenizer> tokenizer,
    std::unique_ptr<executorch::extension::Module> attention_sink_rope_module)
    : module_(std::move(module)),
      prefill_shard_paths_(std::move(prefill_shard_paths)),
      prefill_shard_index_paths_(std::move(prefill_shard_index_paths)),
      prefill_shard_rebuild_(std::move(prefill_shard_rebuild)),
      prefill_qwen3_static_plan_(prefill_qwen3_static_plan),
      prefill_static_aux_size_(prefill_static_aux_size),
      prefill_static_hidden_size_(prefill_static_hidden_size),
      prefill_outputs_logits_(prefill_outputs_logits),
      separate_embed_(separate_embed),
      embedding_matrix_path_(embedding_matrix_path),
      resident_embedding_(resident_embedding),
      static_metadata_(std::move(static_metadata)),
      model_path_(model_path),
      tokenizer_path_(tokenizer_path),
      pte_bytes_(std::move(pte_bytes)),
      eval_mode_(static_cast<EvalMode>(eval_mode)),
      shared_buffer_(shared_buffer),
      tokenizer_(std::move(tokenizer)),
      attention_sink_rope_module_(std::move(attention_sink_rope_module)) {
  if (decoder_model_version == "llama2") {
    decoder_model_version_ = DecoderModelVersion::kLlama2;
  } else if (decoder_model_version == "llama3") {
    decoder_model_version_ = DecoderModelVersion::kLlama3;
  } else if (decoder_model_version == "gemma") {
    decoder_model_version_ = DecoderModelVersion::kGemma;
  } else if (decoder_model_version == "gemma2") {
    decoder_model_version_ = DecoderModelVersion::kGemma2;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "gemma3") {
    decoder_model_version_ = DecoderModelVersion::kGemma3;
    cache_mode_ = CacheMode::HybridCache;
  } else if (decoder_model_version == "granite") {
    decoder_model_version_ = DecoderModelVersion::kGranite;
  } else if (decoder_model_version == "phi_4_mini") {
    decoder_model_version_ = DecoderModelVersion::kPhi4;
  } else if (decoder_model_version == "qwen2_5") {
    decoder_model_version_ = DecoderModelVersion::kQwen2_5;
  } else if (decoder_model_version == "qwen3") {
    decoder_model_version_ = DecoderModelVersion::kQwen3;
  } else if (decoder_model_version == "smollm2_135m") {
    decoder_model_version_ = DecoderModelVersion::kSmollm2_135m;
  } else if (decoder_model_version == "smollm3") {
    decoder_model_version_ = DecoderModelVersion::kSmollm3;
  } else if (decoder_model_version == "codegen") {
    decoder_model_version_ = DecoderModelVersion::kCodegen;
  } else if (decoder_model_version == "glm") {
    decoder_model_version_ = DecoderModelVersion::kGlm;
  } else {
    ET_CHECK_MSG(false, "Unsupported Decoder Model");
  }
}

template <typename T>
bool PDPrefillRunner<T>::is_loaded() const {
  const bool module_ready =
      static_metadata_.enabled || (module_ != nullptr && module_->is_loaded());
  return module_ready && tokenizer_ && decoder_runner_ && prompt_processor_ &&
      kv_manager_ && buffer_manager_;
}

template <typename T>
Error PDPrefillRunner<T>::load() {
  if (is_loaded()) {
    return Error::Ok;
  }

  std::string prompt_processor_method_name;
  if (eval_mode_ == EvalMode::kKVCached) {
    prompt_processor_method_name = "kv_forward";
  } else if (
      eval_mode_ == EvalMode::kHybrid ||
      eval_mode_ == EvalMode::kLookaheadDecoding) {
    prompt_processor_method_name = "prefill_forward";
  } else {
    ET_CHECK_MSG(false, "Unsupported llama evaluation mode");
  }

  if (tokenizer_ == nullptr) {
    tokenizer_ = llm::load_tokenizer(tokenizer_path_);
    if (tokenizer_ == nullptr) {
      ET_LOG(Error, "Failed to load tokenizer with %s", tokenizer_path_.c_str());
      return Error::Internal;
    }
  }

  bool use_int64_token = false;
  int32_t sliding_window = 0;
  int32_t embedding_dim = 0;
  size_t embedding_row_bytes = 0;
  executorch::aten::ScalarType embedding_scalar_type =
      executorch::aten::ScalarType::Float;
  if (static_metadata_.enabled) {
    ET_CHECK_MSG(
        !prefill_shard_paths_.empty(),
        "Manifest-only PD prefill requires prefill shard paths");
    ET_CHECK_MSG(
        !shared_buffer_,
        "Manifest-only PD prefill currently requires --shared_buffer=false");
    vocab_size_ = static_metadata_.vocab_size;
    num_layers_ = static_metadata_.num_layers;
    num_heads_ = static_metadata_.num_heads;
    head_dim_ = static_metadata_.head_dim;
    context_len_ = static_metadata_.context_len;
    prompt_processor_ar_len_ = static_metadata_.prompt_ar_len;
    token_generator_ar_len_ = static_metadata_.token_generator_ar_len;
    use_int64_token = static_metadata_.use_int64_token;
    cache_mode_ = static_metadata_.cache_mode;
    prefill_outputs_logits_ = static_metadata_.outputs_logits;
    separate_embed_ = static_metadata_.use_separate_embed;
    embedding_matrix_path_ = static_metadata_.embedding_matrix_path;
    resident_embedding_ = static_metadata_.resident_embedding;
    sliding_window = static_metadata_.sliding_window > 0
        ? static_metadata_.sliding_window
        : context_len_;
    ET_CHECK_MSG(
        vocab_size_ > 0 && num_layers_ > 0 && num_heads_ > 0 && head_dim_ > 0 &&
            context_len_ > 0 && prompt_processor_ar_len_ > 0,
        "Invalid manifest-only PD metadata");
    ET_LOG(
        Info,
        "using manifest-only PD prefill metadata: layers=%lld ctx=%d ar=%d heads=%lld head_dim=%lld vocab=%d",
        static_cast<long long>(num_layers_),
        context_len_,
        prompt_processor_ar_len_,
        static_cast<long long>(num_heads_),
        static_cast<long long>(head_dim_),
        vocab_size_);
  } else {
    ET_CHECK_MSG(module_ != nullptr, "PDPrefillRunner requires a module without static metadata");
    Result<MethodMeta> method_meta = module_->method_meta(prompt_processor_method_name);
    vocab_size_ = method_meta->output_tensor_meta(0)->sizes()[2];

    num_layers_ = ET_UNWRAP(module_->get("get_n_layers")).toScalar().to<int64_t>();
    ET_CHECK_MSG(num_layers_ != -1, "Could not retrieve num layers");

    auto k_cache_shape = method_meta->output_tensor_meta(1)->sizes();
    num_heads_ = k_cache_shape[1];
    head_dim_ = k_cache_shape[2];
    use_int64_token = method_meta->input_tensor_meta(0)->scalar_type() ==
        executorch::aten::ScalarType::Long;

    auto atten_mask_meta_prompt = method_meta->input_tensor_meta(1);
    prompt_processor_ar_len_ = atten_mask_meta_prompt->sizes()[1];
    context_len_ = atten_mask_meta_prompt->sizes()[2];

    token_generator_ar_len_ = prompt_processor_ar_len_;
    if (module_->method_names()->count("kv_forward") > 0) {
      auto atten_mask_meta_token =
          module_->method_meta("kv_forward")->input_tensor_meta(1);
      token_generator_ar_len_ = atten_mask_meta_token->sizes()[1];
    }
  }

  if (separate_embed_) {
    ET_CHECK_MSG(
        static_metadata_.enabled,
        "Separate embedding currently requires manifest-only static prefill");
    ET_CHECK_MSG(
        !embedding_matrix_path_.empty(),
        "Separate embedding requires an embedding matrix path");
    ET_CHECK_MSG(
        separate_embedding_.load(embedding_matrix_path_, resident_embedding_),
        "Failed to load separate embedding matrix: %s",
        embedding_matrix_path_.c_str());
    ET_CHECK_MSG(
        separate_embedding_.vocab_size() == vocab_size_,
        "Separate embedding vocab mismatch: matrix=%d model=%d",
        separate_embedding_.vocab_size(),
        vocab_size_);
    ET_CHECK_MSG(
        separate_embedding_.embedding_dim() == prefill_static_hidden_size_,
        "Separate embedding dim mismatch: matrix=%d graph=%d",
        separate_embedding_.embedding_dim(),
        prefill_static_hidden_size_);
    embedding_dim = separate_embedding_.embedding_dim();
    ET_CHECK_MSG(
        separate_embedding_.embedding_dtype_code() == 1 ||
            separate_embedding_.embedding_dtype_code() == 2,
        "Separate embedding supports only float32/float16 storage, got dtype code %u",
        separate_embedding_.embedding_dtype_code());
    embedding_row_bytes = static_cast<size_t>(embedding_dim) * sizeof(float);
    embedding_scalar_type = executorch::aten::ScalarType::Float;
  }

  decoder_runner_ = std::make_unique<DecoderRunner>(
      module_.get(),
      vocab_size_,
      0.0f,
      prefill_shard_paths_,
      prefill_shard_index_paths_,
      prefill_shard_rebuild_);
  decoder_runner_->use_qwen3_prefill_static_plan(
      prefill_qwen3_static_plan_,
      prefill_static_aux_size_,
      prefill_static_hidden_size_);
  decoder_runner_->set_prefill_outputs_logits(prefill_outputs_logits_);
  decoder_runner_->set_prefill_separate_embed(separate_embed_);
  decoder_runner_->set_prefill_etdump_config(prefill_etdump_config_);

  int32_t max_cache_len = prompt_processor_ar_len_ == context_len_
      ? context_len_
      : context_len_ - std::min(prompt_processor_ar_len_, token_generator_ar_len_);
  int32_t max_ar_len = std::max(prompt_processor_ar_len_, token_generator_ar_len_);
  max_cache_len_ = max_cache_len;
  prefill_cache_stride_ = prompt_processor_ar_len_ == context_len_
      ? context_len_
      : context_len_ - prompt_processor_ar_len_;

  decoder_runner_->configure_prefill_shards(
      num_layers_,
      context_len_,
      prompt_processor_ar_len_,
      vocab_size_,
      cache_mode_ == CacheMode::HybridCache);
  ET_CHECK_OK_OR_RETURN_ERROR(decoder_runner_->load({prompt_processor_method_name}));

  if (!static_metadata_.enabled) {
    sliding_window = context_len_;
    if (module_->method_names()->count("get_sliding_window") > 0) {
      sliding_window = ET_UNWRAP(module_->get("get_sliding_window")).toInt();
    }
  }

  kv_manager_ = std::make_unique<KVManager<T>>(typename KVManager<T>::Metadata{
      context_len_,
      head_dim_,
      max_ar_len,
      max_cache_len,
      num_heads_,
      num_layers_});

  if (attention_sink_rope_module_ != nullptr) {
    attention_sink_rope_runner_ = std::make_unique<AttentionSinkRopeRunner>(
        attention_sink_rope_module_.get());
    ET_CHECK_OK_OR_RETURN_ERROR(
        attention_sink_rope_runner_->load({prompt_processor_method_name}));
  }

  prompt_processor_ = std::make_unique<PromptProcessor<T>>(
      decoder_runner_.get(),
      kv_manager_.get(),
      prompt_processor_method_name,
      typename PromptProcessor<T>::Metadata{
          context_len_,
          num_heads_,
          num_layers_,
          prompt_processor_ar_len_,
          vocab_size_,
          use_int64_token,
          sliding_window,
          cache_mode_,
          prefill_outputs_logits_,
          separate_embed_,
          embedding_dim,
          embedding_row_bytes,
          embedding_scalar_type,
          separate_embed_ ? &separate_embedding_ : nullptr});

  buffer_manager_ = std::make_unique<ClientMem>();
  if (shared_buffer_) {
    buffer_manager_ = std::make_unique<RpcMem>(
        kv_manager_->total_cache_size_in_bytes(),
        prompt_processor_->total_prompt_processor_io_size_in_bytes(),
        0);
  }

  kv_manager_->init_cache(buffer_manager_.get(), prompt_processor_ar_len_);
  if (static_metadata_.enabled) {
    prompt_processor_->init_io_from_metadata(buffer_manager_.get());
  } else {
    prompt_processor_->init_io(
        buffer_manager_.get(), module_->method_meta(prompt_processor_method_name));
  }
  return Error::Ok;
}

template <typename T>
void PDPrefillRunner<T>::reset() {
  cur_pos_ = 0;
  context_len_ = 0;
  prompt_processor_ar_len_ = 0;
  token_generator_ar_len_ = 0;
  max_cache_len_ = 0;
  vocab_size_ = 0;
  num_layers_ = 0;
  num_heads_ = 0;
  head_dim_ = 0;
  if (prompt_processor_ != nullptr) {
    prompt_processor_->clear_all_logits();
  }
  buffer_manager_.reset();
  kv_manager_.reset();
  prompt_processor_.reset();
  decoder_runner_.reset();
  attention_sink_rope_runner_.reset();
}

template <typename T>
Result<DecoderModelVersion> PDPrefillRunner<T>::get_decoder_model_version() {
  return decoder_model_version_;
}

template <typename T>
const typename PDPrefillRunner<T>::RuntimeStats&
PDPrefillRunner<T>::last_runtime_stats() const {
  return last_runtime_stats_;
}

template <typename T>
std::vector<DecoderRunner::PrefillShardRuntimeStats>
PDPrefillRunner<T>::prefill_shard_runtime_stats() const {
  return decoder_runner_ != nullptr
      ? decoder_runner_->prefill_shard_runtime_stats()
      : std::vector<DecoderRunner::PrefillShardRuntimeStats>{};
}

template <typename T>
Error PDPrefillRunner<T>::evaluate_wikitext_ppl(
    const std::string& wikitext_path,
    int32_t start_token,
    int32_t max_eval_tokens,
    float logits_scale,
    int32_t logits_zero_point,
    double* ppl_out,
    int64_t* scored_tokens_out) {
  ET_CHECK_MSG(ppl_out != nullptr, "ppl_out cannot be null");
  ET_CHECK_MSG(prefill_outputs_logits_, "WikiPPL requires prefill logits");
  ET_CHECK_MSG(start_token >= 0, "start_token cannot be negative");
  ET_CHECK_MSG(max_eval_tokens > 0, "max_eval_tokens must be positive");
  ET_CHECK_OK_OR_RETURN_ERROR(load());

  std::ifstream input(wikitext_path);
  ET_CHECK_MSG(input.is_open(), "Unable to read WikiText: %s", wikitext_path.c_str());
  std::stringstream buffer;
  buffer << input.rdbuf();
  auto encode_res = tokenizer_->encode(buffer.str(), 0, 0);
  ET_CHECK_TK_OK_OR_RETURN_ERROR(
      encode_res.error(), "failed to encode WikiText %s", wikitext_path.c_str());
  std::vector<uint64_t> all_tokens = encode_res.get();
  const int64_t available_rows = static_cast<int64_t>(all_tokens.size()) - 1;
  ET_CHECK_MSG(start_token < available_rows, "WikiText start token is out of range");
  const int64_t score_count =
      std::min<int64_t>(max_eval_tokens, available_rows - start_token);
  const int64_t input_count = static_cast<int64_t>(start_token) + score_count;
  ET_CHECK_MSG(
      input_count <= context_len_ - prompt_processor_ar_len_,
      "WikiPPL input exceeds QNN prefill budget");

  std::vector<uint64_t> input_tokens(
      all_tokens.begin(), all_tokens.begin() + input_count);
  const size_t iterations =
      (input_tokens.size() + prompt_processor_ar_len_ - 1) /
      prompt_processor_ar_len_;
  prompt_processor_->clear_all_logits();
  prompt_processor_->reserve_all_logits(
      iterations * static_cast<size_t>(prompt_processor_ar_len_) * vocab_size_);
  auto prefill_res = prompt_processor_->prefill(
      std::move(input_tokens), 0, true, attention_sink_rope_runner_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());

  const auto& logits = prompt_processor_->get_all_logits();
  ET_CHECK_MSG(
      logits.size() >=
          static_cast<size_t>(input_count) * static_cast<size_t>(vocab_size_),
      "QNN WikiPPL logits buffer is smaller than expected");
  double total_nll = 0.0;
  for (int64_t row = start_token; row < start_token + score_count; ++row) {
    const uint16_t* row_logits =
        logits.data() + static_cast<size_t>(row) * vocab_size_;
    total_nll += nll_from_quantized_logits(
        row_logits,
        vocab_size_,
        all_tokens[row + 1],
        logits_scale,
        logits_zero_point);
  }
  *ppl_out = std::exp(total_nll / static_cast<double>(score_count));
  if (scored_tokens_out != nullptr) {
    *scored_tokens_out = score_count;
  }
  ET_LOG(
      Info,
      "QNN WikiPPL: ppl=%.12f nll=%.12f scored_tokens=%ld start_token=%d",
      *ppl_out,
      total_nll,
      score_count,
      start_token);
  if (start_token > 0) {
    const uint16_t* prompt_last_logits =
        logits.data() + static_cast<size_t>(start_token - 1) * vocab_size_;
    ET_LOG(
        Info,
        "QNN WikiPPL boundary check: u16_prompt_last_top1=%d "
        "expected_first_target=%llu",
        argmax_u16_logits(prompt_last_logits, vocab_size_),
        static_cast<unsigned long long>(all_tokens[start_token]));
  }
  return Error::Ok;
}

template <typename T>
double PDPrefillRunner<T>::prefill_qnn_backend_prewarm_ms() const {
  return decoder_runner_ != nullptr
      ? decoder_runner_->prefill_qnn_backend_prewarm_ms()
      : 0.0;
}

template <typename T>
bool PDPrefillRunner<T>::prefill_qnn_backend_prewarmed() const {
  return decoder_runner_ != nullptr &&
      decoder_runner_->prefill_qnn_backend_prewarmed();
}

template <typename T>
void PDPrefillRunner<T>::set_prefill_etdump_config(
    DecoderRunner::PrefillEtDumpConfig config) {
  ET_CHECK_MSG(
      decoder_runner_ == nullptr,
      "Configure prefill ETDump before PDPrefillRunner::load");
  prefill_etdump_config_ = std::move(config);
}

template <typename T>
Error PDPrefillRunner<T>::export_prefill_handoff(
    const std::string& prompt,
    bool tokenized_prompt,
    int32_t seq_len,
    const std::string& export_dir,
    const std::string& kv_quant_attrs_path) {
  ET_CHECK_MSG(!prompt.empty(), "prompt cannot be null");
  ET_CHECK_MSG(cur_pos_ == 0, "PD prefill export only supports a fresh context");
  if (!is_loaded()) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());
  }

  if (attention_sink_rope_runner_ != nullptr) {
    ET_CHECK_MSG(false, "PD prefill export does not support attention sink in v1");
  }

  if (seq_len <= 0 || seq_len > context_len_) {
    seq_len = context_len_;
  }

  last_runtime_stats_ = {};
  int32_t n_bos = (cur_pos_ == 0) ? 1 : 0;
  const auto tokenize_start = SteadyClock::now();
  std::vector<uint64_t> prompt_tokens;
  if (tokenized_prompt) {
    std::ifstream in_file(prompt, std::ios::binary);
    ET_CHECK_MSG(in_file.is_open(), "Unable to read tokenized prompt: %s", prompt.c_str());
    in_file.seekg(0, std::ios::end);
    size_t file_size = in_file.tellg();
    in_file.seekg(0, std::ios::beg);
    prompt_tokens.resize(file_size / sizeof(uint64_t));
    in_file.read(reinterpret_cast<char*>(prompt_tokens.data()), file_size);
  } else {
    auto encode_res = tokenizer_->encode(prompt, n_bos, 0);
    ET_CHECK_TK_OK_OR_RETURN_ERROR(
        encode_res.error(), "failed to encode prompt %s", prompt.c_str());
    prompt_tokens = encode_res.get();
  }

  const int32_t num_prompt_tokens = static_cast<int32_t>(prompt_tokens.size());
  last_runtime_stats_.prompt_tokens = num_prompt_tokens;
  last_runtime_stats_.tokenize_ms = elapsed_ms(tokenize_start);
  ET_CHECK_MSG(num_prompt_tokens >= 1, "Expected at least 1 prompt token");
  const bool bridge_prompt_tail = !prefill_outputs_logits_;
  std::vector<uint64_t> cached_prompt_tokens = prompt_tokens;
  uint64_t first_token = 0;
  if (bridge_prompt_tail) {
    ET_CHECK_MSG(
        num_prompt_tokens >= 2,
        "No-output PD prefill requires at least two prompt tokens");
    first_token = cached_prompt_tokens.back();
    cached_prompt_tokens.pop_back();
  }
  const int32_t num_cached_tokens =
      static_cast<int32_t>(cached_prompt_tokens.size());
  ET_CHECK_MSG(
      cur_pos_ + num_cached_tokens < seq_len,
      "sequence length exceeded - please increase seq_len");

  const auto prefill_start = SteadyClock::now();
  auto prefill_res = prompt_processor_->prefill(
      cached_prompt_tokens,
      cur_pos_,
      false,
      attention_sink_rope_runner_.get());
  ET_CHECK_OK_OR_RETURN_ERROR(prefill_res.error());
  if (!bridge_prompt_tail) {
    first_token = prefill_res.get();
  }
  last_runtime_stats_.prefill_ms = elapsed_ms(prefill_start);
  cur_pos_ += num_cached_tokens;

  const auto handoff_start = SteadyClock::now();
  const auto quant_attrs_start = SteadyClock::now();
  std::vector<KvQuantAttr> quant_attrs;
  if constexpr (std::is_same_v<T, uint8_t>) {
    const size_t expected_attr_count = static_cast<size_t>(num_layers_) * 2;
    ET_CHECK_MSG(
        !kv_quant_attrs_path.empty(),
        "8-bit KV PD export requires --kv_quant_attrs_path pointing to prefill_kv_quant_attrs.json");
    quant_attrs = read_kv_quant_attrs_from_json(kv_quant_attrs_path, expected_attr_count);
    for (size_t i = 0; i < expected_attr_count; ++i) {
      ET_CHECK_MSG(quant_attrs.at(i).valid, "Missing kv quant attr %zu", i);
    }
  }
  const double quant_attrs_ms = elapsed_ms(quant_attrs_start);

  const auto kv_layout_start = SteadyClock::now();
  const std::vector<uint16_t> canonical_kv = build_canonical_kv(
      kv_manager_.get(),
      decoder_model_version_,
      model_path_,
      num_layers_,
      num_heads_,
      head_dim_,
      num_cached_tokens,
      prefill_cache_stride_,
      quant_attrs);
  std::vector<uint8_t> qnn_u8_kv;
  if constexpr (std::is_same_v<T, uint8_t>) {
    qnn_u8_kv = build_qnn_u8_kv_handoff(
        kv_manager_.get(),
        num_layers_,
        num_heads_,
        head_dim_,
        num_cached_tokens,
        prefill_cache_stride_);
  }
  const double kv_layout_ms = elapsed_ms(kv_layout_start);
  const size_t canonical_kv_bytes = canonical_kv.size() * sizeof(uint16_t);
  const size_t qnn_u8_kv_bytes = qnn_u8_kv.size();
  const bool export_undo_r3 = should_undo_r3_on_export(
      decoder_model_version_, model_path_, num_layers_, num_heads_, head_dim_);

  fs::create_directories(export_dir);
  const fs::path export_path(export_dir);
  const auto metadata_files_start = SteadyClock::now();
  write_binary(
      export_path / "prompt_tokens.bin",
      cached_prompt_tokens.data(),
      cached_prompt_tokens.size() * sizeof(uint64_t));
  write_binary(
      export_path / "first_token.bin",
      &first_token,
      sizeof(first_token));
  const double metadata_files_ms = elapsed_ms(metadata_files_start);

  const auto kv_write_start = SteadyClock::now();
  write_binary(
      export_path / "kv.bin",
      canonical_kv.data(),
      canonical_kv_bytes);
  if constexpr (std::is_same_v<T, uint8_t>) {
    write_binary(
        export_path / "kv_qnn_u8.bin",
        qnn_u8_kv.data(),
        qnn_u8_kv_bytes);
  }
  const double kv_write_ms = elapsed_ms(kv_write_start);

  json manifest;
  manifest["format_version"] = "pd-handoff-v1";
  manifest["decoder_model_version"] = decoder_model_to_string(decoder_model_version_);
  manifest["context_length"] = context_len_;
  manifest["prompt_length"] = num_cached_tokens;
  manifest["original_prompt_length"] = num_prompt_tokens;
  manifest["first_token_is_prompt_tail"] = bridge_prompt_tail;
  manifest["num_layers"] = num_layers_;
  manifest["num_kv_heads"] = num_heads_;
  manifest["head_dim"] = head_dim_;
  manifest["cache_mode"] = cache_mode_to_string(cache_mode_);
  manifest["source_kv_bit_width"] = std::is_same_v<T, uint8_t> ? 8 : 16;
  manifest["canonical_kv_dtype"] = "fp16";
  manifest["canonical_kv_layout"] = {
      {"order", "K_then_V"},
      {"shape", "[layer,kv_head,seq,head_dim]"},
      {"endianness", "little"},
  };
  manifest["canonical_k_export_transform"] =
      export_undo_r3 ? "undo_spinquant_r3" : "identity";
  manifest["prompt_tokens_file"] = "prompt_tokens.bin";
  manifest["prompt_tokens_dtype"] = "uint64";
  manifest["first_token_file"] = "first_token.bin";
  manifest["first_token_id"] = first_token;
  manifest["kv_file"] = "kv.bin";
  manifest["kv_file_size_bytes"] = static_cast<uint64_t>(canonical_kv_bytes);
  if constexpr (std::is_same_v<T, uint8_t>) {
    manifest["qnn_u8_kv_file"] = "kv_qnn_u8.bin";
    manifest["qnn_u8_kv_file_size_bytes"] = static_cast<uint64_t>(qnn_u8_kv_bytes);
    manifest["qnn_u8_kv_dtype"] = "uint8";
    manifest["qnn_u8_kv_layout"] = {
        {"order", "K_then_V"},
        {"shape", "[layer,kv_head,seq,head_dim]"},
        {"k_cache_transform", "spinquant_r3"},
        {"endianness", "little"},
    };
  }
  manifest["first_token_owner"] =
      bridge_prompt_tail ? "prompt_tail_bridge" : "executorch";
  const auto fingerprint_start = SteadyClock::now();
  manifest["pte_fingerprint"] = make_file_fingerprint(model_path_);
  manifest["tokenizer_fingerprint"] = make_file_fingerprint(tokenizer_path_);
  const double fingerprint_ms = elapsed_ms(fingerprint_start);
  manifest["rope"] = {
      {"freq_base", nullptr},
      {"freq_scale", nullptr},
  };
  if constexpr (std::is_same_v<T, uint8_t>) {
    json quant_json = json::array();
    for (size_t i = 0; i < quant_attrs.size(); ++i) {
      quant_json.push_back({
          {"index", i},
          {"scale", quant_attrs[i].scale},
          {"zero_point", quant_attrs[i].zero_point},
          {"dtype", quant_attrs[i].dtype},
      });
    }
    manifest["source_kv_quant_attrs"] = quant_json;
  }

  const auto manifest_write_start = SteadyClock::now();
  std::ofstream manifest_out(export_path / "manifest.json");
  ET_CHECK_MSG(
      manifest_out.is_open(),
      "Unable to write manifest: %s",
      (export_path / "manifest.json").c_str());
  manifest_out << manifest.dump(2) << "\n";
  manifest_out.flush();
  ET_CHECK_MSG(manifest_out.good(), "Failed to write PD manifest");
  const double manifest_write_ms = elapsed_ms(manifest_write_start);
  const double handoff_total_ms = elapsed_ms(handoff_start);
  last_runtime_stats_.handoff_total_ms = handoff_total_ms;
  last_runtime_stats_.kv_layout_ms = kv_layout_ms;
  last_runtime_stats_.kv_write_ms = kv_write_ms;
  last_runtime_stats_.fingerprint_ms = fingerprint_ms;
  ET_LOG(
      Info,
      "PD handoff timing: quant_attrs_ms=%.3f kv_layout_ms=%.3f "
      "metadata_files_ms=%.3f kv_write_ms=%.3f fingerprint_ms=%.3f "
      "manifest_write_ms=%.3f total_ms=%.3f",
      quant_attrs_ms,
      kv_layout_ms,
      metadata_files_ms,
      kv_write_ms,
      fingerprint_ms,
      manifest_write_ms,
      handoff_total_ms);
  ET_LOG(
      Info,
      "PD handoff exported: dir=%s prompt_len=%d cached_prompt_len=%d kv_bytes=%zu first_token=%llu bridge_prompt_tail=%d",
      export_dir.c_str(),
      num_prompt_tokens,
      num_cached_tokens,
      canonical_kv_bytes,
      static_cast<unsigned long long>(first_token),
      static_cast<int>(bridge_prompt_tail));
  return Error::Ok;
}

template class PDPrefillRunner<uint16_t>;
template class PDPrefillRunner<uint8_t>;

} // namespace example
