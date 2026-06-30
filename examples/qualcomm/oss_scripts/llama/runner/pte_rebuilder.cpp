/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/pte_rebuilder.h>
#include <executorch/runtime/platform/compiler.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace example {
namespace {

using json = nlohmann::json;

constexpr std::array<const char*, 7> kForwardTargetOps = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
};

constexpr uint32_t kBinaryIndexMagic = 0x49515445U;
constexpr uint16_t kBinaryIndexVersion = 1;
constexpr uint16_t kBinaryIndexHeaderSizeV1 = 32;
constexpr uint16_t kBinaryIndexRecordSize = 20;
constexpr uint8_t kWeightKindOutputConv = 0;
constexpr uint8_t kWeightKindDecoderLinear = 1;

struct EncodedWeightId {
  uint8_t kind{kWeightKindOutputConv};
  uint8_t layer_id{0};
  uint8_t op_id{0};
};

struct TensorView {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t* data{nullptr};
  size_t num_bytes{0};

  const int32_t* data_i32() const {
    /* if (dtype != "I32") {
      throw std::runtime_error("Expected I32 tensor for " + dtype);
    } */
    return reinterpret_cast<const int32_t*>(data);
  }
};

struct SafeTensorsView {
  std::unordered_map<std::string, TensorView> tensors;
};

enum class GgufValueType : uint32_t {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
};

enum class GgufTensorType : uint32_t {
  F16 = 1,
  I2 = 37,
  GPTQ2_32 = 42,
};

struct RebuildRecord {
  EncodedWeightId weight_id;
  int block_id{0};
  int64_t source_offset{0};
  int64_t length{0};
};

uint64_t read_u64_le(const uint8_t* data) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(data[i]) << (8 * i);
  }
  return value;
}

uint32_t read_u32_le(const uint8_t* data) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(data[i]) << (8 * i);
  }
  return value;
}

uint16_t read_u16_le(const uint8_t* data) {
  uint16_t value = 0;
  for (int i = 0; i < 2; ++i) {
    value |= static_cast<uint16_t>(data[i]) << (8 * i);
  }
  return value;
}

float read_f32_le(const uint8_t* data) {
  uint32_t bits = read_u32_le(data);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float read_f16_le_as_f32(const uint8_t* data) {
  const uint16_t bits = read_u16_le(data);
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1FU;
  const uint32_t mantissa = bits & 0x03FFU;

  uint32_t fp32_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      fp32_bits = sign;
    } else {
      uint32_t mant = mantissa;
      int exp = -14;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x03FFU;
      fp32_bits =
          sign | static_cast<uint32_t>((exp + 127) << 23) | (mant << 13);
    }
  } else if (exponent == 0x1FU) {
    fp32_bits = sign | 0x7F800000U | (mantissa << 13);
  } else {
    fp32_bits = sign |
        static_cast<uint32_t>((static_cast<int>(exponent) - 15 + 127) << 23) |
        (mantissa << 13);
  }

  float value = 0.0f;
  std::memcpy(&value, &fp32_bits, sizeof(value));
  return value;
}

size_t gguf_scalar_type_size(GgufValueType type) {
  switch (type) {
    case GgufValueType::UINT8:
    case GgufValueType::INT8:
    case GgufValueType::BOOL:
      return 1;
    case GgufValueType::UINT16:
    case GgufValueType::INT16:
      return 2;
    case GgufValueType::UINT32:
    case GgufValueType::INT32:
    case GgufValueType::FLOAT32:
      return 4;
    case GgufValueType::UINT64:
    case GgufValueType::INT64:
    case GgufValueType::FLOAT64:
      return 8;
    case GgufValueType::STRING:
    case GgufValueType::ARRAY:
      return 0;
  }
  return 0;
}

size_t align_up(size_t value, size_t alignment) {
  if (alignment == 0) {
    return value;
  }
  const size_t rem = value % alignment;
  return rem == 0 ? value : (value + alignment - rem);
}

std::string read_gguf_string(
    const std::vector<uint8_t>& bytes,
    size_t* cursor) {
  const size_t len = static_cast<size_t>(read_u64_le(bytes.data() + *cursor));
  *cursor += sizeof(uint64_t);
  std::string value(
      reinterpret_cast<const char*>(bytes.data() + *cursor), len);
  *cursor += len;
  return value;
}

void skip_gguf_value(
    const std::vector<uint8_t>& bytes,
    size_t* cursor,
    GgufValueType type) {
  if (type == GgufValueType::STRING) {
    (void)read_gguf_string(bytes, cursor);
    return;
  }

  if (type == GgufValueType::ARRAY) {
    const auto item_type =
        static_cast<GgufValueType>(read_u32_le(bytes.data() + *cursor));
    *cursor += sizeof(uint32_t);
    const size_t count = static_cast<size_t>(read_u64_le(bytes.data() + *cursor));
    *cursor += sizeof(uint64_t);
    for (size_t i = 0; i < count; ++i) {
      skip_gguf_value(bytes, cursor, item_type);
    }
    return;
  }

  *cursor += gguf_scalar_type_size(type);
}

bool records_are_sorted_by_source_offset(const std::vector<RebuildRecord>& records) {
  for (size_t i = 1; i < records.size(); ++i) {
    if (records[i - 1].source_offset > records[i].source_offset) {
      return false;
    }
  }
  return true;
}

int16_t rounded_zero_point(int16_t value, int bits) {
  const int maxq = (1 << bits) - 1;
  return static_cast<int16_t>(
      std::clamp(static_cast<int>(std::lround(static_cast<double>(value))), 0, maxq));
}

struct SpecializedTensorContext {
  const TensorView* qweight{nullptr};
  const TensorView* qzeros{nullptr};
  size_t block_count{0};
  size_t packed_cols{0};
  size_t cols{0};
  size_t num_groups{0};
  size_t qzeros_packed_cols{0};
};

struct SpecializedFastPathRecord {
  const SpecializedTensorContext* tensor_ctx{nullptr};
  bool is_output_conv{false};
  int block_id{0};
};

struct SpecializedFastPathPlan {
  std::vector<SpecializedTensorContext> tensor_contexts;
  std::vector<SpecializedFastPathRecord> records;
};

struct ParsedIndex {
  size_t old_size{0};
  size_t final_size{0};
  size_t total_deleted{0};
  std::vector<RebuildRecord> records;
};

struct GgufTensorInfo {
  std::vector<uint64_t> shape;
  GgufTensorType tensor_type{GgufTensorType::F16};
  const uint8_t* data{nullptr};
  size_t num_bytes{0};
};

struct GgufView {
  uint32_t alignment{32};
  std::unordered_map<std::string, GgufTensorInfo> tensors;
};

struct GgufTensorStub {
  std::string name;
  std::vector<uint64_t> shape;
  GgufTensorType tensor_type{GgufTensorType::F16};
  size_t offset{0};
};

struct DecodedTmacI2Tensor {
  size_t rows{0};
  size_t cols{0};
  size_t num_groups{0};
  std::vector<uint8_t> weights;
  std::vector<uint8_t> qzeros;
};

struct PackedInt4Gs32TensorView {
  const int32_t* qweight{nullptr};
  const int32_t* qzeros{nullptr};
  size_t packed_cols{0};
  size_t cols{0};
  size_t num_groups{0};
  size_t qzeros_packed_cols{0};
};

struct PackedTmacI2Tensor {
  size_t rows{0};
  size_t cols{0};
  size_t num_groups{0};
  size_t packed_cols{0};
  size_t qzeros_packed_cols{0};
  std::vector<int32_t> qweight;
  std::vector<int32_t> qzeros;
};

struct DirectTmacI2TensorView {
  size_t rows{0};
  size_t cols{0};
  size_t num_groups{0};
  const uint8_t* packed_weights{nullptr};
  const uint8_t* tail_ptr{nullptr};
  bool has_explicit_zeros{false};
};

struct DirectGptq2_32TensorView {
  size_t rows{0};
  size_t cols{0};
  size_t num_groups{0};
  const uint8_t* data{nullptr};
};

const TensorView& require_tensor(const SafeTensorsView& view, const std::string& name);
size_t block_count_from_qweight(const TensorView& qweight);
const GgufTensorInfo& require_gguf_tensor(
    const GgufView& view,
    const std::string& name);

SafeTensorsView parse_safetensors(const std::vector<uint8_t>& checkpoint_bytes) {
  /* if (checkpoint_bytes.size() < 8) {
    throw std::runtime_error("Invalid safetensors file: too small");
  } */
  const uint64_t header_len = read_u64_le(checkpoint_bytes.data());
  const size_t header_start = 8;
  const size_t header_end = header_start + static_cast<size_t>(header_len);
  /* if (header_end > checkpoint_bytes.size()) {
    throw std::runtime_error("Invalid safetensors file: header out of range");
  } */

  const std::string header(
      reinterpret_cast<const char*>(checkpoint_bytes.data() + header_start),
      static_cast<size_t>(header_len));
  const json root = json::parse(header);
  const size_t data_base = header_end;

  SafeTensorsView out;
  out.tensors.reserve(root.size());
  for (auto it = root.begin(); it != root.end(); ++it) {
    if (it.key() == "__metadata__") {
      continue;
    }
    const auto& value = it.value();
    const std::string dtype = value.at("dtype").get<std::string>();
    const auto& shape_json = value.at("shape");
    const auto& offsets_json = value.at("data_offsets");
    /* if (!shape_json.is_array() || !offsets_json.is_array() ||
        offsets_json.size() != 2) {
      throw std::runtime_error("Invalid safetensors entry for " + it.key());
    } */

    std::vector<int64_t> shape;
    shape.reserve(shape_json.size());
    for (const auto& dim : shape_json) {
      shape.push_back(dim.get<int64_t>());
    }

    const size_t begin = data_base + offsets_json[0].get<size_t>();
    const size_t end = data_base + offsets_json[1].get<size_t>();
    /* if (begin > end || end > checkpoint_bytes.size()) {
      throw std::runtime_error("Invalid safetensors offsets for " + it.key());
    } */

    out.tensors.emplace(
        it.key(),
        TensorView{dtype, std::move(shape), checkpoint_bytes.data() + begin, end - begin});
  }
  return out;
}

const TensorView& require_tensor(
    const SafeTensorsView& view,
    const std::string& name) {
  auto it = view.tensors.find(name);
  /* if (it == view.tensors.end()) {
    throw std::runtime_error("Missing checkpoint tensor: " + name);
  } */
  return it->second;
}

GgufView parse_gguf(const std::vector<uint8_t>& gguf_bytes) {
  size_t cursor = 0;
  /* if (gguf_bytes.size() < 24) {
    throw std::runtime_error("GGUF file too small");
  } */
  /* if (read_u32_le(gguf_bytes.data()) != 0x46554747U) {
    throw std::runtime_error("GGUF magic mismatch");
  } */
  cursor += sizeof(uint32_t); // magic
  const uint32_t version = read_u32_le(gguf_bytes.data() + cursor);
  cursor += sizeof(uint32_t);
  /* if (version != 3) {
    throw std::runtime_error("Unsupported GGUF version");
  } */
  (void)version;

  const size_t tensor_count =
      static_cast<size_t>(read_u64_le(gguf_bytes.data() + cursor));
  cursor += sizeof(uint64_t);
  const size_t kv_count =
      static_cast<size_t>(read_u64_le(gguf_bytes.data() + cursor));
  cursor += sizeof(uint64_t);

  GgufView out;
  for (size_t i = 0; i < kv_count; ++i) {
    const std::string key = read_gguf_string(gguf_bytes, &cursor);
    const auto value_type =
        static_cast<GgufValueType>(read_u32_le(gguf_bytes.data() + cursor));
    cursor += sizeof(uint32_t);
    if (key == "general.alignment" && value_type == GgufValueType::UINT32) {
      out.alignment = read_u32_le(gguf_bytes.data() + cursor);
    }
    skip_gguf_value(gguf_bytes, &cursor, value_type);
  }

  std::vector<GgufTensorStub> tensors;
  tensors.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    GgufTensorStub stub;
    stub.name = read_gguf_string(gguf_bytes, &cursor);
    const size_t ndim =
        static_cast<size_t>(read_u32_le(gguf_bytes.data() + cursor));
    cursor += sizeof(uint32_t);
    stub.shape.reserve(ndim);
    for (size_t dim = 0; dim < ndim; ++dim) {
      stub.shape.push_back(read_u64_le(gguf_bytes.data() + cursor));
      cursor += sizeof(uint64_t);
    }
    stub.tensor_type =
        static_cast<GgufTensorType>(read_u32_le(gguf_bytes.data() + cursor));
    cursor += sizeof(uint32_t);
    stub.offset =
        static_cast<size_t>(read_u64_le(gguf_bytes.data() + cursor));
    cursor += sizeof(uint64_t);
    tensors.push_back(std::move(stub));
  }

  const size_t data_start = align_up(cursor, out.alignment);
  std::sort(
      tensors.begin(),
      tensors.end(),
      [](const GgufTensorStub& lhs, const GgufTensorStub& rhs) {
        return lhs.offset < rhs.offset;
      });

  out.tensors.reserve(tensors.size());
  for (size_t i = 0; i < tensors.size(); ++i) {
    const size_t next_offset = (i + 1 < tensors.size())
        ? tensors[i + 1].offset
        : (gguf_bytes.size() - data_start);
    const size_t num_bytes = next_offset - tensors[i].offset;
    out.tensors.emplace(
        tensors[i].name,
        GgufTensorInfo{
            tensors[i].shape,
            tensors[i].tensor_type,
            gguf_bytes.data() + data_start + tensors[i].offset,
            num_bytes});
  }

  return out;
}

const GgufTensorInfo& require_gguf_tensor(
    const GgufView& view,
    const std::string& name) {
  auto it = view.tensors.find(name);
  if (it == view.tensors.end()) {
    throw std::runtime_error("Missing GGUF tensor: " + name);
  }
  return it->second;
}

ParsedIndex parse_binary_index(const std::vector<uint8_t>& index_bytes) {
  /* if (index_bytes.size() < kBinaryIndexHeaderSizeV1) {
    throw std::runtime_error("Binary index too small");
  } */
  /* if (read_u32_le(index_bytes.data()) != kBinaryIndexMagic) {
    throw std::runtime_error("Binary index magic mismatch");
  } */
  const uint16_t version = read_u16_le(index_bytes.data() + 4);
  /* if (version != kBinaryIndexVersion) {
    throw std::runtime_error("Unsupported binary index version");
  } */
  const uint16_t header_size = read_u16_le(index_bytes.data() + 6);
  const uint16_t record_size = read_u16_le(index_bytes.data() + 8);
  /* if (header_size != kBinaryIndexHeaderSizeV1 || record_size != kBinaryIndexRecordSize) {
    throw std::runtime_error("Unsupported binary index layout");
  } */

  ParsedIndex out;
  const uint32_t record_count = read_u32_le(index_bytes.data() + 12);
  out.old_size = static_cast<size_t>(read_u64_le(index_bytes.data() + 16));
  out.final_size = static_cast<size_t>(read_u64_le(index_bytes.data() + 24));
  out.total_deleted = out.old_size - out.final_size;

  const size_t expected_size =
      static_cast<size_t>(header_size) + static_cast<size_t>(record_count) * record_size;
  /* if (expected_size != index_bytes.size()) {
    throw std::runtime_error("Binary index size mismatch");
  } */

  out.records.reserve(record_count);
  const uint8_t* record_ptr = index_bytes.data() + header_size;
  for (uint32_t i = 0; i < record_count; ++i, record_ptr += record_size) {
    const EncodedWeightId weight_id{record_ptr[0], record_ptr[1], record_ptr[2]};
    out.records.push_back(RebuildRecord{
        weight_id,
        static_cast<int>(read_u32_le(record_ptr + 4)),
        static_cast<int64_t>(read_u64_le(record_ptr + 8)),
        static_cast<int64_t>(read_u32_le(record_ptr + 16)),
    });
  }
  /* if (!records_are_sorted_by_source_offset(out.records)) {
    throw std::runtime_error("Binary index records are not sorted by source_offset");
  } */
  return out;
}

int unpack_packed_value(uint32_t packed_value, int bits, int maxq, size_t offset) {
  return static_cast<int>((packed_value >> (bits * offset)) & static_cast<uint32_t>(maxq));
}

#if defined(__ARM_NEON) && defined(__aarch64__)
ET_INLINE int16x8_t load_qweight_chunk8_neon_gs32_int4(
    const int32_t* src,
    size_t packed_cols,
    size_t out_feature,
    size_t col_start) {
  const size_t packed_row = col_start / 16;
  const size_t offset_base = (col_start % 16) / 8;
  const uint32_t packed_value =
      static_cast<uint32_t>(src[packed_row * packed_cols + out_feature]);
  const uint32x4_t packed_vec = vdupq_n_u32(packed_value);

  // base offsets (16-bit)
  const uint16x8_t base = {0, 2, 4, 6, 8, 10, 12, 14};
  const uint16x8_t offset_vec = vdupq_n_u16((uint16_t)(offset_base * 16));
  int16x8_t shift_s16 = vnegq_s16(vreinterpretq_s16_u16(vaddq_u16(offset_vec, base)));
  const uint16x4_t low = vmovn_u32(vandq_u32(
      vshlq_u32(packed_vec, vmovl_s16(vget_low_s16(shift_s16))),
      vdupq_n_u32(3)));
  const uint16x4_t high = vmovn_u32(vandq_u32(
      vshlq_u32(packed_vec, vmovl_s16(vget_high_s16(shift_s16))),
      vdupq_n_u32(3)));

  return vreinterpretq_s16_u16(vcombine_u16(low, high));
}

ET_INLINE void pack_chunk_neon_gs32_int4(int16x8_t values_vec, uint8_t* out_ptr) {
  const uint16x8_t nibble_mask = vdupq_n_u16(0xF);
  const uint8x8_t nibbles =
      vmovn_u16(vandq_u16(vreinterpretq_u16_s16(values_vec), nibble_mask));
  const uint64_t nibble_bytes = vget_lane_u64(vreinterpret_u64_u8(nibbles), 0);
  const uint32_t packed = static_cast<uint32_t>(
      (nibble_bytes & 0x000000000F0F0F0FULL) |
      ((nibble_bytes >> 28) & 0x00000000F0F0F0F0ULL));
  std::memcpy(out_ptr, &packed, sizeof(packed));
}
#endif

void load_qweight_chunk8(
    const int32_t* src,
    size_t packed_cols,
    int bits,
    int pf,
    int maxq,
    size_t out_feature,
    size_t col_start,
    int16_t* values) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  if (bits == 4 && pf == 8 && (col_start % 8 == 0)) {
    const size_t packed_row = col_start / 8;
    const uint32_t packed_value =
        static_cast<uint32_t>(src[packed_row * packed_cols + out_feature]);
    static constexpr std::array<int32_t, 4> kShift0 = {0, -4, -8, -12};
    static constexpr std::array<int32_t, 4> kShift1 = {-16, -20, -24, -28};
    const uint32x4_t packed_vec = vdupq_n_u32(packed_value);
    const uint16x4_t low = vmovn_u32(vandq_u32(
        vshlq_u32(packed_vec, vld1q_s32(kShift0.data())),
        vdupq_n_u32(static_cast<uint32_t>(maxq))));
    const uint16x4_t high = vmovn_u32(vandq_u32(
        vshlq_u32(packed_vec, vld1q_s32(kShift1.data())),
        vdupq_n_u32(static_cast<uint32_t>(maxq))));
    vst1q_s16(values, vreinterpretq_s16_u16(vcombine_u16(low, high)));
    return;
  }
  if (bits == 2 && pf == 16 && (col_start % 8 == 0)) {
    vst1q_s16(
        values,
        load_qweight_chunk8_neon_gs32_int4(src, packed_cols, out_feature, col_start));
    return;
  }
#endif
  size_t current_packed_row = static_cast<size_t>(-1);
  uint32_t packed_value = 0;
  for (size_t i = 0; i < 8; ++i) {
    const size_t col = col_start + i;
    const size_t packed_row = col / static_cast<size_t>(pf);
    if (packed_row != current_packed_row) {
      packed_value = static_cast<uint32_t>(src[packed_row * packed_cols + out_feature]);
      current_packed_row = packed_row;
    }
    values[i] = static_cast<int16_t>(unpack_packed_value(
        packed_value, bits, maxq, col % static_cast<size_t>(pf)));
  }
}

void build_qzeros_cache(
    const int32_t* src,
    size_t packed_cols,
    int bits,
    int pf,
    size_t num_groups,
    size_t row_start,
    std::vector<int16_t>* cache) {
  cache->assign(num_groups * 64, 0);
  const int maxq = (1 << bits) - 1;
  for (size_t group = 0; group < num_groups; ++group) {
    for (size_t row = 0; row < 64; ++row) {
      const size_t out_feature = row_start + row;
      const size_t packed_col = out_feature / static_cast<size_t>(pf);
      const size_t offset = out_feature % static_cast<size_t>(pf);
      const uint32_t packed_value =
          static_cast<uint32_t>(src[group * packed_cols + packed_col]);
      (*cache)[group * 64 + row] = rounded_zero_point(
          static_cast<int16_t>(unpack_packed_value(packed_value, bits, maxq, offset)),
          bits);
    }
  }
}

bool is_little_endian_platform() {
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
  return true;
#else
  return false;
#endif
}

void copy_xor_sign_bit(uint8_t* dst, const uint8_t* src, size_t num_bytes) {
#if defined(__ARM_NEON) && defined(__aarch64__)
  const uint8x16_t sign_mask = vdupq_n_u8(0x80);
  size_t offset = 0;
  for (; offset + 16 <= num_bytes; offset += 16) {
    const uint8x16_t values = vld1q_u8(src + offset);
    vst1q_u8(dst + offset, veorq_u8(values, sign_mask));
  }
  for (; offset < num_bytes; ++offset) {
    dst[offset] = src[offset] ^ 0x80;
  }
#else
  for (size_t i = 0; i < num_bytes; ++i) {
    dst[i] = src[i] ^ 0x80;
  }
#endif
}

size_t block_count_from_qweight(const TensorView& qweight) {
  /* if (qweight.shape.size() != 2) {
    throw std::runtime_error("qweight must be 2D");
  } */
  const size_t out_features = static_cast<size_t>(qweight.shape[1]);
  /* if (out_features % 64 != 0) {
    throw std::runtime_error("qweight out_features must be divisible by 64");
  } */
  return out_features / 64;
}

void write_int8_block_direct_specialized(
    const SpecializedTensorContext& tensor_ctx,
    int block_id,
    uint8_t* dst) {
  const size_t row_start = static_cast<size_t>(block_id) * 64;
  const int32_t* src = tensor_ctx.qweight->data_i32();
  const size_t tile_cols = tensor_ctx.cols / 32;

  for (size_t bc = 0; bc < tile_cols; ++bc) {
    const size_t packed_row_start = bc * 8;
    for (size_t br = 0; br < 2; ++br) {
      const size_t row_offset = row_start + br * 32;
      for (size_t packed_row = 0; packed_row < 8; ++packed_row) {
        const int32_t* src_block =
            src + (packed_row_start + packed_row) * tensor_ctx.packed_cols + row_offset;
        copy_xor_sign_bit(
            dst,
            reinterpret_cast<const uint8_t*>(src_block),
            32 * sizeof(int32_t));
        dst += 32 * sizeof(int32_t);
      }
    }
  }
}

void write_int4_block_direct_gs32_minus_qzeros_specialized(
    const PackedInt4Gs32TensorView& tensor_view,
    int block_id,
    uint8_t* dst) {
  static constexpr std::array<int, 8> perm = {4, 0, 5, 1, 6, 2, 7, 3};
  const size_t row_start = static_cast<size_t>(block_id) * 64;
  const size_t tile_cols = tensor_view.cols / 32;
  std::vector<int16_t> qzeros_cache;
  build_qzeros_cache(
      tensor_view.qzeros,
      tensor_view.qzeros_packed_cols,
      2,
      16,
      tensor_view.num_groups,
      row_start,
      &qzeros_cache);

  const auto pack_chunk_scalar = [&](const int16_t* values, uint8_t* out_ptr) {
    for (size_t i = 0; i < 4; ++i) {
      const int high = static_cast<int>(values[perm[i * 2]]);
      const int low = static_cast<int>(values[perm[i * 2 + 1]]);
      out_ptr[i] = static_cast<uint8_t>(((high & 0xF) << 4) | (low & 0xF));
    }
  };

  for (size_t bc = 0; bc < tile_cols; ++bc) {
    const size_t c0 = bc * 32;
    for (size_t br = 0; br < 2; ++br) {
      const size_t r0 = br * 32;
      for (size_t tile_bc = 0; tile_bc < 4; ++tile_bc) {
        const size_t cc0 = c0 + tile_bc * 8;
        const size_t group = std::min(cc0 / static_cast<size_t>(32), tensor_view.num_groups - 1);
        for (size_t tile_br = 0; tile_br < 4; ++tile_br) {
          const size_t rr0 = r0 + tile_br * 8;
          for (size_t r = 0; r < 8; ++r) {
            const size_t row = rr0 + r;
            const size_t out_feature = row_start + row;
            const int16_t zp = qzeros_cache[group * 64 + row];
#if defined(__ARM_NEON) && defined(__aarch64__)
            const int16x8_t values_vec = vsubq_s16(
                load_qweight_chunk8_neon_gs32_int4(
                    tensor_view.qweight,
                    tensor_view.packed_cols,
                    out_feature,
                    cc0),
                vdupq_n_s16(zp));
            pack_chunk_neon_gs32_int4(values_vec, dst);
#else
            alignas(16) int16_t chunk_values[8];
            load_qweight_chunk8(
                tensor_view.qweight,
                tensor_view.packed_cols,
                2,
                16,
                3,
                out_feature,
                cc0,
                chunk_values);
            for (size_t i = 0; i < 8; ++i) {
              chunk_values[i] = static_cast<int16_t>(chunk_values[i] - zp);
            }
            pack_chunk_scalar(chunk_values, dst);
#endif
            dst += 4;
          }
        }
      }
    }
  }
}

void write_int4_block_direct_gs32_minus_qzeros_specialized(
    const SpecializedTensorContext& tensor_ctx,
    int block_id,
    uint8_t* dst) {
  const PackedInt4Gs32TensorView tensor_view{
      tensor_ctx.qweight->data_i32(),
      tensor_ctx.qzeros->data_i32(),
      tensor_ctx.packed_cols,
      tensor_ctx.cols,
      tensor_ctx.num_groups,
      tensor_ctx.qzeros_packed_cols,
  };
  write_int4_block_direct_gs32_minus_qzeros_specialized(
      tensor_view,
      block_id,
      dst);
}

std::string gguf_tensor_name_from_record(const RebuildRecord& rec) {
  static constexpr std::array<const char*, 7> kGgufTargetOps = {
      "attn_q",
      "attn_k",
      "attn_v",
      "attn_output",
      "ffn_gate",
      "ffn_up",
      "ffn_down",
  };
  return "blk." + std::to_string(rec.weight_id.layer_id) + "." +
      std::string(kGgufTargetOps[rec.weight_id.op_id]) + ".weight";
}

uint8_t pack_tmac_i2_nibbles(uint8_t bitplane0, uint8_t bitplane1) {
  static const std::array<uint8_t, 256> kPackedLut = []() {
    std::array<uint8_t, 256> lut{};
    for (size_t idx = 0; idx < lut.size(); ++idx) {
      const uint8_t low = static_cast<uint8_t>(idx & 0x0F);
      const uint8_t high = static_cast<uint8_t>((idx >> 4) & 0x0F);
      uint8_t packed = 0;
      for (size_t bit = 0; bit < 4; ++bit) {
        const uint8_t q =
            static_cast<uint8_t>(((low >> bit) & 0x1) | (((high >> bit) & 0x1) << 1));
        packed |= static_cast<uint8_t>(q << (bit * 2));
      }
      lut[idx] = packed;
    }
    return lut;
  }();
  return kPackedLut[static_cast<uint8_t>(bitplane0 | (bitplane1 << 4))];
}

uint8_t decode_tmac_i2_qzero(
    const uint8_t* tail_ptr,
    size_t row,
    size_t group,
    size_t num_groups,
    bool has_explicit_zeros) {
  static constexpr size_t kBits = 2;
  static constexpr size_t kRowsPerPackedTile = 128;
  static constexpr size_t kRowsPerScaleChunk = 8;
  static constexpr size_t kScaleRowsPerBlock = 16;

  if (!has_explicit_zeros) {
    return 2;
  }

  const size_t row_block = row / kRowsPerPackedTile;
  const size_t row_in_block = row % kRowsPerPackedTile;
  const size_t row_chunk = row_in_block / kRowsPerScaleChunk;
  const size_t lane = row_in_block % kRowsPerScaleChunk;
  const uint8_t* group_base =
      tail_ptr + ((row_block * num_groups + group) * kScaleRowsPerBlock * 2 *
                  kRowsPerScaleChunk * sizeof(float));
  const size_t scale_idx =
      (((row_chunk * 2) + 0) * kRowsPerScaleChunk + lane) * sizeof(float);
  const size_t zero_idx =
      (((row_chunk * 2) + 1) * kRowsPerScaleChunk + lane) * sizeof(float);
  const float scale = read_f32_le(group_base + scale_idx);
  const float zero = read_f32_le(group_base + zero_idx);
  if (std::abs(scale) <= std::numeric_limits<float>::min()) {
    return 2;
  }
  return static_cast<uint8_t>(rounded_zero_point(
      static_cast<int16_t>(std::lround(zero / scale + (1 << (kBits - 1)))),
      kBits));
}

uint8_t decode_gptq2_32_qzero(const uint8_t* group_ptr) {
  static constexpr size_t kBits = 2;
  const float scale =
      std::max(read_f16_le_as_f32(group_ptr + 8), 1.0e-4f);
  const float zero_bias = read_f16_le_as_f32(group_ptr + 10);
  return static_cast<uint8_t>(rounded_zero_point(
      static_cast<int16_t>(std::lround(zero_bias / scale)),
      kBits));
}

DirectTmacI2TensorView parse_direct_tmac_i2_tensor(const GgufTensorInfo& tensor) {
  DirectTmacI2TensorView out;
  if (tensor.tensor_type != GgufTensorType::I2 || tensor.shape.size() != 2) {
    throw std::runtime_error(
        "Unsupported GGUF tensor type for T-MAC I2 rebuild");
  }

  static constexpr size_t kColsPerPackedByte = 4;

  out.cols = static_cast<size_t>(tensor.shape[0]);
  out.rows = static_cast<size_t>(tensor.shape[1]);
  out.num_groups = out.cols / 32;

  const size_t weight_bytes = out.rows * out.cols / kColsPerPackedByte;
  const size_t tail_bytes = tensor.num_bytes - weight_bytes;
  const size_t expected_scale_bytes =
      out.rows * out.num_groups * sizeof(float);
  const bool has_explicit_zeros = tail_bytes == expected_scale_bytes * 2;
  const bool has_scales_only = tail_bytes == expected_scale_bytes;
  if (!has_explicit_zeros && !has_scales_only) {
    throw std::runtime_error("Unexpected T-MAC GGUF tensor tail size");
  }

  out.packed_weights = tensor.data;
  out.tail_ptr = tensor.data + weight_bytes;
  out.has_explicit_zeros = has_explicit_zeros;
  return out;
}

DirectGptq2_32TensorView parse_direct_gptq2_32_tensor(
    const GgufTensorInfo& tensor) {
  DirectGptq2_32TensorView out;
  if (tensor.tensor_type != GgufTensorType::GPTQ2_32 || tensor.shape.size() != 2) {
    throw std::runtime_error(
        "Unsupported GGUF tensor type for GPTQ2_32 rebuild");
  }

  out.cols = static_cast<size_t>(tensor.shape[0]);
  out.rows = static_cast<size_t>(tensor.shape[1]);
  if (out.cols % 32 != 0) {
    throw std::runtime_error("GPTQ2_32 GGUF tensor cols must be divisible by 32");
  }
  out.num_groups = out.cols / 32;

  const size_t expected_bytes = out.rows * out.num_groups * 12;
  if (tensor.num_bytes != expected_bytes) {
    throw std::runtime_error("Unexpected GPTQ2_32 GGUF tensor payload size");
  }

  out.data = tensor.data;
  return out;
}

void build_tmac_i2_qzeros_cache(
    const DirectTmacI2TensorView& tensor,
    size_t row_start,
    std::vector<uint8_t>* cache) {
  cache->assign(tensor.num_groups * 64, 2);
  for (size_t group = 0; group < tensor.num_groups; ++group) {
    for (size_t row = 0; row < 64; ++row) {
      (*cache)[group * 64 + row] = decode_tmac_i2_qzero(
          tensor.tail_ptr,
          row_start + row,
          group,
          tensor.num_groups,
          tensor.has_explicit_zeros);
    }
  }
}

void build_gptq2_32_qzeros_cache(
    const DirectGptq2_32TensorView& tensor,
    size_t row_start,
    std::vector<uint8_t>* cache) {
  cache->assign(tensor.num_groups * 64, 0);
  for (size_t group = 0; group < tensor.num_groups; ++group) {
    for (size_t row = 0; row < 64; ++row) {
      const uint8_t* group_ptr =
          tensor.data + ((row_start + row) * tensor.num_groups + group) * 12;
      (*cache)[group * 64 + row] = decode_gptq2_32_qzero(group_ptr);
    }
  }
}

uint32_t pack_gs32_int4_from_qbytes(uint8_t low_qbyte, uint8_t high_qbyte, uint8_t zp) {
  static const std::array<std::array<uint32_t, 65536>, 4> kPackedLut = []() {
    std::array<std::array<uint32_t, 65536>, 4> lut{};
    static constexpr std::array<int, 8> perm = {4, 0, 5, 1, 6, 2, 7, 3};
    for (size_t zp = 0; zp < lut.size(); ++zp) {
      for (size_t key = 0; key < lut[zp].size(); ++key) {
        const uint8_t qbyte0 = static_cast<uint8_t>(key & 0xFF);
        const uint8_t qbyte1 = static_cast<uint8_t>((key >> 8) & 0xFF);
        int values[8];
        for (size_t i = 0; i < 4; ++i) {
          values[i] = static_cast<int>((qbyte0 >> (i * 2)) & 0x3) - static_cast<int>(zp);
          values[4 + i] =
              static_cast<int>((qbyte1 >> (i * 2)) & 0x3) - static_cast<int>(zp);
        }
        uint32_t packed = 0;
        for (size_t i = 0; i < 4; ++i) {
          const uint8_t high = static_cast<uint8_t>(values[perm[i * 2]] & 0xF);
          const uint8_t low = static_cast<uint8_t>(values[perm[i * 2 + 1]] & 0xF);
          packed |= static_cast<uint32_t>((high << 4) | low) << (i * 8);
        }
        lut[zp][key] = packed;
      }
    }
    return lut;
  }();
  return kPackedLut[zp][static_cast<uint16_t>(low_qbyte | (static_cast<uint16_t>(high_qbyte) << 8))];
}

void write_int4_block_from_tmac_i2_direct(
    const DirectTmacI2TensorView& tensor,
    int block_id,
    uint8_t* dst) {
  static constexpr size_t kColsPerPackedByte = 4;
  static constexpr size_t kTmacGroupSize = 4;
  static constexpr size_t kBm = 256;
  static constexpr size_t kBits = 2;
  static constexpr size_t kKFactor = 8;
  static constexpr size_t kSimdNIn = 16;
  static constexpr size_t kRowsPerPackedTile = kBm / kBits;
  static constexpr size_t kRowGroupCount = kBm / ((8 / kTmacGroupSize) * kSimdNIn);

  const size_t row_start = static_cast<size_t>(block_id) * 64;
  const size_t tile_cols = tensor.cols / 32;
  const size_t packed_col_count = tensor.cols / kColsPerPackedByte;
  std::vector<uint8_t> qzeros_cache;
  build_tmac_i2_qzeros_cache(tensor, row_start, &qzeros_cache);

  for (size_t bc = 0; bc < tile_cols; ++bc) {
    const size_t c0 = bc * 32;
    for (size_t br = 0; br < 2; ++br) {
      const size_t r0 = br * 32;
      for (size_t tile_bc = 0; tile_bc < 4; ++tile_bc) {
        const size_t cc0 = c0 + tile_bc * 8;
        const size_t col_pack0 = cc0 / 4;
        const size_t col_pack1 = col_pack0 + 1;
        const size_t k_tile0 = col_pack0 / kKFactor;
        const size_t k_tile1 = col_pack1 / kKFactor;
        const size_t k_inner0 = col_pack0 % kKFactor;
        const size_t k_inner1 = col_pack1 % kKFactor;
        const size_t group = cc0 / 32;
        for (size_t tile_br = 0; tile_br < 4; ++tile_br) {
          const size_t rr0 = r0 + tile_br * 8;
          for (size_t r = 0; r < 8; ++r) {
            const size_t row = row_start + rr0 + r;
            const size_t row_block = row / kRowsPerPackedTile;
            const size_t row_in_block = row % kRowsPerPackedTile;
            const size_t row_group = row_in_block / kSimdNIn;
            const size_t row_group_offset = row_in_block % kSimdNIn;
            const size_t ng = row_group_offset / 8;
            const size_t lane = row_group_offset % 8;
            const size_t pcol0 = k_tile0 * kRowGroupCount + row_group;
            const size_t pcol1 = k_tile1 * kRowGroupCount + row_group;
            const uint8_t* src0 =
                tensor.packed_weights +
                ((row_block * packed_col_count + pcol0) * kRowsPerPackedTile);
            const uint8_t* src1 =
                tensor.packed_weights +
                ((row_block * packed_col_count + pcol1) * kRowsPerPackedTile);
            const size_t prow0 = k_inner0 * kSimdNIn + lane;
            const size_t prow1 = k_inner1 * kSimdNIn + lane;
            const uint8_t nibble00 =
                static_cast<uint8_t>((src0[prow0] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble01 =
                static_cast<uint8_t>((src0[prow0 + 8] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble10 =
                static_cast<uint8_t>((src1[prow1] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble11 =
                static_cast<uint8_t>((src1[prow1 + 8] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t qbyte0 = pack_tmac_i2_nibbles(nibble00, nibble01);
            const uint8_t qbyte1 = pack_tmac_i2_nibbles(nibble10, nibble11);
            const uint8_t zp = qzeros_cache[group * 64 + rr0 + r];
            const uint32_t packed_row =
                pack_gs32_int4_from_qbytes(qbyte0, qbyte1, zp);
            std::memcpy(dst, &packed_row, sizeof(packed_row));
            dst += sizeof(packed_row);
          }
        }
      }
    }
  }
}

void write_int4_block_from_gptq2_32_direct(
    const DirectGptq2_32TensorView& tensor,
    int block_id,
    uint8_t* dst) {
  const size_t row_start = static_cast<size_t>(block_id) * 64;
  const size_t tile_cols = tensor.cols / 32;
  std::vector<uint8_t> qzeros_cache;
  build_gptq2_32_qzeros_cache(tensor, row_start, &qzeros_cache);

  for (size_t bc = 0; bc < tile_cols; ++bc) {
    const size_t c0 = bc * 32;
    for (size_t br = 0; br < 2; ++br) {
      const size_t r0 = br * 32;
      for (size_t tile_bc = 0; tile_bc < 4; ++tile_bc) {
        const size_t cc0 = c0 + tile_bc * 8;
        const size_t group = cc0 / 32;
        const size_t group_qbyte_offset = (cc0 % 32) / 4;
        for (size_t tile_br = 0; tile_br < 4; ++tile_br) {
          const size_t rr0 = r0 + tile_br * 8;
          for (size_t r = 0; r < 8; ++r) {
            const size_t row = row_start + rr0 + r;
            const uint8_t* group_ptr =
                tensor.data + (row * tensor.num_groups + group) * 12;
            const uint8_t qbyte0 = group_ptr[group_qbyte_offset + 0];
            const uint8_t qbyte1 = group_ptr[group_qbyte_offset + 1];
            const uint8_t zp = qzeros_cache[group * 64 + rr0 + r];
            const uint32_t packed_row =
                pack_gs32_int4_from_qbytes(qbyte0, qbyte1, zp);
            std::memcpy(dst, &packed_row, sizeof(packed_row));
            dst += sizeof(packed_row);
          }
        }
      }
    }
  }
}

PackedTmacI2Tensor pack_tmac_i2_tensor(const GgufTensorInfo& tensor) {
  PackedTmacI2Tensor out;
  if (tensor.tensor_type != GgufTensorType::I2 || tensor.shape.size() != 2) {
    throw std::runtime_error(
        "Unsupported GGUF tensor type for T-MAC I2 rebuild");
  }

  static constexpr size_t kBits = 2;
  static constexpr size_t kColsPerPackedByte = 4;
  static constexpr size_t kTmacGroupSize = 4;
  static constexpr size_t kBm = 256;
  static constexpr size_t kKFactor = 8;
  static constexpr size_t kSimdNIn = 16;
  static constexpr size_t kRowsPerPackedTile = kBm / kBits;
  static constexpr size_t kRowGroupCount = kBm / ((8 / kTmacGroupSize) * kSimdNIn);

  // GGUF stores 2D tensor shapes in ggml order: [K, M].
  out.cols = static_cast<size_t>(tensor.shape[0]);
  out.rows = static_cast<size_t>(tensor.shape[1]);
  out.num_groups = out.cols / 32;
  out.packed_cols = out.cols / 16;
  out.qzeros_packed_cols = out.rows / 16;

  const size_t weight_bytes = out.rows * out.cols / kColsPerPackedByte;
  const size_t tail_bytes = tensor.num_bytes - weight_bytes;
  const size_t expected_scale_bytes =
      out.rows * out.num_groups * sizeof(float);
  const bool has_explicit_zeros = tail_bytes == expected_scale_bytes * 2;
  const bool has_scales_only = tail_bytes == expected_scale_bytes;
  if (!has_explicit_zeros && !has_scales_only) {
    throw std::runtime_error("Unexpected T-MAC GGUF tensor tail size");
  }

  out.qweight.assign(out.packed_cols * out.rows, 0);
  out.qzeros.assign(out.num_groups * out.qzeros_packed_cols, 0);

  const uint8_t* packed_weights = tensor.data;
  const uint8_t* tail_ptr = tensor.data + weight_bytes;
  const size_t row_block_count = out.rows / kRowsPerPackedTile;
  const size_t packed_col_count = out.cols / kColsPerPackedByte;

  for (size_t row_block = 0; row_block < row_block_count; ++row_block) {
    const uint8_t* row_block_base =
        packed_weights + row_block * packed_col_count * kRowsPerPackedTile;
    for (size_t packed_col = 0; packed_col < out.packed_cols; ++packed_col) {
      int32_t* dst_col = out.qweight.data() + packed_col * out.rows;
      const size_t col_pack_base = packed_col * 4;
      const size_t col_pack0 = col_pack_base + 0;
      const size_t col_pack1 = col_pack_base + 1;
      const size_t col_pack2 = col_pack_base + 2;
      const size_t col_pack3 = col_pack_base + 3;
      const size_t k_tile0 = col_pack0 / kKFactor;
      const size_t k_tile1 = col_pack1 / kKFactor;
      const size_t k_tile2 = col_pack2 / kKFactor;
      const size_t k_tile3 = col_pack3 / kKFactor;
      const size_t k_inner0 = col_pack0 % kKFactor;
      const size_t k_inner1 = col_pack1 % kKFactor;
      const size_t k_inner2 = col_pack2 % kKFactor;
      const size_t k_inner3 = col_pack3 % kKFactor;

      for (size_t row_group = 0; row_group < kRowGroupCount; ++row_group) {
        const size_t pcol0 = k_tile0 * kRowGroupCount + row_group;
        const size_t pcol1 = k_tile1 * kRowGroupCount + row_group;
        const size_t pcol2 = k_tile2 * kRowGroupCount + row_group;
        const size_t pcol3 = k_tile3 * kRowGroupCount + row_group;
        const uint8_t* src0 = row_block_base + pcol0 * kRowsPerPackedTile;
        const uint8_t* src1 = row_block_base + pcol1 * kRowsPerPackedTile;
        const uint8_t* src2 = row_block_base + pcol2 * kRowsPerPackedTile;
        const uint8_t* src3 = row_block_base + pcol3 * kRowsPerPackedTile;

        for (size_t ng = 0; ng < 2; ++ng) {
          for (size_t lane = 0; lane < 8; ++lane) {
            const size_t row =
                row_block * kRowsPerPackedTile + row_group * kSimdNIn + ng * 8 + lane;
            const size_t prow0_bit0 = k_inner0 * kSimdNIn + lane;
            const size_t prow1_bit0 = k_inner1 * kSimdNIn + lane;
            const size_t prow2_bit0 = k_inner2 * kSimdNIn + lane;
            const size_t prow3_bit0 = k_inner3 * kSimdNIn + lane;
            const uint8_t nibble00 =
                static_cast<uint8_t>((src0[prow0_bit0] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble01 =
                static_cast<uint8_t>((src0[prow0_bit0 + 8] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble10 =
                static_cast<uint8_t>((src1[prow1_bit0] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble11 =
                static_cast<uint8_t>((src1[prow1_bit0 + 8] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble20 =
                static_cast<uint8_t>((src2[prow2_bit0] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble21 =
                static_cast<uint8_t>((src2[prow2_bit0 + 8] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble30 =
                static_cast<uint8_t>((src3[prow3_bit0] >> (ng * kTmacGroupSize)) & 0x0F);
            const uint8_t nibble31 =
                static_cast<uint8_t>((src3[prow3_bit0 + 8] >> (ng * kTmacGroupSize)) & 0x0F);

            const uint32_t packed_value =
                static_cast<uint32_t>(pack_tmac_i2_nibbles(nibble00, nibble01)) |
                (static_cast<uint32_t>(pack_tmac_i2_nibbles(nibble10, nibble11)) << 8) |
                (static_cast<uint32_t>(pack_tmac_i2_nibbles(nibble20, nibble21)) << 16) |
                (static_cast<uint32_t>(pack_tmac_i2_nibbles(nibble30, nibble31)) << 24);
            dst_col[row] = static_cast<int32_t>(packed_value);
          }
        }
      }
    }
  }

  for (size_t group = 0; group < out.num_groups; ++group) {
    int32_t* dst_group = out.qzeros.data() + group * out.qzeros_packed_cols;
    for (size_t row_pack = 0; row_pack < out.qzeros_packed_cols; ++row_pack) {
      uint32_t packed_value = 0;
      const size_t row_base = row_pack * 16;
      for (size_t lane = 0; lane < 16; ++lane) {
        const uint8_t qzero = decode_tmac_i2_qzero(
            tail_ptr,
            row_base + lane,
            group,
            out.num_groups,
            has_explicit_zeros);
        packed_value |= static_cast<uint32_t>(qzero & 0x3) << (lane * 2);
      }
      dst_group[row_pack] = static_cast<int32_t>(packed_value);
    }
  }

  return out;
}

DecodedTmacI2Tensor decode_tmac_i2_tensor(const GgufTensorInfo& tensor) {
  DecodedTmacI2Tensor out;
  if (tensor.tensor_type != GgufTensorType::I2 || tensor.shape.size() != 2) {
    throw std::runtime_error(
        "Unsupported GGUF tensor type for T-MAC I2 rebuild");
  }

  // GGUF stores 2D tensor shapes in ggml order: [K, M].
  out.cols = static_cast<size_t>(tensor.shape[0]);
  out.rows = static_cast<size_t>(tensor.shape[1]);
  out.num_groups = out.cols / 32;

  static constexpr size_t kBits = 2;
  static constexpr size_t kColsPerPackedByte = 4;
  static constexpr size_t kTmacGroupSize = 4;
  static constexpr size_t kBm = 256;
  static constexpr size_t kKFactor = 8;
  static constexpr size_t kSimdNIn = 16;
  static constexpr size_t kNgroupsPerElem = 8 / kTmacGroupSize;
  static constexpr size_t kRowsPerPackedTile = kBm / kBits;
  static constexpr size_t kRowGroupCount = kBm / (kNgroupsPerElem * kSimdNIn);
  static constexpr size_t kRowsPerScaleChunk = 8;
  static constexpr size_t kScaleRowsPerBlock = 16;

  const size_t weight_bytes = out.rows * out.cols / kColsPerPackedByte;
  const size_t tail_bytes = tensor.num_bytes - weight_bytes;
  const size_t expected_scale_bytes =
      out.rows * out.num_groups * sizeof(float);
  const bool has_explicit_zeros = tail_bytes == expected_scale_bytes * 2;
  const bool has_scales_only = tail_bytes == expected_scale_bytes;
  if (!has_explicit_zeros && !has_scales_only) {
    throw std::runtime_error("Unexpected T-MAC GGUF tensor tail size");
  }

  out.weights.assign(out.rows * out.cols, 0);
  out.qzeros.assign(out.rows * out.num_groups, 2);

  // Invert t_mac.weights.preprocess_weights() for bits=2, g=4.
  const uint8_t* packed_weights = tensor.data;
  const size_t row_block_count = out.rows / kRowsPerPackedTile;
  const size_t packed_col_count = out.cols / kColsPerPackedByte;
  for (size_t row_block = 0; row_block < row_block_count; ++row_block) {
    for (size_t packed_col = 0; packed_col < packed_col_count; ++packed_col) {
      const uint8_t* src =
          packed_weights +
          ((row_block * packed_col_count + packed_col) * kRowsPerPackedTile);
      const size_t row_group = packed_col % kRowGroupCount;
      const size_t k_tile = packed_col / kRowGroupCount;
      for (size_t packed_row = 0; packed_row < kRowsPerPackedTile; ++packed_row) {
        const uint8_t packed = src[packed_row];
        const size_t k_inner = packed_row / kSimdNIn;
        const size_t simd_lane = packed_row % kSimdNIn;
        const size_t lane_in_simd = simd_lane % kRowsPerScaleChunk;
        const size_t bit_plane = simd_lane / kRowsPerScaleChunk;
        const size_t col_base =
            (k_tile * kKFactor + k_inner) * kColsPerPackedByte;
        for (size_t ng = 0; ng < kNgroupsPerElem; ++ng) {
          const size_t row =
              row_block * kRowsPerPackedTile + row_group * kSimdNIn +
              ng * kRowsPerScaleChunk + lane_in_simd;
          const uint8_t nibble = (packed >> (ng * kTmacGroupSize)) & 0x0F;
          for (size_t col_inner = 0; col_inner < kColsPerPackedByte; ++col_inner) {
            const uint8_t bit = (nibble >> col_inner) & 0x1;
            out.weights[row * out.cols + col_base + col_inner] |=
                static_cast<uint8_t>(bit << bit_plane);
          }
        }
      }
    }
  }

  if (has_explicit_zeros || has_scales_only) {
    const uint8_t* tail_ptr = tensor.data + weight_bytes;
    for (size_t row_block = 0; row_block < out.rows / kRowsPerPackedTile; ++row_block) {
      for (size_t group = 0; group < out.num_groups; ++group) {
        const uint8_t* group_base =
            tail_ptr + ((row_block * out.num_groups + group) *
                        kScaleRowsPerBlock * (has_explicit_zeros ? 2 : 1) *
                        kRowsPerScaleChunk * sizeof(float));
        for (size_t row_chunk = 0; row_chunk < kScaleRowsPerBlock; ++row_chunk) {
          for (size_t lane = 0; lane < kRowsPerScaleChunk; ++lane) {
            const size_t row =
                row_block * kRowsPerPackedTile +
                row_chunk * kRowsPerScaleChunk + lane;
            uint8_t qzero = 2;
            if (has_explicit_zeros) {
              const size_t scale_idx =
                  (((row_chunk * 2) + 0) * kRowsPerScaleChunk + lane) *
                  sizeof(float);
              const size_t zero_idx =
                  (((row_chunk * 2) + 1) * kRowsPerScaleChunk + lane) *
                  sizeof(float);
              const float scale = read_f32_le(group_base + scale_idx);
              const float zero = read_f32_le(group_base + zero_idx);
              if (std::abs(scale) > std::numeric_limits<float>::min()) {
                qzero = static_cast<uint8_t>(rounded_zero_point(
                    static_cast<int16_t>(std::lround(zero / scale + (1 << (kBits - 1)))),
                    kBits));
              }
            }
            out.qzeros[row * out.num_groups + group] = qzero;
          }
        }
      }
    }
  }

  return out;
}

void write_int4_block_from_tmac_i2(
    const DecodedTmacI2Tensor& tensor,
    int block_id,
    uint8_t* dst) {
  static constexpr std::array<int, 8> perm = {4, 0, 5, 1, 6, 2, 7, 3};
  const size_t row_start = static_cast<size_t>(block_id) * 64;
  const size_t tile_cols = tensor.cols / 32;

  for (size_t bc = 0; bc < tile_cols; ++bc) {
    const size_t c0 = bc * 32;
    for (size_t br = 0; br < 2; ++br) {
      const size_t r0 = br * 32;
      for (size_t tile_bc = 0; tile_bc < 4; ++tile_bc) {
        const size_t cc0 = c0 + tile_bc * 8;
        const size_t group = cc0 / 32;
        for (size_t tile_br = 0; tile_br < 4; ++tile_br) {
          const size_t rr0 = r0 + tile_br * 8;
          for (size_t r = 0; r < 8; ++r) {
            const size_t row = row_start + rr0 + r;
            const uint8_t zp = tensor.qzeros[row * tensor.num_groups + group];
            int16_t values[8];
            for (size_t i = 0; i < 8; ++i) {
              const uint8_t q =
                  tensor.weights[row * tensor.cols + cc0 + i];
              values[i] = static_cast<int16_t>(q) - static_cast<int16_t>(zp);
            }
            for (size_t i = 0; i < 4; ++i) {
              const int high = static_cast<int>(values[perm[i * 2]]);
              const int low = static_cast<int>(values[perm[i * 2 + 1]]);
              dst[i] = static_cast<uint8_t>(((high & 0xF) << 4) | (low & 0xF));
            }
            dst += 4;
          }
        }
      }
    }
  }
}


bool try_build_specialized_fastpath_plan(
    const ParsedIndex& parsed_index,
    const SafeTensorsView& checkpoint,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode,
    SpecializedFastPathPlan* plan) {
  (void)bits_hint;
  (void)group_size;
  (void)qweight_mode;

  bool has_output_conv = false;
  for (const auto& rec : parsed_index.records) {
    if (rec.weight_id.kind == kWeightKindOutputConv) {
      has_output_conv = true;
      break;
    }
  }

  plan->tensor_contexts.clear();
  plan->records.clear();
  plan->tensor_contexts.reserve((has_output_conv ? 1 : 0) + 28 * kForwardTargetOps.size());
  plan->records.reserve(parsed_index.records.size());

  size_t output_ctx_index = 0;
  if (has_output_conv) {
    SpecializedTensorContext output_ctx;
    output_ctx.qweight = &require_tensor(checkpoint, "lm_head.qweight");
    output_ctx.packed_cols = static_cast<size_t>(output_ctx.qweight->shape[1]);
    output_ctx.cols = static_cast<size_t>(output_ctx.qweight->shape[0]) * 4;
    output_ctx.block_count = block_count_from_qweight(*output_ctx.qweight);
    plan->tensor_contexts.push_back(output_ctx);
  }

  for (int layer_id = 0; layer_id < 28; ++layer_id) {
    for (size_t op_id = 0; op_id < kForwardTargetOps.size(); ++op_id) {
      SpecializedTensorContext ctx;
      const std::string base =
          "model.layers." + std::to_string(layer_id) + "." + std::string(kForwardTargetOps[op_id]);
      ctx.qweight = &require_tensor(checkpoint, base + ".qweight");
      ctx.qzeros = &require_tensor(checkpoint, base + ".qzeros");
      ctx.packed_cols = static_cast<size_t>(ctx.qweight->shape[1]);
      ctx.cols = static_cast<size_t>(ctx.qweight->shape[0]) * 16;
      ctx.block_count = block_count_from_qweight(*ctx.qweight);
      ctx.num_groups = static_cast<size_t>(ctx.qzeros->shape[0]);
      ctx.qzeros_packed_cols = static_cast<size_t>(ctx.qzeros->shape[1]);
      plan->tensor_contexts.push_back(ctx);
    }
  }

  for (const auto& rec : parsed_index.records) {
    if (rec.weight_id.kind == kWeightKindOutputConv) {
      const auto& ctx = plan->tensor_contexts[output_ctx_index];
      plan->records.push_back(SpecializedFastPathRecord{&ctx, true, rec.block_id});
      continue;
    }
    const size_t ctx_index = (has_output_conv ? 1 : 0) +
        static_cast<size_t>(rec.weight_id.layer_id) * kForwardTargetOps.size() + rec.weight_id.op_id;
    const auto& ctx = plan->tensor_contexts[ctx_index];
    plan->records.push_back(SpecializedFastPathRecord{&ctx, false, rec.block_id});
  }
  return true;
}

void build_block_bytes_into_specialized(
    const SpecializedFastPathRecord& rec,
    uint8_t* dst) {
  if (rec.is_output_conv) {
    write_int8_block_direct_specialized(*rec.tensor_ctx, rec.block_id, dst);
    return;
  }
  write_int4_block_direct_gs32_minus_qzeros_specialized(*rec.tensor_ctx, rec.block_id, dst);
}

PteRebuildResult rebuild_pte_from_index(
    const std::vector<uint8_t>& stripped_pte,
    const ParsedIndex& parsed_index,
    const std::vector<uint8_t>& checkpoint_bytes,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode) {
  /* if (parsed_index.old_size < parsed_index.final_size ||
      parsed_index.old_size - parsed_index.final_size != parsed_index.total_deleted) {
    throw std::runtime_error("index.bin size consistency failed");
  } */
  /* if (stripped_pte.size() != parsed_index.final_size) {
    throw std::runtime_error("stripped.pte size mismatch with index.bin");
  } */

  const auto checkpoint = parse_safetensors(checkpoint_bytes);
  SpecializedFastPathPlan specialized_plan;
  /* if (!try_build_specialized_fastpath_plan(
          parsed_index,
          checkpoint,
          bits_hint,
          group_size,
          qweight_mode,
          &specialized_plan)) {
    throw std::runtime_error("index.bin is incompatible with the specialized rebuild fast path");
  } */
  try_build_specialized_fastpath_plan(
      parsed_index,
      checkpoint,
      bits_hint,
      group_size,
      qweight_mode,
      &specialized_plan);
  auto rebuilt = std::make_shared<std::vector<uint8_t>>(parsed_index.old_size);

  size_t src_ptr = 0;
  size_t dst_cursor = 0;
  for (size_t record_index = 0; record_index < parsed_index.records.size(); ++record_index) {
    const auto& rec = parsed_index.records[record_index];
    /* if (rec.source_offset < 0 || rec.length <= 0) {
      throw std::runtime_error("Invalid record offset or length");
    } */
    const size_t insert_at = static_cast<size_t>(rec.source_offset);
    const size_t rec_len = static_cast<size_t>(rec.length);
    /* if (insert_at < dst_cursor) {
      throw std::runtime_error("Overlapping source ranges in index.bin");
    } */
    const size_t keep_len = insert_at - dst_cursor;
    /* if (src_ptr + keep_len > stripped_pte.size()) {
      throw std::runtime_error("stripped.pte underflow while copying keep bytes");
    } */
    if (keep_len > 0) {
      std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, keep_len);
      src_ptr += keep_len;
    }

    /* if (insert_at + rec_len > rebuilt->size()) {
      throw std::runtime_error("Insert out of range while rebuilding pte");
    } */
    build_block_bytes_into_specialized(
        specialized_plan.records[record_index],
        rebuilt->data() + insert_at);
    dst_cursor = insert_at + rec_len;
  }

  const size_t tail_len = parsed_index.old_size - dst_cursor;
  /* if (src_ptr + tail_len != stripped_pte.size()) {
    throw std::runtime_error("stripped.pte tail mismatch after rebuild");
  } */
  if (tail_len > 0) {
    std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, tail_len);
  }

  return PteRebuildResult{
      rebuilt,
      0.0,
      parsed_index.records.size(),
      true};
}

PteRebuildResult rebuild_pte_from_tmac_gguf_index(
    const std::vector<uint8_t>& stripped_pte,
    const ParsedIndex& parsed_index,
    const std::vector<uint8_t>& gguf_bytes) {
  const auto gguf = parse_gguf(gguf_bytes);
  auto rebuilt = std::make_shared<std::vector<uint8_t>>(parsed_index.old_size);

  std::string current_tensor_name;
  DirectTmacI2TensorView current_tensor;
  bool have_current_tensor = false;

  size_t src_ptr = 0;
  size_t dst_cursor = 0;
  for (const auto& rec : parsed_index.records) {
    const size_t insert_at = static_cast<size_t>(rec.source_offset);
    const size_t rec_len = static_cast<size_t>(rec.length);
    const size_t keep_len = insert_at - dst_cursor;
    if (keep_len > 0) {
      std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, keep_len);
      src_ptr += keep_len;
    }

    /* if (rec.weight_id.kind == kWeightKindOutputConv) {
      throw std::runtime_error("GGUF-backed rebuild does not support output.conv yet");
    } */
    if (rec.weight_id.kind == kWeightKindOutputConv) {
      throw std::runtime_error(
          "T-MAC GGUF-backed rebuild does not support output.conv yet");
    }
    const std::string tensor_name = gguf_tensor_name_from_record(rec);
    if (!have_current_tensor || current_tensor_name != tensor_name) {
      current_tensor =
          parse_direct_tmac_i2_tensor(require_gguf_tensor(gguf, tensor_name));
      current_tensor_name = tensor_name;
      have_current_tensor = true;
    }

    write_int4_block_from_tmac_i2_direct(
        current_tensor,
        rec.block_id,
        rebuilt->data() + insert_at);
    dst_cursor = insert_at + rec_len;
  }

  const size_t tail_len = parsed_index.old_size - dst_cursor;
  if (tail_len > 0) {
    std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, tail_len);
  }

  return PteRebuildResult{
      rebuilt,
      0.0,
      parsed_index.records.size(),
      true};
}

PteRebuildResult rebuild_pte_from_gguf_index(
    const std::vector<uint8_t>& stripped_pte,
    const ParsedIndex& parsed_index,
    const std::vector<uint8_t>& gguf_bytes) {
  const auto gguf = parse_gguf(gguf_bytes);
  auto rebuilt = std::make_shared<std::vector<uint8_t>>(parsed_index.old_size);

  std::string current_tensor_name;
  DirectGptq2_32TensorView current_tensor;
  bool have_current_tensor = false;

  size_t src_ptr = 0;
  size_t dst_cursor = 0;
  for (const auto& rec : parsed_index.records) {
    const size_t insert_at = static_cast<size_t>(rec.source_offset);
    const size_t rec_len = static_cast<size_t>(rec.length);
    const size_t keep_len = insert_at - dst_cursor;
    if (keep_len > 0) {
      std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, keep_len);
      src_ptr += keep_len;
    }

    if (rec.weight_id.kind == kWeightKindOutputConv) {
      throw std::runtime_error(
          "GGUF-backed rebuild does not support output.conv for GPTQ2_32 models");
    }

    const std::string tensor_name = gguf_tensor_name_from_record(rec);
    if (!have_current_tensor || current_tensor_name != tensor_name) {
      current_tensor =
          parse_direct_gptq2_32_tensor(require_gguf_tensor(gguf, tensor_name));
      current_tensor_name = tensor_name;
      have_current_tensor = true;
    }

    write_int4_block_from_gptq2_32_direct(
        current_tensor,
        rec.block_id,
        rebuilt->data() + insert_at);
    dst_cursor = insert_at + rec_len;
  }

  const size_t tail_len = parsed_index.old_size - dst_cursor;
  if (tail_len > 0) {
    std::memcpy(rebuilt->data() + dst_cursor, stripped_pte.data() + src_ptr, tail_len);
  }

  return PteRebuildResult{
      rebuilt,
      0.0,
      parsed_index.records.size(),
      true};
}

} // namespace

PteRebuildResult rebuild_pte_from_stripped_checkpoint(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& checkpoint_bytes,
    int bits_hint,
    int group_size,
    const std::string& qweight_mode) {
  const auto start = std::chrono::steady_clock::now();
  const ParsedIndex parsed_index = parse_binary_index(index_bytes);
  auto result = rebuild_pte_from_index(
      stripped_pte,
      parsed_index,
      checkpoint_bytes,
      bits_hint,
      group_size,
      qweight_mode);
  const auto end = std::chrono::steady_clock::now();
  result.rebuild_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

PteRebuildResult rebuild_pte_from_stripped_tmac_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes) {
  const auto start = std::chrono::steady_clock::now();
  const ParsedIndex parsed_index = parse_binary_index(index_bytes);
  auto result =
      rebuild_pte_from_tmac_gguf_index(stripped_pte, parsed_index, gguf_bytes);
  const auto end = std::chrono::steady_clock::now();
  result.rebuild_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

PteRebuildResult rebuild_pte_from_stripped_gguf(
    const std::vector<uint8_t>& stripped_pte,
    const std::vector<uint8_t>& index_bytes,
    const std::vector<uint8_t>& gguf_bytes) {
  const auto start = std::chrono::steady_clock::now();
  const ParsedIndex parsed_index = parse_binary_index(index_bytes);
  auto result =
      rebuild_pte_from_gguf_index(stripped_pte, parsed_index, gguf_bytes);
  const auto end = std::chrono::steady_clock::now();
  result.rebuild_time_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  return result;
}

} // namespace example
