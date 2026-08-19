/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#include "MtkPteWeightRebuilder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace example {
namespace {

constexpr std::array<uint8_t, 8> kMagic = {
    'M', 'T', 'K', 'S', 'T', 'R', 'P', '1'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kRecordSize = 32;
constexpr size_t kHeaderSize = 48;

struct Header {
  uint64_t originalSize{0};
  uint64_t strippedSize{0};
  uint64_t weightsSize{0};
  uint32_t recordCount{0};
};

struct Record {
  uint64_t sourceOffset{0};
  uint64_t length{0};
  uint64_t payloadOffset{0};
  uint32_t kind{0};
};

uint32_t ReadU32(const uint8_t* data) {
  uint32_t value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

uint64_t ReadU64(const uint8_t* data) {
  uint64_t value;
  std::memcpy(&value, data, sizeof(value));
  return value;
}

size_t FileSize(std::ifstream& stream, const std::string& path) {
  stream.seekg(0, std::ios::end);
  const auto end = stream.tellg();
  if (end < 0) {
    throw std::runtime_error("Unable to determine file size: " + path);
  }
  stream.seekg(0, std::ios::beg);
  return static_cast<size_t>(end);
}

std::vector<uint8_t> ReadFile(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Unable to open file: " + path);
  }
  const size_t size = FileSize(stream, path);
  std::vector<uint8_t> data(size);
  if (size != 0) {
    stream.read(reinterpret_cast<char*>(data.data()), size);
  }
  if (!stream) {
    throw std::runtime_error("Unable to read file: " + path);
  }
  return data;
}

std::pair<Header, std::vector<Record>> ParseIndex(const std::string& path) {
  const auto data = ReadFile(path);
  if (data.size() < kHeaderSize ||
      !std::equal(kMagic.begin(), kMagic.end(), data.begin())) {
    throw std::runtime_error("Invalid MTK strip index header: " + path);
  }
  if (ReadU32(data.data() + 8) != kVersion ||
      ReadU32(data.data() + 12) != kRecordSize) {
    throw std::runtime_error("Unsupported MTK strip index version: " + path);
  }
  Header header{
      ReadU64(data.data() + 16),
      ReadU64(data.data() + 24),
      ReadU64(data.data() + 32),
      ReadU32(data.data() + 40),
  };
  const size_t expected = kHeaderSize + header.recordCount * kRecordSize;
  if (data.size() != expected) {
    throw std::runtime_error("MTK strip index size mismatch: " + path);
  }
  std::vector<Record> records;
  records.reserve(header.recordCount);
  for (size_t i = 0; i < header.recordCount; ++i) {
    const auto* record = data.data() + kHeaderSize + i * kRecordSize;
    records.push_back(Record{
        ReadU64(record),
        ReadU64(record + 8),
        ReadU64(record + 16),
        ReadU32(record + 24),
    });
  }
  return {header, std::move(records)};
}

void ReadAt(
    std::ifstream& stream,
    const std::string& path,
    uint64_t offset,
    uint8_t* destination,
    size_t length) {
  if (length == 0) {
    return;
  }
  stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  stream.read(reinterpret_cast<char*>(destination), length);
  if (!stream) {
    throw std::runtime_error("Short read while rebuilding from: " + path);
  }
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

MtkPteRebuildResult RebuildMtkPteWeights(
    const std::string& strippedPtePath,
    const std::string& indexPath,
    const std::string& weightsPath) {
  const auto totalStart = std::chrono::steady_clock::now();
  const auto [header, records] = ParseIndex(indexPath);
  std::ifstream stripped(strippedPtePath, std::ios::binary);
  std::ifstream weights(weightsPath, std::ios::binary);
  if (!stripped || !weights) {
    throw std::runtime_error("Unable to open MTK stripped PTE or weight payload");
  }
  if (FileSize(stripped, strippedPtePath) != header.strippedSize ||
      FileSize(weights, weightsPath) != header.weightsSize) {
    throw std::runtime_error("MTK strip input size does not match index");
  }

  auto output = std::make_shared<std::vector<uint8_t>>(header.originalSize);
  uint64_t strippedCursor = 0;
  uint64_t destinationCursor = 0;
  double staticCopyMs = 0.0;
  double weightCopyMs = 0.0;
  for (const auto& record : records) {
    if (record.sourceOffset < destinationCursor ||
        record.sourceOffset + record.length > header.originalSize ||
        record.payloadOffset + record.length > header.weightsSize) {
      throw std::runtime_error("Invalid or overlapping MTK strip record");
    }
    const size_t keep = record.sourceOffset - destinationCursor;
    auto start = std::chrono::steady_clock::now();
    ReadAt(
        stripped,
        strippedPtePath,
        strippedCursor,
        output->data() + destinationCursor,
        keep);
    staticCopyMs += ElapsedMs(start);
    start = std::chrono::steady_clock::now();
    ReadAt(
        weights,
        weightsPath,
        record.payloadOffset,
        output->data() + record.sourceOffset,
        record.length);
    weightCopyMs += ElapsedMs(start);
    strippedCursor += keep;
    destinationCursor = record.sourceOffset + record.length;
  }
  const size_t tail = header.originalSize - destinationCursor;
  const auto tailStart = std::chrono::steady_clock::now();
  ReadAt(
      stripped,
      strippedPtePath,
      strippedCursor,
      output->data() + destinationCursor,
      tail);
  staticCopyMs += ElapsedMs(tailStart);
  if (strippedCursor + tail != header.strippedSize) {
    throw std::runtime_error("MTK stripped PTE was not consumed exactly");
  }

  return MtkPteRebuildResult{
      output,
      MtkPteRebuildStats{
          ElapsedMs(totalStart),
          staticCopyMs,
          weightCopyMs,
          static_cast<size_t>(header.originalSize),
          static_cast<size_t>(header.strippedSize),
          static_cast<size_t>(header.weightsSize),
          records.size(),
      },
  };
}

namespace {

constexpr std::array<uint8_t, 8> kGgufIndexMagic = {
    'M', 'T', 'K', 'G', 'G', 'U', 'F', '2'};
constexpr uint32_t kGgufIndexVersion = 2;
constexpr uint32_t kGgufIndexRecordSize = 40;
constexpr size_t kGgufIndexHeaderSize = 48;
constexpr uint32_t kGgufTensorTypeGptq2_32 = 42;

enum class GgufValueType : uint32_t {
  Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3, Uint32 = 4,
  Int32 = 5, Float32 = 6, Bool = 7, String = 8, Array = 9,
  Uint64 = 10, Int64 = 11, Float64 = 12,
};

struct MtkGgufRecord {
  uint64_t sourceOffset{0};
  uint64_t length{0};
  uint32_t layer{0};
  uint32_t op{0};
  uint32_t rows{0};
  uint32_t cols{0};
  uint32_t kind{0};
  const uint8_t* tensor{nullptr};
  size_t tensorBytes{0};
};

struct MtkGgufIndexHeader {
  uint64_t originalSize{0};
  uint64_t strippedSize{0};
  uint64_t materializedSize{0};
  uint32_t recordCount{0};
  uint32_t sourceGroupSize{0};
};

struct GgufTensor {
  std::vector<uint64_t> shape;
  uint32_t type{0};
  const uint8_t* data{nullptr};
  size_t bytes{0};
};

size_t GgufScalarSize(GgufValueType type) {
  switch (type) {
    case GgufValueType::Uint8:
    case GgufValueType::Int8:
    case GgufValueType::Bool:
      return 1;
    case GgufValueType::Uint16:
    case GgufValueType::Int16:
      return 2;
    case GgufValueType::Uint32:
    case GgufValueType::Int32:
    case GgufValueType::Float32:
      return 4;
    case GgufValueType::Uint64:
    case GgufValueType::Int64:
    case GgufValueType::Float64:
      return 8;
    default:
      return 0;
  }
}

void RequireRange(size_t cursor, size_t amount, size_t size, const char* what) {
  if (cursor > size || amount > size - cursor) {
    throw std::runtime_error(std::string("Truncated ") + what);
  }
}

std::string ReadGgufString(const uint8_t* data, size_t size, size_t* cursor) {
  RequireRange(*cursor, 8, size, "GGUF string length");
  const size_t length = static_cast<size_t>(ReadU64(data + *cursor));
  *cursor += 8;
  RequireRange(*cursor, length, size, "GGUF string");
  std::string value(reinterpret_cast<const char*>(data + *cursor), length);
  *cursor += length;
  return value;
}

void SkipGgufValue(
    const uint8_t* data,
    size_t size,
    size_t* cursor,
    GgufValueType type) {
  if (type == GgufValueType::String) {
    (void)ReadGgufString(data, size, cursor);
    return;
  }
  if (type == GgufValueType::Array) {
    RequireRange(*cursor, 12, size, "GGUF array header");
    const auto item = static_cast<GgufValueType>(ReadU32(data + *cursor));
    *cursor += 4;
    const size_t count = static_cast<size_t>(ReadU64(data + *cursor));
    *cursor += 8;
    for (size_t index = 0; index < count; ++index) {
      SkipGgufValue(data, size, cursor, item);
    }
    return;
  }
  const size_t scalar = GgufScalarSize(type);
  if (scalar == 0) {
    throw std::runtime_error("Unsupported GGUF metadata type");
  }
  RequireRange(*cursor, scalar, size, "GGUF metadata value");
  *cursor += scalar;
}

size_t AlignUp(size_t value, size_t alignment) {
  const size_t remainder = alignment == 0 ? 0 : value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

const char* GgufOpName(uint32_t op) {
  static constexpr std::array<const char*, 7> names = {
      "attn_q", "attn_k", "attn_v", "attn_output",
      "ffn_gate", "ffn_up", "ffn_down"};
  if (op >= names.size()) {
    throw std::runtime_error("Invalid MTK GGUF op id");
  }
  return names[op];
}

float ReadF16(const uint8_t* data) {
  const uint16_t bits = static_cast<uint16_t>(data[0]) |
      static_cast<uint16_t>(data[1]) << 8;
  const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
  const uint32_t exponent = (bits >> 10) & 0x1FU;
  const uint32_t mantissa = bits & 0x03FFU;
  uint32_t fp32 = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      fp32 = sign;
    } else {
      uint32_t mant = mantissa;
      int exp = -14;
      while ((mant & 0x0400U) == 0) {
        mant <<= 1;
        --exp;
      }
      fp32 = sign | static_cast<uint32_t>((exp + 127) << 23) |
          ((mant & 0x03FFU) << 13);
    }
  } else if (exponent == 0x1FU) {
    fp32 = sign | 0x7F800000U | (mantissa << 13);
  } else {
    fp32 = sign |
        static_cast<uint32_t>((static_cast<int>(exponent) + 112) << 23) |
        (mantissa << 13);
  }
  float value = 0.0f;
  std::memcpy(&value, &fp32, sizeof(value));
  return value;
}

uint8_t DecodeZeroPoint(const uint8_t* metadata) {
  const float scale = std::max(ReadF16(metadata), 1.0e-4f);
  const float zeroBias = ReadF16(metadata + 2);
  return static_cast<uint8_t>(std::max(0L, std::min(3L, std::lround(zeroBias / scale))));
}

size_t MtkInt4Offset(size_t row, size_t col, size_t rows, size_t cols) {
  return (row / 16) * (2 * cols) +
      ((row % 16) / 4) * (rows * cols / 8) +
      (col / 32) * 64 + (row % 4) * 16 + (col % 32) / 2;
}

void MaterializeGptq2Gs32ToMtkInt4(
    const MtkGgufRecord& record,
    uint32_t sourceGroupSize,
    uint8_t* destination) {
  if (record.kind != 4 || sourceGroupSize != 32 || record.rows % 64 != 0 ||
      record.cols % 32 != 0 || record.length != record.rows * record.cols / 2) {
    throw std::runtime_error("Unsupported MTK GPTQ2 record dimensions or precision");
  }
  constexpr size_t rowsPerBlock = 64;
  constexpr size_t qbytesPerGroup = rowsPerBlock * 8;
  constexpr size_t metadataPerGroup = rowsPerBlock * 4;
  constexpr size_t bytesPerGroup = qbytesPerGroup + metadataPerGroup;
  const size_t groups = record.cols / 32;
  const size_t blockBytes = groups * bytesPerGroup;
  if (record.tensorBytes != (record.rows / rowsPerBlock) * blockBytes) {
    throw std::runtime_error("Unexpected GPTQ2_32 GS32 tensor byte count");
  }
  std::memset(destination, 0, static_cast<size_t>(record.length));
  for (size_t block = 0; block < record.rows / rowsPerBlock; ++block) {
    const uint8_t* blockData = record.tensor + block * blockBytes;
    for (size_t localRow = 0; localRow < rowsPerBlock; ++localRow) {
      float minimum = std::numeric_limits<float>::infinity();
      float maximum = -std::numeric_limits<float>::infinity();
      for (size_t groupIndex = 0; groupIndex < groups; ++groupIndex) {
        const uint8_t* group = blockData + groupIndex * bytesPerGroup;
        const uint8_t* metadata = group + qbytesPerGroup;
        const uint8_t zeroPoint = DecodeZeroPoint(metadata + localRow * 4);
        const float sourceScale =
            std::max(ReadF16(metadata + localRow * 4), 1.0e-4f);
        const size_t br = localRow / 32;
        const size_t tileRow = (localRow % 32) / 8;
        const size_t laneRow = localRow % 8;
        for (size_t tileCol = 0; tileCol < 4; ++tileCol) {
          const size_t pairOffset =
              ((((br * 4 + tileCol) * 4 + tileRow) * 8 + laneRow) * 2);
          for (size_t within = 0; within < 8; ++within) {
            const uint8_t packed2 = group[pairOffset + within / 4];
            const uint8_t code = (packed2 >> ((within % 4) * 2)) & 0x3;
            const float weight =
                (static_cast<int>(code) - static_cast<int>(zeroPoint)) *
                sourceScale;
            minimum = std::min(minimum, weight);
            maximum = std::max(maximum, weight);
          }
        }
      }
      const float targetScale = std::max((maximum - minimum) / 15.0f, 1.0e-6f);
      const int targetZeroPoint = std::max(
          -8,
          std::min(7, static_cast<int>(std::nearbyint(-8.0f - minimum / targetScale))));
      for (size_t groupIndex = 0; groupIndex < groups; ++groupIndex) {
        const uint8_t* group = blockData + groupIndex * bytesPerGroup;
        const uint8_t* metadata = group + qbytesPerGroup;
        const uint8_t sourceZeroPoint = DecodeZeroPoint(metadata + localRow * 4);
        const float sourceScale =
            std::max(ReadF16(metadata + localRow * 4), 1.0e-4f);
        const size_t br = localRow / 32;
        const size_t tileRow = (localRow % 32) / 8;
        const size_t laneRow = localRow % 8;
        for (size_t tileCol = 0; tileCol < 4; ++tileCol) {
          const size_t pairOffset =
              ((((br * 4 + tileCol) * 4 + tileRow) * 8 + laneRow) * 2);
          for (size_t within = 0; within < 8; ++within) {
            const uint8_t packed2 = group[pairOffset + within / 4];
            const uint8_t code = (packed2 >> ((within % 4) * 2)) & 0x3;
            const float weight =
                (static_cast<int>(code) - static_cast<int>(sourceZeroPoint)) *
                sourceScale;
            const int quantized = std::max(
                -8,
                std::min(
                    7,
                    static_cast<int>(std::nearbyint(weight / targetScale)) +
                        targetZeroPoint));
            const uint8_t nibble = static_cast<uint8_t>(quantized & 0xF);
            const size_t row = block * rowsPerBlock + localRow;
            const size_t col = groupIndex * 32 + tileCol * 8 + within;
            const size_t offset = MtkInt4Offset(row, col, record.rows, record.cols);
            destination[offset] |= static_cast<uint8_t>(
                nibble << ((col & 1) ? 4 : 0));
          }
        }
      }
    }
  }
}

} // namespace

class MtkGgufWeightSource {
 public:
  explicit MtkGgufWeightSource(const std::string& path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) {
      throw std::runtime_error("Unable to open MTK GGUF source: " + path);
    }
    struct stat status {};
    if (fstat(fd_, &status) != 0 || status.st_size <= 0) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Unable to stat MTK GGUF source: " + path);
    }
    size_ = static_cast<size_t>(status.st_size);
    void* mapped = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      throw std::runtime_error("Unable to mmap MTK GGUF source: " + path);
    }
    data_ = static_cast<const uint8_t*>(mapped);
    Parse();
  }

  ~MtkGgufWeightSource() {
    if (data_) {
      munmap(const_cast<uint8_t*>(data_), size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  void DiscardPages() {
    if (data_) {
      (void)madvise(const_cast<uint8_t*>(data_), size_, MADV_DONTNEED);
    }
  }

  const GgufTensor& Tensor(const std::string& name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
      throw std::runtime_error("Missing MTK rebuild GGUF tensor: " + name);
    }
    return found->second;
  }

  bool IsGs32Source() const { return gs32Source_; }

 private:
  void Parse() {
    RequireRange(0, 24, size_, "GGUF header");
    if (ReadU32(data_) != 0x46554747U || ReadU32(data_ + 4) != 3) {
      throw std::runtime_error("MTK rebuild requires GGUF v3");
    }
    const size_t tensorCount = static_cast<size_t>(ReadU64(data_ + 8));
    const size_t kvCount = static_cast<size_t>(ReadU64(data_ + 16));
    size_t cursor = 24;
    size_t alignment = 32;
    for (size_t index = 0; index < kvCount; ++index) {
      const std::string key = ReadGgufString(data_, size_, &cursor);
      RequireRange(cursor, 4, size_, "GGUF metadata type");
      const auto type = static_cast<GgufValueType>(ReadU32(data_ + cursor));
      cursor += 4;
      if (key == "general.alignment" && type == GgufValueType::Uint32) {
        RequireRange(cursor, 4, size_, "GGUF alignment");
        alignment = ReadU32(data_ + cursor);
      }
      if (key == "general.gptq2_32.layout" && type == GgufValueType::String) {
        gs32Source_ = ReadGgufString(data_, size_, &cursor) == "gs32_source_v1";
      } else {
        SkipGgufValue(data_, size_, &cursor, type);
      }
    }
    struct Stub {
      std::string name;
      std::vector<uint64_t> shape;
      uint32_t type;
      size_t offset;
    };
    std::vector<Stub> stubs;
    stubs.reserve(tensorCount);
    for (size_t index = 0; index < tensorCount; ++index) {
      Stub stub;
      stub.name = ReadGgufString(data_, size_, &cursor);
      RequireRange(cursor, 4, size_, "GGUF tensor rank");
      const size_t rank = ReadU32(data_ + cursor);
      cursor += 4;
      RequireRange(cursor, rank * 8 + 12, size_, "GGUF tensor descriptor");
      for (size_t dim = 0; dim < rank; ++dim) {
        stub.shape.push_back(ReadU64(data_ + cursor));
        cursor += 8;
      }
      stub.type = ReadU32(data_ + cursor);
      cursor += 4;
      stub.offset = static_cast<size_t>(ReadU64(data_ + cursor));
      cursor += 8;
      stubs.push_back(std::move(stub));
    }
    const size_t dataStart = AlignUp(cursor, alignment);
    std::sort(stubs.begin(), stubs.end(), [](const Stub& left, const Stub& right) {
      return left.offset < right.offset;
    });
    for (size_t index = 0; index < stubs.size(); ++index) {
      const size_t next = index + 1 < stubs.size()
          ? stubs[index + 1].offset
          : size_ - dataStart;
      if (stubs[index].offset > next || dataStart + next > size_) {
        throw std::runtime_error("Invalid GGUF tensor offsets");
      }
      tensors_.emplace(stubs[index].name, GgufTensor{
          stubs[index].shape,
          stubs[index].type,
          data_ + dataStart + stubs[index].offset,
          next - stubs[index].offset});
    }
    if (!gs32Source_) {
      throw std::runtime_error(
          "MTK rebuild currently requires general.gptq2_32.layout=gs32_source_v1");
    }
  }

  std::string path_;
  int fd_{-1};
  const uint8_t* data_{nullptr};
  size_t size_{0};
  bool gs32Source_{false};
  std::unordered_map<std::string, GgufTensor> tensors_;
};

class MtkGgufPteRecipe {
 public:
  std::shared_ptr<MtkGgufWeightSource> source;
  MtkGgufIndexHeader header;
  std::vector<MtkGgufRecord> records;
};

std::shared_ptr<MtkGgufWeightSource> OpenMtkGgufWeightSource(
    const std::string& ggufPath) {
  return std::make_shared<MtkGgufWeightSource>(ggufPath);
}

std::shared_ptr<MtkGgufPteRecipe> PrepareMtkGgufPteRecipe(
    const std::shared_ptr<MtkGgufWeightSource>& source,
    const std::string& indexPath) {
  if (!source) {
    throw std::runtime_error("MTK GGUF source is null");
  }
  const auto data = ReadFile(indexPath);
  if (data.size() < kGgufIndexHeaderSize ||
      !std::equal(kGgufIndexMagic.begin(), kGgufIndexMagic.end(), data.begin()) ||
      ReadU32(data.data() + 8) != kGgufIndexVersion ||
      ReadU32(data.data() + 12) != kGgufIndexRecordSize) {
    throw std::runtime_error("Invalid MTK GGUF rebuild index: " + indexPath);
  }
  auto recipe = std::make_shared<MtkGgufPteRecipe>();
  recipe->source = source;
  recipe->header = MtkGgufIndexHeader{
      ReadU64(data.data() + 16), ReadU64(data.data() + 24),
      ReadU64(data.data() + 32), ReadU32(data.data() + 40),
      ReadU32(data.data() + 44)};
  if (data.size() != kGgufIndexHeaderSize +
          recipe->header.recordCount * kGgufIndexRecordSize) {
    throw std::runtime_error("MTK GGUF rebuild index size mismatch");
  }
  for (size_t index = 0; index < recipe->header.recordCount; ++index) {
    const uint8_t* ptr = data.data() + kGgufIndexHeaderSize +
        index * kGgufIndexRecordSize;
    MtkGgufRecord record{
        ReadU64(ptr), ReadU64(ptr + 8), ReadU32(ptr + 16),
        ReadU32(ptr + 20), ReadU32(ptr + 24), ReadU32(ptr + 28),
        ReadU32(ptr + 32), nullptr, 0};
    const std::string tensorName = "blk." + std::to_string(record.layer) + "." +
        GgufOpName(record.op) + ".weight";
    const auto& tensor = source->Tensor(tensorName);
    if (tensor.type != kGgufTensorTypeGptq2_32 || tensor.shape.size() != 2 ||
        tensor.shape[0] != record.cols || tensor.shape[1] != record.rows) {
      throw std::runtime_error("MTK GGUF tensor type/shape mismatch: " + tensorName);
    }
    record.tensor = tensor.data;
    record.tensorBytes = tensor.bytes;
    recipe->records.push_back(record);
  }
  return recipe;
}

MtkPteRebuildResult RebuildMtkPteWeightsFromGguf(
    const std::string& strippedPtePath,
    const MtkGgufPteRecipe& recipe) {
  const auto totalStart = std::chrono::steady_clock::now();
  std::ifstream stripped(strippedPtePath, std::ios::binary);
  if (!stripped || FileSize(stripped, strippedPtePath) != recipe.header.strippedSize) {
    throw std::runtime_error("MTK stripped PTE size does not match GGUF index");
  }
  auto output = std::make_shared<std::vector<uint8_t>>(recipe.header.originalSize);
  uint64_t strippedCursor = 0;
  uint64_t destinationCursor = 0;
  double staticCopyMs = 0.0;
  double materializeMs = 0.0;
  for (const auto& record : recipe.records) {
    if (record.sourceOffset < destinationCursor ||
        record.sourceOffset + record.length > recipe.header.originalSize) {
      throw std::runtime_error("Invalid or overlapping MTK GGUF rebuild record");
    }
    const size_t keep = record.sourceOffset - destinationCursor;
    auto start = std::chrono::steady_clock::now();
    ReadAt(stripped, strippedPtePath, strippedCursor,
        output->data() + destinationCursor, keep);
    staticCopyMs += ElapsedMs(start);
    strippedCursor += keep;
    destinationCursor = record.sourceOffset + record.length;
  }
  const size_t tail = recipe.header.originalSize - destinationCursor;
  const auto tailStart = std::chrono::steady_clock::now();
  ReadAt(stripped, strippedPtePath, strippedCursor,
      output->data() + destinationCursor, tail);
  staticCopyMs += ElapsedMs(tailStart);
  if (strippedCursor + tail != recipe.header.strippedSize) {
    throw std::runtime_error("MTK stripped PTE was not consumed exactly");
  }
  const auto materializeStart = std::chrono::steady_clock::now();
  const size_t workerCount = std::min<size_t>(4, recipe.records.size());
  std::atomic<size_t> nextRecord{0};
  std::exception_ptr workerError;
  std::mutex workerErrorMutex;
  const auto materializeRecords = [&]() {
    try {
      while (true) {
        const size_t index = nextRecord.fetch_add(1);
        if (index >= recipe.records.size()) {
          break;
        }
        const auto& record = recipe.records[index];
        MaterializeGptq2Gs32ToMtkInt4(
            record, recipe.header.sourceGroupSize,
            output->data() + record.sourceOffset);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(workerErrorMutex);
      if (!workerError) {
        workerError = std::current_exception();
      }
    }
  };
  if (workerCount <= 1) {
    materializeRecords();
  } else {
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (size_t worker = 0; worker < workerCount; ++worker) {
      workers.emplace_back(materializeRecords);
    }
    for (auto& worker : workers) {
      worker.join();
    }
  }
  if (workerError) {
    std::rethrow_exception(workerError);
  }
  materializeMs = ElapsedMs(materializeStart);
  // Each reconstructed PTE owns its materialized bytes. Drop this mapping's
  // source pages immediately so a 14-chunk request does not retain every GGUF
  // tensor touched so far. The mmap remains valid and later chunks refault only
  // their own tensors; llama.cpp Decode uses a separate mapping.
  recipe.source->DiscardPages();
  return MtkPteRebuildResult{
      output,
      MtkPteRebuildStats{
          ElapsedMs(totalStart), staticCopyMs, materializeMs,
          static_cast<size_t>(recipe.header.originalSize),
          static_cast<size_t>(recipe.header.strippedSize),
          static_cast<size_t>(recipe.header.materializedSize),
          recipe.records.size()}};
}

void DiscardMtkGgufSourcePages(
    const std::shared_ptr<MtkGgufWeightSource>& source) {
  if (source) {
    source->DiscardPages();
  }
}

} // namespace example
