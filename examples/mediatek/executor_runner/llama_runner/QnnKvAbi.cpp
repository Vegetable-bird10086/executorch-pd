/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#include "QnnKvAbi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

#include <executorch/runtime/core/portable_type/half.h>

namespace example {
namespace {

template <typename T>
T Read(std::ifstream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) {
    throw std::runtime_error("truncated QNN KV ABI file");
  }
  return value;
}

int8_t SylvesterSign(size_t row, size_t column) {
  size_t bits = row & column;
  bool odd = false;
  while (bits != 0) {
    odd = !odd;
    bits &= bits - 1;
  }
  return odd ? -1 : 1;
}

void FastWalshHadamard(float* values, size_t size) {
  for (size_t width = 1; width < size; width *= 2) {
    for (size_t base = 0; base < size; base += 2 * width) {
      for (size_t index = 0; index < width; ++index) {
        const float lhs = values[base + index];
        const float rhs = values[base + width + index];
        values[base + index] = lhs + rhs;
        values[base + width + index] = lhs - rhs;
      }
    }
  }
}

uint8_t Quantize(float real, float scale, int32_t offset, QnnKvAbiStats& stats) {
  int64_t code = -static_cast<int64_t>(offset);
  if (std::isfinite(real)) {
    code = static_cast<int64_t>(
               std::nearbyint(static_cast<double>(real) / scale)) -
        offset;
  } else {
    ++stats.nonFinite;
  }
  code = std::clamp<int64_t>(code, 0, UINT8_MAX);
  stats.codeZero += code == 0;
  stats.code255 += code == UINT8_MAX;
  ++stats.values;
  return static_cast<uint8_t>(code);
}

} // namespace

QnnKvAbi::QnnKvAbi(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("unable to open QNN KV ABI: " + path);
  }
  std::array<char, 8> magic{};
  input.read(magic.data(), magic.size());
  const std::array<char, 8> expected = {'Q', 'N', 'N', 'K', 'V', 'A', 'B', 'I'};
  if (!input || magic != expected || Read<uint32_t>(input) != 1) {
    throw std::runtime_error("unsupported QNN KV ABI header");
  }
  numLayers_ = Read<uint32_t>(input);
  numHeads_ = Read<uint32_t>(input);
  headDim_ = Read<uint32_t>(input);
  if (numLayers_ == 0 || numHeads_ == 0 || headDim_ == 0 ||
      (headDim_ & (headDim_ - 1)) != 0) {
    throw std::runtime_error("invalid QNN KV ABI dimensions");
  }
  layers_.resize(numLayers_);
  std::vector<int8_t> matrix(headDim_ * headDim_);
  for (size_t layer = 0; layer < numLayers_; ++layer) {
    Layer& metadata = layers_[layer];
    metadata.rotationUnit = Read<float>(input);
    input.read(reinterpret_cast<char*>(matrix.data()), matrix.size());
    if (!input || !(metadata.rotationUnit > 0.0f) ||
        !std::isfinite(metadata.rotationUnit)) {
      throw std::runtime_error("invalid QNN K rotation metadata");
    }
    for (size_t row = 0; row < headDim_; ++row) {
      for (size_t column = 0; column < headDim_; ++column) {
        if (matrix[row * headDim_ + column] != SylvesterSign(row, column)) {
          throw std::runtime_error(
              "QNN K rotation is not the validated Sylvester Hadamard");
        }
      }
    }
    metadata.key.resize(numHeads_);
    metadata.value.resize(numHeads_);
    for (size_t kind = 0; kind < 2; ++kind) {
      auto& qparams = kind == 0 ? metadata.key : metadata.value;
      for (Affine& affine : qparams) {
        affine.scale = Read<float>(input);
        affine.offset = Read<int32_t>(input);
        if (!(affine.scale > 0.0f) || !std::isfinite(affine.scale)) {
          throw std::runtime_error("invalid QNN KV affine scale");
        }
      }
    }
  }
  if (input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("QNN KV ABI has trailing data");
  }
}

size_t QnnKvAbi::NumLayers() const {
  return numLayers_;
}

size_t QnnKvAbi::NumHeads() const {
  return numHeads_;
}

size_t QnnKvAbi::HeadDim() const {
  return headDim_;
}

std::vector<uint8_t> QnnKvAbi::ConvertCanonicalFp16(
    const std::vector<uint16_t>& canonicalKv,
    size_t promptLength,
    QnnKvAbiStats* stats) const {
  const size_t perKind = numLayers_ * numHeads_ * promptLength * headDim_;
  if (promptLength == 0 || canonicalKv.size() != 2 * perKind) {
    throw std::runtime_error("canonical FP16 KV size does not match QNN ABI");
  }
  std::vector<uint8_t> output(canonicalKv.size());
  ConvertCanonicalFp16Layers(
      canonicalKv, 0, promptLength, output, stats);
  return output;
}

void QnnKvAbi::ConvertCanonicalFp16Layers(
    const std::vector<uint16_t>& localCanonicalKv,
    size_t firstLayer,
    size_t promptLength,
    std::vector<uint8_t>& output,
    QnnKvAbiStats* accumulatedStats) const {
  const size_t layerStride = numHeads_ * promptLength * headDim_;
  const size_t outputPerKind = numLayers_ * layerStride;
  if (promptLength == 0 || localCanonicalKv.empty() ||
      localCanonicalKv.size() % (2 * layerStride) != 0 ||
      output.size() != 2 * outputPerKind) {
    throw std::runtime_error("layer-sliced FP16 KV does not match QNN ABI");
  }
  const size_t localLayers = localCanonicalKv.size() / (2 * layerStride);
  if (firstLayer + localLayers > numLayers_) {
    throw std::runtime_error("layer-sliced FP16 KV exceeds QNN ABI layers");
  }
  const size_t localPerKind = localLayers * layerStride;
  QnnKvAbiStats local{};
  std::vector<float> transformed(headDim_);
  for (size_t kind = 0; kind < 2; ++kind) {
    const size_t sourceKindBase = kind * localPerKind;
    const size_t outputKindBase = kind * outputPerKind;
    for (size_t localLayer = 0; localLayer < localLayers; ++localLayer) {
      const size_t layer = firstLayer + localLayer;
      const Layer& metadata = layers_[layer];
      const auto& qparams = kind == 0 ? metadata.key : metadata.value;
      for (size_t head = 0; head < numHeads_; ++head) {
        const size_t sourceHeadBase = sourceKindBase +
            (localLayer * numHeads_ + head) * promptLength * headDim_;
        const size_t outputHeadBase = outputKindBase +
            (layer * numHeads_ + head) * promptLength * headDim_;
        for (size_t token = 0; token < promptLength; ++token) {
          const size_t sourceTokenBase =
              sourceHeadBase + token * headDim_;
          const size_t outputTokenBase =
              outputHeadBase + token * headDim_;
          for (size_t dim = 0; dim < headDim_; ++dim) {
            transformed[dim] = executorch::runtime::etensor::internal::
                fp16_ieee_to_fp32_value(
                    localCanonicalKv[sourceTokenBase + dim]);
          }
          if (kind == 0) {
            FastWalshHadamard(transformed.data(), headDim_);
          }
          for (size_t dim = 0; dim < headDim_; ++dim) {
            const float real = kind == 0
                ? transformed[dim] * metadata.rotationUnit
                : transformed[dim];
            output[outputTokenBase + dim] = Quantize(
                real, qparams[head].scale, qparams[head].offset, local);
          }
        }
      }
    }
  }
  if (accumulatedStats != nullptr) {
    accumulatedStats->values += local.values;
    accumulatedStats->nonFinite += local.nonFinite;
    accumulatedStats->codeZero += local.codeZero;
    accumulatedStats->code255 += local.code255;
  }
}

} // namespace example
