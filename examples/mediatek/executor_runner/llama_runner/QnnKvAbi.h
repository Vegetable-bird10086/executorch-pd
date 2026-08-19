/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace example {

struct QnnKvAbiStats {
  size_t values{0};
  size_t nonFinite{0};
  size_t codeZero{0};
  size_t code255{0};
};

// Compact, backend-independent description of the QNN Decode KV boundary.
// The source is the validated llama.cpp QNN runtime profile. MTK uses this
// only for PD handoff; normal MTK generation never constructs this object.
class QnnKvAbi {
 public:
  explicit QnnKvAbi(const std::string& path);

  size_t NumLayers() const;
  size_t NumHeads() const;
  size_t HeadDim() const;

  std::vector<uint8_t> ConvertCanonicalFp16(
      const std::vector<uint16_t>& canonicalKv,
      size_t promptLength,
      QnnKvAbiStats* stats = nullptr) const;

  void ConvertCanonicalFp16Layers(
      const std::vector<uint16_t>& localCanonicalKv,
      size_t firstLayer,
      size_t promptLength,
      std::vector<uint8_t>& output,
      QnnKvAbiStats* accumulatedStats = nullptr) const;

 private:
  struct Affine {
    float scale{0.0f};
    int32_t offset{0};
  };
  struct Layer {
    float rotationUnit{0.0f};
    std::vector<Affine> key;
    std::vector<Affine> value;
  };

  size_t numLayers_{0};
  size_t numHeads_{0};
  size_t headDim_{0};
  std::vector<Layer> layers_;
};

} // namespace example
