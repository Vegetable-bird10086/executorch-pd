/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace example {

class SeparateEmbedding {
 public:
  SeparateEmbedding() = default;
  SeparateEmbedding(const SeparateEmbedding&) = delete;
  SeparateEmbedding& operator=(const SeparateEmbedding&) = delete;
  ~SeparateEmbedding();
  bool load(const std::string& matrix_path, bool resident = true);
  bool is_loaded() const {
    return loaded_;
  }
  bool is_quantized() const {
    return quantized_;
  }
  int32_t vocab_size() const {
    return vocab_size_;
  }
  int32_t embedding_dim() const {
    return embedding_dim_;
  }

  uint32_t embedding_dtype_code() const;
  size_t embedding_elem_size() const;
  size_t row_bytes() const;
  void copy_row(uint64_t token_id, uint8_t* out_embedding_row, size_t bytes) const;
  void copy_row_to_float(
      uint64_t token_id,
      float* out_embedding_row,
      size_t elems) const;
  float quant_scale() const;
  int32_t quant_zero_point() const;
  void dequantize_row_to_float(
      uint64_t token_id,
      float* out_embedding_row,
      size_t elems) const;

 private:
  struct TensorBlock {
    uint32_t dtype_code{0};
    std::vector<uint32_t> shape;
    uint64_t nbytes{0};
    uint64_t data_offset{0};
    std::vector<uint8_t> data;
  };

  bool read_tensor_block(
      std::ifstream& matrix_file,
      TensorBlock* block,
      bool load_payload);
  bool read_from_file(
      const TensorBlock& block,
      uint64_t offset,
      void* dst,
      size_t bytes) const;
  void validate_shape_or_throw(
      const TensorBlock& block,
      const std::string& block_name) const;

  bool loaded_{false};
  int matrix_fd_{-1};
  bool quantized_{false};
  bool resident_{true};
  int32_t vocab_size_{0};
  int32_t embedding_dim_{0};
  TensorBlock qweight_;
  TensorBlock scale_;
  TensorBlock zp_;
  TensorBlock weight_;
};

} // namespace example
