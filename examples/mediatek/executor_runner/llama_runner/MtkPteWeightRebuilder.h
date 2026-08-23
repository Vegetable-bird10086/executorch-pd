/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace example {

struct MtkPteRebuildStats {
  double total_ms{0.0};
  double static_copy_ms{0.0};
  double weight_copy_ms{0.0};
  size_t original_size{0};
  size_t stripped_size{0};
  size_t weight_size{0};
  size_t record_count{0};
};

struct MtkPteRebuildResult {
  std::shared_ptr<std::vector<uint8_t>> pte;
  MtkPteRebuildStats stats;
};

class MtkGgufWeightSource;
class MtkGgufPteRecipe;

// Parse and retain a read-only mmap of the shared 2-bit GGUF.  Recipes keep
// this object alive and resolve tensor pointers once during Prepare().
std::shared_ptr<MtkGgufWeightSource> OpenMtkGgufWeightSource(
    const std::string& ggufPath);

// QNN-aligned joint-PD model store: read the GGUF once into a sealed memfd and
// expose one stable read-only pointer to both MTK rebuild and llama.cpp Decode.
std::shared_ptr<MtkGgufWeightSource> LoadMtkGgufWeightSourceIntoRam(
    const std::string& ggufPath);
const void* MtkGgufWeightSourceData(
    const std::shared_ptr<MtkGgufWeightSource>& source);
size_t MtkGgufWeightSourceSize(
    const std::shared_ptr<MtkGgufWeightSource>& source);
bool MtkGgufWeightSourceIsRamStore(
    const std::shared_ptr<MtkGgufWeightSource>& source);

std::shared_ptr<MtkGgufPteRecipe> PrepareMtkGgufPteRecipe(
    const std::shared_ptr<MtkGgufWeightSource>& source,
    const std::string& indexPath);

MtkPteRebuildResult RebuildMtkPteWeightsFromGguf(
    const std::string& strippedPtePath,
    const MtkGgufPteRecipe& recipe,
    std::shared_ptr<std::vector<uint8_t>> outputBuffer = nullptr);

// Drop only this mapping's resident source pages after Prefill. The Decode
// mapping of the same GGUF remains valid and can fault pages independently.
void DiscardMtkGgufSourcePages(
    const std::shared_ptr<MtkGgufWeightSource>& source);

// Reconstruct one complete MTK PTE without loading weights.bin or
// stripped.pte into a second full-size memory buffer. The returned storage must
// remain alive for as long as an ExecuTorch BufferDataLoader references it.
MtkPteRebuildResult RebuildMtkPteWeights(
    const std::string& strippedPtePath,
    const std::string& indexPath,
    const std::string& weightsPath,
    std::shared_ptr<std::vector<uint8_t>> outputBuffer = nullptr);

} // namespace example
