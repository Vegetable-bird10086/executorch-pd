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

#include "LlamaConfig.h"
#include "MtkPteWeightRebuilder.h"
#include "QnnKvAbi.h"

namespace example {

struct MtkStrippedChunkPaths {
  std::string strippedPte;
  std::string index;
  std::string weights;
};

struct MtkStageMajorChunkStats {
  MtkPteRebuildStats rebuild;
  double loadMs{0.0};
  double pipelineWaitMs{0.0};
  double executeMs{0.0};
  double kvPackMs{0.0};
  double releaseMs{0.0};
  size_t rssAfterLoadBytes{0};
  size_t rssAfterExecuteBytes{0};
  size_t rssAfterReleaseBytes{0};
  size_t hwmAfterExecuteBytes{0};
};

struct MtkStageMajorPrefillResult {
  std::vector<char> lastLogits;
  std::vector<uint16_t> canonicalKv;
  std::vector<uint8_t> qnnU8Kv;
  QnnKvAbiStats qnnKvStats;
  std::vector<MtkStageMajorChunkStats> chunks;
  size_t numKvHeads{0};
  size_t headDim{0};
  double embeddingMs{0.0};
  double rotaryMs{0.0};
  double initialChunkWaitMs{0.0};
  double activePrefillMs{0.0};
  double totalMs{0.0};
  double pureExecuteMs{0.0};
  size_t finalRssBytes{0};
  size_t peakHwmBytes{0};
};

// Request-session owner for the low-memory Prefill path. Prepare() may run on
// a background thread while tokenizer/GGUF initialization proceeds. Only
// chunk 0 remains resident between requests; the per-request pipeline still
// bounds the remaining live state to current, loaded-next, and rebuilt-next.
class MtkStageMajorPrefillSession {
 public:
  MtkStageMajorPrefillSession(
      const LlamaModelOptions& options,
      std::string tokenEmbeddingPath,
      std::vector<MtkStrippedChunkPaths> chunks,
      std::string ggufWeightPath,
      bool finalOutputLogits,
      bool enableThreeStagePipeline,
      bool enableAsyncRelease,
      bool persistentChunk0,
      const QnnKvAbi* qnnKvAbi = nullptr);
  ~MtkStageMajorPrefillSession();

  MtkStageMajorPrefillSession(const MtkStageMajorPrefillSession&) = delete;
  MtkStageMajorPrefillSession& operator=(
      const MtkStageMajorPrefillSession&) = delete;

  // Idempotent. Safe to launch with std::async before request tokenization or
  // llama.cpp Decode model initialization.
  void Prepare();

  MtkStageMajorPrefillResult Run(
      const std::vector<uint64_t>& promptTokens);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Executes one reconstructed layer chunk through every prompt block before
// releasing it. Hidden states are staged between chunks, so only one MTK
// model and one chunk-local KV cache need to be resident at a time.
MtkStageMajorPrefillResult RunMtkStageMajorPrefill(
    const LlamaModelOptions& options,
    const std::string& tokenEmbeddingPath,
    const std::vector<MtkStrippedChunkPaths>& chunks,
    const std::vector<uint64_t>& promptTokens,
    const std::string& ggufWeightPath,
    bool finalOutputLogits,
    bool enableThreeStagePipeline,
    bool enableAsyncRelease);

} // namespace example
