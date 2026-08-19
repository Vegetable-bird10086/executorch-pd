/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#include "MtkStageMajorPrefill.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

#include <executorch/runtime/platform/log.h>

#include "LlamaModelChunk.h"
#include "llm_helper/include/llm_types.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"

namespace example {
namespace {

using llm_helper::RotaryEmbeddingMasterLut;
using llm_helper::TokenEmbeddingLut;
using Clock = std::chrono::steady_clock;

double ElapsedMs(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct ProcessMemory {
  size_t rssBytes{0};
  size_t hwmBytes{0};
};

ProcessMemory ReadProcessMemory() {
  std::ifstream status("/proc/self/status");
  ProcessMemory memory;
  std::string key;
  while (status >> key) {
    if (key == "VmRSS:" || key == "VmHWM:") {
      size_t valueKb = 0;
      std::string unit;
      status >> valueKb >> unit;
      if (key == "VmRSS:") {
        memory.rssBytes = valueKb * 1024;
      } else {
        memory.hwmBytes = valueKb * 1024;
      }
    } else {
      std::string rest;
      std::getline(status, rest);
    }
  }
  return memory;
}

struct PromptBlock {
  std::vector<uint64_t> tokens;
  size_t leftPadding{0};
};

std::vector<PromptBlock> BuildPromptBlocks(
    const std::vector<uint64_t>& tokens,
    size_t batchSize) {
  if (tokens.empty() || batchSize == 0) {
    throw std::runtime_error("Stage-major Prefill requires non-empty tokens and batch");
  }
  std::vector<PromptBlock> blocks;
  size_t cursor = 0;
  size_t first = tokens.size() % batchSize;
  if (first == 0) {
    first = std::min(batchSize, tokens.size());
  }
  while (cursor < tokens.size()) {
    const size_t count = cursor == 0 ? first : batchSize;
    PromptBlock block;
    block.leftPadding = cursor == 0 ? batchSize - count : 0;
    block.tokens.assign(tokens.begin() + cursor, tokens.begin() + cursor + count);
    blocks.push_back(std::move(block));
    cursor += count;
  }
  return blocks;
}

class ThreeStageChunkPipeline {
 public:
  ThreeStageChunkPipeline(
      const LlamaModelOptions& options,
      const std::vector<MtkStrippedChunkPaths>& paths,
      const std::vector<std::shared_ptr<MtkGgufPteRecipe>>& ggufRecipes,
      size_t firstPipelineChunk,
      size_t batchSize,
      size_t cachesPerChunk,
      bool finalOutputLogits,
      const RotaryEmbeddingMasterLut* rotary,
      std::vector<MtkStageMajorChunkStats>* stats)
      : options_(options),
        paths_(paths),
        ggufRecipes_(ggufRecipes),
        firstPipelineChunk_(firstPipelineChunk),
        batchSize_(batchSize),
        cachesPerChunk_(cachesPerChunk),
        finalOutputLogits_(finalOutputLogits),
        rotary_(rotary),
        stats_(stats),
        rebuilt_(paths.size()),
        loaded_(paths.size()),
        rebuiltReady_(paths.size(), false),
        loadedReady_(paths.size(), false),
        rebuildPermit_(std::min(firstPipelineChunk + 1, paths.size() - 1)),
        loadPermit_(std::min(firstPipelineChunk, paths.size() - 1)) {
    rebuildWorker_ = std::thread([this]() { RebuildLoop(); });
    loadWorker_ = std::thread([this]() { LoadLoop(); });
    ET_LOG(
        Info,
        "MTK three-stage pipeline started: chunks=%zu first_pipeline_chunk=%zu "
        "rebuild_ahead=2 load_ahead=1",
        paths_.size(),
        firstPipelineChunk_);
  }

  ~ThreeStageChunkPipeline() {
    Stop();
  }

  void PermitForExecution(size_t chunkIndex) {
    const size_t last = paths_.size() - 1;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      rebuildPermit_ =
          std::max(rebuildPermit_, std::min(chunkIndex + 2, last));
      loadPermit_ = std::max(loadPermit_, std::min(chunkIndex + 1, last));
    }
    condition_.notify_all();
  }

  std::unique_ptr<LlamaModelChunk> TakeLoaded(size_t chunkIndex) {
    const auto waitStart = Clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return stop_ || workerError_ || loadedReady_[chunkIndex];
    });
    (*stats_)[chunkIndex].pipelineWaitMs += ElapsedMs(waitStart);
    RethrowWorkerErrorLocked();
    if (stop_ || !loadedReady_[chunkIndex]) {
      throw std::runtime_error("MTK three-stage pipeline stopped before load");
    }
    return std::move(loaded_[chunkIndex]);
  }

  void WaitUntilLoaded(size_t chunkIndex) {
    const auto waitStart = Clock::now();
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return stop_ || workerError_ || loadedReady_[chunkIndex];
    });
    (*stats_)[chunkIndex].pipelineWaitMs += ElapsedMs(waitStart);
    RethrowWorkerErrorLocked();
    if (stop_ || !loadedReady_[chunkIndex]) {
      throw std::runtime_error("MTK three-stage pipeline stopped before next load");
    }
  }

  void Stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    if (rebuildWorker_.joinable()) {
      rebuildWorker_.join();
    }
    if (loadWorker_.joinable()) {
      loadWorker_.join();
    }
  }

 private:
  void RethrowWorkerErrorLocked() const {
    if (workerError_) {
      std::rethrow_exception(workerError_);
    }
  }

  void RecordWorkerError() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!workerError_) {
        workerError_ = std::current_exception();
      }
      stop_ = true;
    }
    condition_.notify_all();
  }

  void RebuildLoop() {
    try {
      for (size_t index = firstPipelineChunk_; index < paths_.size(); ++index) {
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock, [&]() {
            return stop_ || index <= rebuildPermit_;
          });
          if (stop_) {
            return;
          }
        }
        auto result = ggufRecipes_.empty()
            ? RebuildMtkPteWeights(
                  paths_[index].strippedPte,
                  paths_[index].index,
                  paths_[index].weights)
            : RebuildMtkPteWeightsFromGguf(
                  paths_[index].strippedPte, *ggufRecipes_[index]);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          (*stats_)[index].rebuild = result.stats;
          rebuilt_[index].emplace(std::move(result));
          rebuiltReady_[index] = true;
        }
        condition_.notify_all();
      }
    } catch (...) {
      RecordWorkerError();
    }
  }

  void LoadLoop() {
    try {
      for (size_t index = firstPipelineChunk_; index < paths_.size(); ++index) {
        std::optional<MtkPteRebuildResult> rebuilt;
        {
          std::unique_lock<std::mutex> lock(mutex_);
          condition_.wait(lock, [&]() {
            return stop_ ||
                (rebuiltReady_[index] && index <= loadPermit_);
          });
          if (stop_) {
            return;
          }
          rebuilt = std::move(rebuilt_[index]);
          rebuilt_[index].reset();
        }

        const bool finalChunk = index + 1 == paths_.size();
        ModelPathMap modelPaths{
            {batchSize_, "mtk-rebuilt-chunk-" + std::to_string(index)}};
        auto chunk = std::make_unique<LlamaModelChunk>(
            modelPaths,
            options_,
            true,
            batchSize_,
            cachesPerChunk_,
            1,
            finalChunk
                ? (finalOutputLogits_ ? options_.logit_shard_count : 0)
                : 1,
            options_.window_size != 0,
            index,
            rotary_);
        chunk->SetModelBytes(rebuilt->pte);
        const auto loadStart = Clock::now();
        chunk->Initialize();
        const double loadMs = ElapsedMs(loadStart);
        const ProcessMemory memoryAfterLoad = ReadProcessMemory();
        rebuilt.reset();
        {
          std::lock_guard<std::mutex> lock(mutex_);
          (*stats_)[index].loadMs = loadMs;
          (*stats_)[index].rssAfterLoadBytes = memoryAfterLoad.rssBytes;
          loaded_[index] = std::move(chunk);
          loadedReady_[index] = true;
        }
        condition_.notify_all();
      }
    } catch (...) {
      RecordWorkerError();
    }
  }

 private:
  const LlamaModelOptions& options_;
  const std::vector<MtkStrippedChunkPaths>& paths_;
  const std::vector<std::shared_ptr<MtkGgufPteRecipe>>& ggufRecipes_;
  size_t firstPipelineChunk_;
  size_t batchSize_;
  size_t cachesPerChunk_;
  bool finalOutputLogits_;
  const RotaryEmbeddingMasterLut* rotary_;
  std::vector<MtkStageMajorChunkStats>* stats_;
  std::vector<std::optional<MtkPteRebuildResult>> rebuilt_;
  std::vector<std::unique_ptr<LlamaModelChunk>> loaded_;
  std::vector<bool> rebuiltReady_;
  std::vector<bool> loadedReady_;
  size_t rebuildPermit_;
  size_t loadPermit_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::exception_ptr workerError_;
  bool stop_{false};
  std::thread rebuildWorker_;
  std::thread loadWorker_;
};

} // namespace

struct MtkStageMajorPrefillSession::Impl {
  LlamaModelOptions options;
  std::string tokenEmbeddingPath;
  std::vector<MtkStrippedChunkPaths> chunks;
  std::string ggufWeightPath;
  std::shared_ptr<MtkGgufWeightSource> ggufSource;
  std::vector<std::shared_ptr<MtkGgufPteRecipe>> ggufRecipes;
  bool finalOutputLogits{false};
  bool enableThreeStagePipeline{true};
  bool enableAsyncRelease{false};
  bool persistentChunk0{true};
  const QnnKvAbi* qnnKvAbi{nullptr};
  std::once_flag prepareOnce;
  std::unique_ptr<RotaryEmbeddingMasterLut> rotary;
  std::unique_ptr<LlamaModelChunk> chunk0;
  MtkStageMajorChunkStats chunk0Stats;
  double rotaryMs{0.0};
};

MtkStageMajorPrefillSession::MtkStageMajorPrefillSession(
    const LlamaModelOptions& options,
    std::string tokenEmbeddingPath,
    std::vector<MtkStrippedChunkPaths> chunks,
    std::string ggufWeightPath,
    const bool finalOutputLogits,
    const bool enableThreeStagePipeline,
    const bool enableAsyncRelease,
    const bool persistentChunk0,
    const QnnKvAbi* qnnKvAbi)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = options;
  impl_->tokenEmbeddingPath = std::move(tokenEmbeddingPath);
  impl_->chunks = std::move(chunks);
  impl_->ggufWeightPath = std::move(ggufWeightPath);
  impl_->finalOutputLogits = finalOutputLogits;
  impl_->enableThreeStagePipeline = enableThreeStagePipeline;
  impl_->enableAsyncRelease = enableAsyncRelease;
  impl_->persistentChunk0 = persistentChunk0;
  impl_->qnnKvAbi = qnnKvAbi;
}

MtkStageMajorPrefillSession::~MtkStageMajorPrefillSession() {
  if (impl_ && impl_->chunk0) {
    impl_->chunk0->Release();
    impl_->chunk0.reset();
  }
}

void MtkStageMajorPrefillSession::Prepare() {
  std::call_once(impl_->prepareOnce, [&]() {
    const auto& options = impl_->options;
    const auto& chunks = impl_->chunks;
    if (chunks.empty() || options.num_layer % chunks.size() != 0) {
      throw std::runtime_error(
          "Stage-major chunks must evenly divide decoder layers");
    }
    const size_t headDim = options.head_dim
        ? options.head_dim
        : options.hidden_size / options.num_head;
    const size_t rotaryDim = headDim * options.partial_rotary_factor;
    impl_->rotary = std::make_unique<RotaryEmbeddingMasterLut>(
        options.rot_emb_type,
        options.max_token_length,
        rotaryDim,
        options.rot_emb_base);
    const auto rotaryStart = Clock::now();
    impl_->rotary->generate();
    impl_->rotaryMs = ElapsedMs(rotaryStart);
    if (!impl_->ggufWeightPath.empty()) {
      impl_->ggufSource = OpenMtkGgufWeightSource(impl_->ggufWeightPath);
      impl_->ggufRecipes.reserve(chunks.size());
      for (const auto& chunk : chunks) {
        impl_->ggufRecipes.push_back(
            PrepareMtkGgufPteRecipe(impl_->ggufSource, chunk.index));
      }
      ET_LOG(
          Info,
          "MTK GGUF rebuild source prepared: chunks=%zu source=%s",
          impl_->ggufRecipes.size(),
          impl_->ggufWeightPath.c_str());
    }
    if (!impl_->persistentChunk0) {
      return;
    }

    const size_t layersPerChunk = options.num_layer / chunks.size();
    const size_t cachesPerChunk = 2 * layersPerChunk;
    auto rebuilt = impl_->ggufRecipes.empty()
        ? RebuildMtkPteWeights(
              chunks.front().strippedPte,
              chunks.front().index,
              chunks.front().weights)
        : RebuildMtkPteWeightsFromGguf(
              chunks.front().strippedPte, *impl_->ggufRecipes.front());
    impl_->chunk0Stats.rebuild = rebuilt.stats;
    ModelPathMap modelPaths{{
        options.prompt_token_batch_size, "mtk-persistent-chunk-0"}};
    impl_->chunk0 = std::make_unique<LlamaModelChunk>(
        modelPaths,
        options,
        true,
        options.prompt_token_batch_size,
        cachesPerChunk,
        1,
        chunks.size() == 1
            ? (impl_->finalOutputLogits ? options.logit_shard_count : 0)
            : 1,
        options.window_size != 0,
        0,
        impl_->rotary.get());
    impl_->chunk0->SetModelBytes(rebuilt.pte);
    const auto loadStart = Clock::now();
    impl_->chunk0->Initialize();
    impl_->chunk0Stats.loadMs = ElapsedMs(loadStart);
    impl_->chunk0Stats.rssAfterLoadBytes = ReadProcessMemory().rssBytes;
    ET_LOG(
        Info,
        "MTK persistent chunk 0 prepared: rebuild_ms=%.3f load_ms=%.3f "
        "rotary_ms=%.3f rss_mib=%.2f",
        impl_->chunk0Stats.rebuild.total_ms,
        impl_->chunk0Stats.loadMs,
        impl_->rotaryMs,
        impl_->chunk0Stats.rssAfterLoadBytes / (1024.0 * 1024.0));
  });
}

MtkStageMajorPrefillResult MtkStageMajorPrefillSession::Run(
    const std::vector<uint64_t>& promptTokens) {
  Prepare();
  const auto& options = impl_->options;
  const auto& tokenEmbeddingPath = impl_->tokenEmbeddingPath;
  const auto& chunks = impl_->chunks;
  const bool finalOutputLogits = impl_->finalOutputLogits;
  const bool enableThreeStagePipeline = impl_->enableThreeStagePipeline;
  const bool enableAsyncRelease = impl_->enableAsyncRelease;
  const bool persistentChunk0 = impl_->persistentChunk0;
  const auto totalStart = Clock::now();
  if (chunks.empty() || options.num_layer % chunks.size() != 0) {
    throw std::runtime_error("Stage-major chunks must evenly divide decoder layers");
  }
  if (promptTokens.size() > options.cache_size) {
    throw std::runtime_error("Stage-major prompt exceeds compiled MTK cache length");
  }
  const size_t batchSize = options.prompt_token_batch_size;
  const auto blocks = BuildPromptBlocks(promptTokens, batchSize);
  const size_t inputTypeSize = llm_helper::getLLMTypeSize(options.model_input_type);
  const size_t blockBytes = batchSize * options.hidden_size * inputTypeSize;
  std::vector<uint8_t> hidden(blocks.size() * blockBytes);

  auto& rotary = *impl_->rotary;

  const size_t layersPerChunk = options.num_layer / chunks.size();
  const size_t cachesPerChunk = 2 * layersPerChunk;
  MtkStageMajorPrefillResult result;
  result.chunks.resize(chunks.size());
  result.rotaryMs = impl_->rotaryMs;
  if (persistentChunk0) {
    result.chunks[0] = impl_->chunk0Stats;
    impl_->chunk0->Reset();
  }

  std::unique_ptr<ThreeStageChunkPipeline> pipeline;
  if (enableThreeStagePipeline) {
    pipeline = std::make_unique<ThreeStageChunkPipeline>(
        options,
        chunks,
        impl_->ggufRecipes,
        persistentChunk0 ? 1 : 0,
        batchSize,
        cachesPerChunk,
        finalOutputLogits,
        &rotary,
        &result.chunks);
  }

  // Start chunk-0 preparation before the external embedding lookup. This
  // mirrors QNN's persistent-shard0 warmup and hides cold file/backend work
  // behind request-side input preparation where possible.
  const auto embeddingStart = Clock::now();
  TokenEmbeddingLut embedding(
      tokenEmbeddingPath, options.model_input_type, options.hidden_size);
  embedding.setDiscardAfterLookup(true);
  for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
    auto padded = blocks[blockIndex].tokens;
    padded.insert(padded.begin(), blocks[blockIndex].leftPadding, uint64_t{0});
    embedding.setOutput(hidden.data() + blockIndex * blockBytes, blockBytes);
    embedding.lookupEmbedding(padded);
  }
  result.embeddingMs = ElapsedMs(embeddingStart);

  std::optional<Clock::time_point> activePrefillStart;
  struct PendingReleaseResult {
    size_t chunkIndex{0};
    double releaseMs{0.0};
    ProcessMemory memory;
  };
  std::future<PendingReleaseResult> pendingRelease;
  auto logChunkStats = [&](const size_t chunkIndex) {
    const auto& stats = result.chunks[chunkIndex];
    ET_LOG(
        Info,
        "MTK stage-major chunk=%zu rebuild_ms=%.3f load_ms=%.3f "
        "pipeline_wait_ms=%.3f execute_ms=%.3f kv_pack_ms=%.3f release_ms=%.3f "
        "rss_load_mib=%.2f rss_execute_mib=%.2f rss_release_mib=%.2f hwm_mib=%.2f",
        chunkIndex,
        stats.rebuild.total_ms,
        stats.loadMs,
        stats.pipelineWaitMs,
        stats.executeMs,
        stats.kvPackMs,
        stats.releaseMs,
        stats.rssAfterLoadBytes / (1024.0 * 1024.0),
        stats.rssAfterExecuteBytes / (1024.0 * 1024.0),
        stats.rssAfterReleaseBytes / (1024.0 * 1024.0),
        stats.hwmAfterExecuteBytes / (1024.0 * 1024.0));
  };
  auto completePendingRelease = [&]() {
    if (!pendingRelease.valid()) {
      return;
    }
    const auto released = pendingRelease.get();
    auto& releasedStats = result.chunks[released.chunkIndex];
    releasedStats.releaseMs = released.releaseMs;
    releasedStats.rssAfterReleaseBytes = released.memory.rssBytes;
    logChunkStats(released.chunkIndex);
  };
  for (size_t chunkIndex = 0; chunkIndex < chunks.size(); ++chunkIndex) {
    auto& stats = result.chunks[chunkIndex];
    const bool finalChunk = chunkIndex + 1 == chunks.size();
    std::unique_ptr<LlamaModelChunk> chunk;
    LlamaModelChunk* activeChunk = nullptr;
    std::shared_ptr<std::vector<uint8_t>> serialRebuiltPte;
    if (persistentChunk0 && chunkIndex == 0) {
      activeChunk = impl_->chunk0.get();
      if (pipeline) {
        pipeline->PermitForExecution(chunkIndex);
      }
      ET_LOG(Info, "MTK stage-major using persistent chunk 0");
    } else if (pipeline) {
      chunk = pipeline->TakeLoaded(chunkIndex);
      activeChunk = chunk.get();
      pipeline->PermitForExecution(chunkIndex);
      ET_LOG(
          Info,
          "MTK three-stage: execute=%zu load=%zu rebuild=%zu",
          chunkIndex,
          std::min(chunkIndex + 1, chunks.size() - 1),
          std::min(chunkIndex + 2, chunks.size() - 1));
    } else {
      auto rebuilt = impl_->ggufRecipes.empty()
          ? RebuildMtkPteWeights(
                chunks[chunkIndex].strippedPte,
                chunks[chunkIndex].index,
                chunks[chunkIndex].weights)
          : RebuildMtkPteWeightsFromGguf(
                chunks[chunkIndex].strippedPte,
                *impl_->ggufRecipes[chunkIndex]);
      stats.rebuild = rebuilt.stats;
      serialRebuiltPte = rebuilt.pte;
      ModelPathMap modelPaths{
          {batchSize, "mtk-rebuilt-chunk-" + std::to_string(chunkIndex)}};
      chunk = std::make_unique<LlamaModelChunk>(
          modelPaths,
          options,
          true,
          batchSize,
          cachesPerChunk,
          1,
          finalChunk
              ? (finalOutputLogits ? options.logit_shard_count : 0)
              : 1,
          options.window_size != 0,
          chunkIndex,
          &rotary);
      chunk->SetModelBytes(serialRebuiltPte);
      const auto loadStart = Clock::now();
      chunk->Initialize();
      activeChunk = chunk.get();
      stats.loadMs = ElapsedMs(loadStart);
      stats.rssAfterLoadBytes = ReadProcessMemory().rssBytes;
    }
    if (chunkIndex == 0) {
      result.initialChunkWaitMs = stats.pipelineWaitMs;
      activePrefillStart = Clock::now();
    }

    for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
      BufferInfo input{
          hidden.data() + blockIndex * blockBytes,
          blockBytes,
          blockBytes,
      };
      activeChunk->SetInputBuffer(input);
      activeChunk->SetLeftPadding(blocks[blockIndex].leftPadding);
      const auto executeStart = Clock::now();
      activeChunk->Run();
      stats.executeMs += ElapsedMs(executeStart);

      if (!finalChunk) {
        const auto output = activeChunk->GetOutputBuffer();
        if (output.nbytesUsed != blockBytes) {
          throw std::runtime_error("MTK chunk hidden output size mismatch");
        }
        std::memcpy(
            hidden.data() + blockIndex * blockBytes,
            output.data,
            blockBytes);
      } else if (finalOutputLogits && blockIndex + 1 == blocks.size()) {
        const size_t outputTypeSize =
            llm_helper::getLLMTypeSize(options.model_output_type);
        result.lastLogits.clear();
        result.lastLogits.reserve(options.vocab_size * outputTypeSize);
        for (size_t shard = 0; shard < options.logit_shard_count; ++shard) {
          const auto output = activeChunk->GetOutputBuffer(shard);
          if (output.nbytesUsed % batchSize != 0) {
            throw std::runtime_error("MTK logit shard is not divisible by token batch");
          }
          const size_t tokenBytes = output.nbytesUsed / batchSize;
          const auto* last = static_cast<const char*>(output.data) +
              (batchSize - 1) * tokenBytes;
          result.lastLogits.insert(result.lastLogits.end(), last, last + tokenBytes);
        }
      }
    }

    const ProcessMemory memoryAfterExecute = ReadProcessMemory();
    stats.rssAfterExecuteBytes = memoryAfterExecute.rssBytes;
    stats.hwmAfterExecuteBytes = memoryAfterExecute.hwmBytes;

    if (chunkIndex == 0) {
      result.numKvHeads = activeChunk->GetNumKVHeads();
      result.headDim = activeChunk->GetCacheHeadDim();
      const size_t allKvValues =
          2 * options.num_layer * result.numKvHeads * promptTokens.size() *
          result.headDim;
      if (impl_->qnnKvAbi != nullptr) {
        if (impl_->qnnKvAbi->NumLayers() != options.num_layer ||
            impl_->qnnKvAbi->NumHeads() != result.numKvHeads ||
            impl_->qnnKvAbi->HeadDim() != result.headDim) {
          throw std::runtime_error("QNN KV ABI dimensions do not match MTK cache");
        }
        result.qnnU8Kv.resize(allKvValues);
      } else {
        result.canonicalKv.resize(allKvValues);
      }
    }
    const auto packStart = Clock::now();
    if (impl_->qnnKvAbi != nullptr) {
      std::vector<uint16_t> localCanonicalKv(
          2 * layersPerChunk * result.numKvHeads * promptTokens.size() *
          result.headDim);
      activeChunk->CopyCacheToLocalCanonicalFp16(
          promptTokens.size(), localCanonicalKv);
      impl_->qnnKvAbi->ConvertCanonicalFp16Layers(
          localCanonicalKv,
          chunkIndex * layersPerChunk,
          promptTokens.size(),
          result.qnnU8Kv,
          &result.qnnKvStats);
    } else {
      activeChunk->CopyCacheToCanonicalFp16(
          promptTokens.size(),
          chunkIndex * layersPerChunk,
          options.num_layer,
          result.canonicalKv);
    }
    stats.kvPackMs = ElapsedMs(packStart);

    // Match QNN's bounded three-stage lifetime: ensure the next context is
    // ready before destroying the current one, then keep at most current,
    // loaded-next, and rebuilt-next+1 resident.
    if (pipeline && chunkIndex + 1 < chunks.size()) {
      pipeline->WaitUntilLoaded(chunkIndex + 1);
    }

    if (persistentChunk0 && chunkIndex == 0) {
      stats.releaseMs = 0.0;
      stats.rssAfterReleaseBytes = ReadProcessMemory().rssBytes;
      logChunkStats(chunkIndex);
    } else if (pipeline && enableAsyncRelease) {
      // The previous cleanup has had the whole current NPU execution window
      // to finish. Collect it before submitting another job, bounding cleanup
      // concurrency and retained model storage to one chunk.
      completePendingRelease();
      pendingRelease = std::async(
          std::launch::async,
          [chunkIndex,
           releaseChunk = std::move(chunk),
           releaseBacking = std::move(serialRebuiltPte)]() mutable {
            const auto releaseStart = Clock::now();
            releaseChunk->Release();
            releaseChunk.reset();
            releaseBacking.reset();
            return PendingReleaseResult{
                chunkIndex,
                ElapsedMs(releaseStart),
                ReadProcessMemory(),
            };
          });
    } else {
      const auto releaseStart = Clock::now();
      chunk->Release();
      chunk.reset();
      serialRebuiltPte.reset();
      stats.releaseMs = ElapsedMs(releaseStart);
      stats.rssAfterReleaseBytes = ReadProcessMemory().rssBytes;
      logChunkStats(chunkIndex);
    }
    result.pureExecuteMs += stats.executeMs;
  }
  completePendingRelease();
  result.activePrefillMs = ElapsedMs(*activePrefillStart);
  if (pipeline) {
    pipeline->Stop();
  }
  // Prefill reconstruction is complete. Release this mapping's GGUF pages;
  // llama.cpp Decode owns a separate mapping of the same file.
  DiscardMtkGgufSourcePages(impl_->ggufSource);
  if (finalOutputLogits && result.lastLogits.size() !=
          options.vocab_size *
              llm_helper::getLLMTypeSize(options.model_output_type)) {
    throw std::runtime_error("MTK stage-major final logits size mismatch");
  }
  result.totalMs = ElapsedMs(totalStart);
  const ProcessMemory finalMemory = ReadProcessMemory();
  result.finalRssBytes = finalMemory.rssBytes;
  result.peakHwmBytes = finalMemory.hwmBytes;
  ET_LOG(
      Info,
      "MTK stage-major summary: chunks=%zu blocks=%zu cached_tokens=%zu "
      "three_stage=%d async_release=%d persistent_chunk0=%d "
      "embedding_ms=%.3f rotary_ms=%.3f initial_chunk_wait_ms=%.3f "
      "active_prefill_ms=%.3f "
      "pure_npu_execute_ms=%.3f total_ms=%.3f final_rss_mib=%.2f peak_hwm_mib=%.2f",
      chunks.size(),
      blocks.size(),
      promptTokens.size(),
      static_cast<int>(enableThreeStagePipeline),
      static_cast<int>(enableThreeStagePipeline && enableAsyncRelease),
      static_cast<int>(persistentChunk0),
      result.embeddingMs,
      result.rotaryMs,
      result.initialChunkWaitMs,
      result.activePrefillMs,
      result.pureExecuteMs,
      result.totalMs,
      result.finalRssBytes / (1024.0 * 1024.0),
      result.peakHwmBytes / (1024.0 * 1024.0));
  return result;
}

MtkStageMajorPrefillResult RunMtkStageMajorPrefill(
    const LlamaModelOptions& options,
    const std::string& tokenEmbeddingPath,
    const std::vector<MtkStrippedChunkPaths>& chunks,
    const std::vector<uint64_t>& promptTokens,
    const std::string& ggufWeightPath,
    const bool finalOutputLogits,
    const bool enableThreeStagePipeline,
    const bool enableAsyncRelease) {
  MtkStageMajorPrefillSession session(
      options,
      tokenEmbeddingPath,
      chunks,
      ggufWeightPath,
      finalOutputLogits,
      enableThreeStagePipeline,
      enableAsyncRelease,
      false,
      nullptr);
  return session.Run(promptTokens);
}

} // namespace example
