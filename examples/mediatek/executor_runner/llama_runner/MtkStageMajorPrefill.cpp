/*
 * Copyright (c) 2026
 * Licensed under the BSD-style license found in the ExecuTorch LICENSE file.
 */

#include "MtkStageMajorPrefill.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

#include <executorch/runtime/core/portable_type/half.h>
#include <executorch/runtime/platform/log.h>

#include <fcntl.h>
#include <unistd.h>

#include "LlamaModelChunk.h"
#include "llm_helper/include/llm_types.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"

namespace example {
namespace {

using llm_helper::RotaryEmbeddingMasterLut;
using llm_helper::TokenEmbeddingLut;
using Clock = std::chrono::steady_clock;

class StageEmbeddingLookup {
 public:
  StageEmbeddingLookup(
      const std::string& path,
      llm_helper::LLMType modelInputType,
      size_t hiddenSize)
      : hiddenSize_(hiddenSize), modelInputType_(modelInputType) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      throw std::runtime_error("Failed to open token embedding: " + path);
    }

    char magic[4]{};
    ReadAt(0, magic, sizeof(magic));
    if (std::memcmp(magic, "SEMB", sizeof(magic)) != 0) {
      close(fd_);
      fd_ = -1;
      raw_ = std::make_unique<TokenEmbeddingLut>(
          path, modelInputType_, hiddenSize_);
      raw_->setDiscardAfterLookup(true);
      ET_LOG(Info, "MTK PD embedding: legacy raw %s", path.c_str());
      return;
    }

    uint32_t version = 0;
    uint32_t quantized = 0;
    uint32_t dtype = 0;
    uint32_t ndim = 0;
    uint64_t nbytes = 0;
    ReadAt(4, &version, sizeof(version));
    ReadAt(8, &quantized, sizeof(quantized));
    ReadAt(12, &dtype, sizeof(dtype));
    ReadAt(16, &ndim, sizeof(ndim));
    ReadAt(20, &nbytes, sizeof(nbytes));
    if (version != 1 || quantized != 0 || dtype != 2 || ndim != 2) {
      throw std::runtime_error(
          "MTK PD requires an unquantized FP16 SEMB v1 embedding");
    }
    uint32_t shape[2]{};
    ReadAt(28, shape, sizeof(shape));
    vocabSize_ = shape[0];
    if (shape[1] != hiddenSize_ ||
        nbytes != static_cast<uint64_t>(vocabSize_) * hiddenSize_ *
                sizeof(uint16_t)) {
      throw std::runtime_error("FP16 SEMB embedding shape/size mismatch");
    }
    if (modelInputType_ != llm_helper::FP32) {
      throw std::runtime_error(
          "FP16 SEMB conversion currently requires an FP32 MTK graph input");
    }
    dataOffset_ = 36;
    fp16Row_.resize(hiddenSize_);
    ET_LOG(
        Info,
        "MTK PD embedding: QNN FP16 SEMB row-on-demand vocab=%zu hidden=%zu",
        vocabSize_,
        hiddenSize_);
  }

  ~StageEmbeddingLookup() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  void Lookup(
      const std::vector<uint64_t>& tokens,
      void* output,
      size_t outputBytes) {
    if (raw_) {
      raw_->setOutput(output, outputBytes);
      raw_->lookupEmbedding(tokens);
      return;
    }
    const size_t expectedBytes =
        tokens.size() * hiddenSize_ * sizeof(float);
    if (output == nullptr || outputBytes != expectedBytes) {
      throw std::runtime_error("Invalid MTK PD embedding output buffer");
    }
    auto* outputFp32 = static_cast<float*>(output);
    for (size_t row = 0; row < tokens.size(); ++row) {
      if (tokens[row] >= vocabSize_) {
        throw std::runtime_error("MTK PD embedding token is out of range");
      }
      const uint64_t offset =
          dataOffset_ + tokens[row] * hiddenSize_ * sizeof(uint16_t);
      ReadAt(offset, fp16Row_.data(), fp16Row_.size() * sizeof(uint16_t));
      for (size_t column = 0; column < hiddenSize_; ++column) {
        executorch::runtime::etensor::Half value;
        std::memcpy(&value, &fp16Row_[column], sizeof(value));
        outputFp32[row * hiddenSize_ + column] = static_cast<float>(value);
      }
    }
  }

 private:
  void ReadAt(uint64_t offset, void* output, size_t bytes) const {
    auto* cursor = static_cast<uint8_t*>(output);
    size_t remaining = bytes;
    while (remaining > 0) {
      const ssize_t readBytes = pread(fd_, cursor, remaining, offset);
      if (readBytes < 0 && errno == EINTR) {
        continue;
      }
      if (readBytes <= 0) {
        throw std::runtime_error("Failed to read token embedding data");
      }
      cursor += readBytes;
      offset += static_cast<uint64_t>(readBytes);
      remaining -= static_cast<size_t>(readBytes);
    }
  }

  size_t hiddenSize_{0};
  llm_helper::LLMType modelInputType_{llm_helper::INVALID};
  int fd_{-1};
  size_t vocabSize_{0};
  uint64_t dataOffset_{0};
  std::vector<uint16_t> fp16Row_;
  std::unique_ptr<TokenEmbeddingLut> raw_;
};

// In normal BufferDataLoader mode Program/Method keeps the reconstructed PTE
// backing alive until ModelChunk::Release(), requiring three slots for
// execute(i), load(i+1), and rebuild(i+2). The opt-in detachable loader copies
// the retained Program metadata and returns backing at load tail, reducing the
// hard limit to two slots.
class MtkPteRebuildBufferPool {
 public:
  explicit MtkPteRebuildBufferPool(size_t maxBuffers)
      : maxBuffers_(maxBuffers) {}

  std::shared_ptr<std::vector<uint8_t>> Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return cancelled_ || !idle_.empty() || allocationCount_ < maxBuffers_;
    });
    if (cancelled_) {
      throw std::runtime_error("MTK PTE rebuild buffer pool cancelled");
    }
    if (idle_.empty()) {
      ++allocationCount_;
      return std::make_shared<std::vector<uint8_t>>();
    }
    auto largest = std::max_element(
        idle_.begin(),
        idle_.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs->capacity() < rhs->capacity();
        });
    auto buffer = std::move(*largest);
    idle_.erase(largest);
    ++reuseCount_;
    return buffer;
  }

  void Release(std::shared_ptr<std::vector<uint8_t>> buffer) {
    if (!buffer) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      idle_.push_back(std::move(buffer));
    }
    condition_.notify_one();
  }

  void Cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
    condition_.notify_all();
  }

  size_t AllocationCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocationCount_;
  }

  size_t ReuseCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reuseCount_;
  }

  size_t IdleCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return idle_.size();
  }

  size_t IdleCapacityBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto& buffer : idle_) {
      total += buffer->capacity();
    }
    return total;
  }

  void Clear() {
    std::vector<std::shared_ptr<std::vector<uint8_t>>> released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released.swap(idle_);
    }
  }

 private:
  const size_t maxBuffers_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<std::shared_ptr<std::vector<uint8_t>>> idle_;
  bool cancelled_{false};
  size_t allocationCount_{0};
  size_t reuseCount_{0};
};

// Load runs one chunk ahead of execute, so two complete Neuron IO/AHWB sets
// are sufficient: execute(i) owns one and load(i+1) owns the other. Chunk 0 is
// persistent and intentionally separate; the structurally different final
// chunk is also not pooled.
class MtkNeuronIoSlotPool {
 public:
  explicit MtkNeuronIoSlotPool(size_t maxSlots) : maxSlots_(maxSlots) {}

  ModelIoBufferSet Acquire() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&]() {
      return cancelled_ || !idle_.empty() || createdSlots_ < maxSlots_;
    });
    if (cancelled_) {
      throw std::runtime_error("MTK Neuron IO pool cancelled");
    }
    if (!idle_.empty()) {
      ModelIoBufferSet slot = std::move(idle_.back());
      idle_.pop_back();
      ++reuseCount_;
      return slot;
    }
    ++createdSlots_;
    ++allocationCount_;
    return {};
  }

  void Release(ModelIoBufferSet slot) {
    if (slot.empty()) {
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      idle_.push_back(std::move(slot));
    }
    condition_.notify_one();
  }

  void Cancel() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      cancelled_ = true;
    }
    condition_.notify_all();
  }

  size_t AllocationCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocationCount_;
  }

  size_t ReuseCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return reuseCount_;
  }

  size_t IdleCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return idle_.size();
  }

  size_t IdleBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    std::unordered_set<void*> seen;
    for (const auto& slot : idle_) {
      const auto count = [&](const std::vector<BufferInfo>& infos) {
        for (const auto& info : infos) {
          if (info.data != nullptr && seen.insert(info.data).second) {
            total += info.nbytes;
          }
        }
      };
      count(slot.inputs);
      count(slot.outputs);
    }
    return total;
  }

  void Clear() {
    std::vector<ModelIoBufferSet> released;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      released.swap(idle_);
    }
    for (auto& slot : released) {
      ReleaseModelIoBufferSet(std::move(slot));
    }
  }

 private:
  const size_t maxSlots_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::vector<ModelIoBufferSet> idle_;
  bool cancelled_{false};
  size_t createdSlots_{0};
  size_t allocationCount_{0};
  size_t reuseCount_{0};
};

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
      std::shared_ptr<MtkPteRebuildBufferPool> rebuildBufferPool,
      std::shared_ptr<MtkNeuronIoSlotPool> ioSlotPool,
      bool detachPteBackingAfterLoad,
      std::vector<MtkStageMajorChunkStats>* stats)
      : options_(options),
        paths_(paths),
        ggufRecipes_(ggufRecipes),
        firstPipelineChunk_(firstPipelineChunk),
        batchSize_(batchSize),
        cachesPerChunk_(cachesPerChunk),
        finalOutputLogits_(finalOutputLogits),
        rotary_(rotary),
        rebuildBufferPool_(std::move(rebuildBufferPool)),
        ioSlotPool_(std::move(ioSlotPool)),
        detachPteBackingAfterLoad_(detachPteBackingAfterLoad),
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
    rebuildBufferPool_->Cancel();
    ioSlotPool_->Cancel();
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
                  paths_[index].weights,
                  rebuildBufferPool_->Acquire())
            : RebuildMtkPteWeightsFromGguf(
                  paths_[index].strippedPte,
                  *ggufRecipes_[index],
                  rebuildBufferPool_->Acquire());
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
        chunk->SetDetachPteBackingAfterLoad(detachPteBackingAfterLoad_);
        const bool pooledIo = !finalChunk;
        if (pooledIo) {
          auto ioSlot = ioSlotPool_->Acquire();
          if (!ioSlot.empty()) {
            chunk->SetIoBufferSet(std::move(ioSlot));
          }
        }
        const auto loadStart = Clock::now();
        chunk->Initialize();
        const double loadMs = ElapsedMs(loadStart);
        const ProcessMemory memoryAfterLoad = ReadProcessMemory();
        if (detachPteBackingAfterLoad_) {
          auto detached = chunk->DetachLoadedModelBytes();
          rebuilt->pte.reset();
          rebuildBufferPool_->Release(std::move(detached));
        }
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
  std::shared_ptr<MtkPteRebuildBufferPool> rebuildBufferPool_;
  std::shared_ptr<MtkNeuronIoSlotPool> ioSlotPool_;
  bool detachPteBackingAfterLoad_;
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
  bool detachPteBackingAfterLoad{false};
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
    std::shared_ptr<MtkGgufWeightSource> sharedGgufSource,
    const bool finalOutputLogits,
    const bool enableThreeStagePipeline,
    const bool enableAsyncRelease,
    const bool persistentChunk0,
    const bool detachPteBackingAfterLoad,
    const QnnKvAbi* qnnKvAbi)
    : impl_(std::make_unique<Impl>()) {
  impl_->options = options;
  impl_->tokenEmbeddingPath = std::move(tokenEmbeddingPath);
  impl_->chunks = std::move(chunks);
  impl_->ggufWeightPath = std::move(ggufWeightPath);
  impl_->ggufSource = std::move(sharedGgufSource);
  impl_->finalOutputLogits = finalOutputLogits;
  impl_->enableThreeStagePipeline = enableThreeStagePipeline;
  impl_->enableAsyncRelease = enableAsyncRelease;
  impl_->persistentChunk0 = persistentChunk0;
  impl_->detachPteBackingAfterLoad = detachPteBackingAfterLoad;
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
      if (!impl_->ggufSource) {
        impl_->ggufSource = OpenMtkGgufWeightSource(impl_->ggufWeightPath);
      }
      impl_->ggufRecipes.reserve(chunks.size());
      for (const auto& chunk : chunks) {
        impl_->ggufRecipes.push_back(
            PrepareMtkGgufPteRecipe(impl_->ggufSource, chunk.index));
      }
      ET_LOG(
          Info,
          "MTK GGUF rebuild source prepared: chunks=%zu source=%s ram_store=%d",
          impl_->ggufRecipes.size(),
          impl_->ggufWeightPath.c_str(),
          static_cast<int>(MtkGgufWeightSourceIsRamStore(impl_->ggufSource)));
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
    impl_->chunk0->SetDetachPteBackingAfterLoad(
        impl_->detachPteBackingAfterLoad);
    const auto loadStart = Clock::now();
    impl_->chunk0->Initialize();
    if (impl_->detachPteBackingAfterLoad) {
      auto detached = impl_->chunk0->DetachLoadedModelBytes();
      rebuilt.pte.reset();
      detached.reset();
    }
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
  const bool detachPteBackingAfterLoad = impl_->detachPteBackingAfterLoad;
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
  std::shared_ptr<MtkPteRebuildBufferPool> rebuildBufferPool;
  std::shared_ptr<MtkNeuronIoSlotPool> ioSlotPool;
  if (enableThreeStagePipeline) {
    rebuildBufferPool = std::make_shared<MtkPteRebuildBufferPool>(
        detachPteBackingAfterLoad ? 2 : 3);
    ioSlotPool = std::make_shared<MtkNeuronIoSlotPool>(2);
    pipeline = std::make_unique<ThreeStageChunkPipeline>(
        options,
        chunks,
        impl_->ggufRecipes,
        persistentChunk0 ? 1 : 0,
        batchSize,
        cachesPerChunk,
        finalOutputLogits,
        &rotary,
        rebuildBufferPool,
        ioSlotPool,
        detachPteBackingAfterLoad,
        &result.chunks);
  }

  // Start chunk-0 preparation before the external embedding lookup. This
  // mirrors QNN's persistent-shard0 warmup and hides cold file/backend work
  // behind request-side input preparation where possible.
  const auto embeddingStart = Clock::now();
  StageEmbeddingLookup embedding(
      tokenEmbeddingPath, options.model_input_type, options.hidden_size);
  for (size_t blockIndex = 0; blockIndex < blocks.size(); ++blockIndex) {
    auto padded = blocks[blockIndex].tokens;
    padded.insert(padded.begin(), blocks[blockIndex].leftPadding, uint64_t{0});
    embedding.Lookup(
        padded, hidden.data() + blockIndex * blockBytes, blockBytes);
  }
  result.embeddingMs = ElapsedMs(embeddingStart);

  std::optional<Clock::time_point> activePrefillStart;
  const bool parallelKvHandoff = impl_->qnnKvAbi != nullptr && pipeline != nullptr;
  struct PendingReleaseResult {
    size_t chunkIndex{0};
    double kvPackMs{0.0};
    double releaseMs{0.0};
    ProcessMemory memory;
    QnnKvAbiStats qnnStats;
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
    releasedStats.kvPackMs = released.kvPackMs;
    releasedStats.releaseMs = released.releaseMs;
    releasedStats.rssAfterReleaseBytes = released.memory.rssBytes;
    result.kvHandoffWorkerMs += released.kvPackMs;
    result.qnnKvStats.values += released.qnnStats.values;
    result.qnnKvStats.nonFinite += released.qnnStats.nonFinite;
    result.qnnKvStats.codeZero += released.qnnStats.codeZero;
    result.qnnKvStats.code255 += released.qnnStats.code255;
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
        result.qnnDirectKvOffsetBytes =
            promptTokens.size() * sizeof(uint64_t);
        result.qnnDirectHandoff.resize(
            result.qnnDirectKvOffsetBytes + allKvValues);
        std::memcpy(
            result.qnnDirectHandoff.data(),
            promptTokens.data(),
            result.qnnDirectKvOffsetBytes);
      } else {
        result.canonicalKv.resize(allKvValues);
      }
    }
    if (!parallelKvHandoff) {
      const auto packStart = Clock::now();
      if (impl_->qnnKvAbi != nullptr) {
        activeChunk->CopyCacheToQnnU8(
            promptTokens.size(),
            chunkIndex * layersPerChunk,
            *impl_->qnnKvAbi,
            result.qnnDirectHandoff.data() +
                result.qnnDirectKvOffsetBytes,
            result.qnnDirectHandoff.size() -
                result.qnnDirectKvOffsetBytes,
            &result.qnnKvStats);
      } else {
        activeChunk->CopyCacheToCanonicalFp16(
            promptTokens.size(),
            chunkIndex * layersPerChunk,
            options.num_layer,
            result.canonicalKv);
      }
      stats.kvPackMs = ElapsedMs(packStart);
    }

    // Match QNN's bounded three-stage lifetime: ensure the next context is
    // ready before destroying the current one, then keep at most current,
    // loaded-next, and rebuilt-next+1 resident.
    if (pipeline && chunkIndex + 1 < chunks.size()) {
      pipeline->WaitUntilLoaded(chunkIndex + 1);
    }

    if (parallelKvHandoff) {
      // Match QNN's incremental compactor: the previous chunk has the entire
      // next NPU execute window to finish packing before we reuse its IO slot.
      completePendingRelease();
      LlamaModelChunk* persistentPackChunk =
          persistentChunk0 && chunkIndex == 0 ? activeChunk : nullptr;
      pendingRelease = std::async(
          std::launch::async,
          [chunkIndex,
           persistentPackChunk,
           releaseChunk = std::move(chunk),
           releaseBacking = std::move(serialRebuiltPte),
           rebuildBufferPool,
           ioSlotPool,
           recycleIo = chunkIndex + 1 < chunks.size(),
           qnnKvAbi = impl_->qnnKvAbi,
           promptLength = promptTokens.size(),
           layerOffset = chunkIndex * layersPerChunk,
           handoffKv = result.qnnDirectHandoff.data() +
               result.qnnDirectKvOffsetBytes,
           handoffKvBytes = result.qnnDirectHandoff.size() -
               result.qnnDirectKvOffsetBytes]() mutable {
            LlamaModelChunk* packChunk =
                persistentPackChunk != nullptr
                ? persistentPackChunk
                : releaseChunk.get();
            QnnKvAbiStats localStats{};
            const auto packStart = Clock::now();
            packChunk->CopyCacheToQnnU8(
                promptLength,
                layerOffset,
                *qnnKvAbi,
                handoffKv,
                handoffKvBytes,
                &localStats);
            const double packMs = ElapsedMs(packStart);

            double releaseMs = 0.0;
            if (releaseChunk) {
              const auto releaseStart = Clock::now();
              ModelIoBufferSet ioSlot;
              if (recycleIo) {
                ioSlot = releaseChunk->ReleaseAndTakeIoBufferSet();
              } else {
                releaseChunk->Release();
              }
              if (rebuildBufferPool) {
                releaseBacking = releaseChunk->TakeReleasedModelBytes();
              }
              releaseChunk.reset();
              if (recycleIo) {
                ioSlotPool->Release(std::move(ioSlot));
              }
              if (rebuildBufferPool) {
                rebuildBufferPool->Release(std::move(releaseBacking));
              } else {
                releaseBacking.reset();
              }
              releaseMs = ElapsedMs(releaseStart);
            }
            return PendingReleaseResult{
                chunkIndex,
                packMs,
                releaseMs,
                ReadProcessMemory(),
                localStats,
            };
          });
    } else if (persistentChunk0 && chunkIndex == 0) {
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
           releaseBacking = std::move(serialRebuiltPte),
           rebuildBufferPool,
           ioSlotPool,
           recycleIo = chunkIndex + 1 < chunks.size()]() mutable {
            const auto releaseStart = Clock::now();
            ModelIoBufferSet ioSlot;
            if (recycleIo) {
              ioSlot = releaseChunk->ReleaseAndTakeIoBufferSet();
            } else {
              releaseChunk->Release();
            }
            if (rebuildBufferPool) {
              releaseBacking = releaseChunk->TakeReleasedModelBytes();
            }
            releaseChunk.reset();
            if (recycleIo) {
              ioSlotPool->Release(std::move(ioSlot));
            }
            if (rebuildBufferPool) {
              rebuildBufferPool->Release(std::move(releaseBacking));
            } else {
              releaseBacking.reset();
            }
            return PendingReleaseResult{
                chunkIndex,
                0.0,
                ElapsedMs(releaseStart),
                ReadProcessMemory(),
                {},
            };
          });
    } else {
      const auto releaseStart = Clock::now();
      ModelIoBufferSet ioSlot;
      const bool recycleIo = pipeline && chunkIndex + 1 < chunks.size();
      if (recycleIo) {
        ioSlot = chunk->ReleaseAndTakeIoBufferSet();
      } else {
        chunk->Release();
      }
      if (rebuildBufferPool) {
        serialRebuiltPte = chunk->TakeReleasedModelBytes();
      }
      chunk.reset();
      if (recycleIo) {
        ioSlotPool->Release(std::move(ioSlot));
      }
      if (rebuildBufferPool) {
        rebuildBufferPool->Release(std::move(serialRebuiltPte));
      } else {
        serialRebuiltPte.reset();
      }
      stats.releaseMs = ElapsedMs(releaseStart);
      stats.rssAfterReleaseBytes = ReadProcessMemory().rssBytes;
      logChunkStats(chunkIndex);
    }
    result.pureExecuteMs += stats.executeMs;
  }
  result.activePrefillMs = ElapsedMs(*activePrefillStart);
  const auto handoffBoundaryStart = Clock::now();
  completePendingRelease();
  result.kvHandoffBoundaryMs = ElapsedMs(handoffBoundaryStart);
  if (pipeline) {
    pipeline->Stop();
    ET_LOG(
        Info,
        "MTK PTE rebuild buffer pool complete: allocations=%zu reuses=%zu "
        "idle_slots=%zu idle_capacity_bytes=%zu",
        rebuildBufferPool->AllocationCount(),
        rebuildBufferPool->ReuseCount(),
        rebuildBufferPool->IdleCount(),
        rebuildBufferPool->IdleCapacityBytes());
    ET_LOG(
        Info,
        "MTK Neuron IO slot pool complete: allocations=%zu reuses=%zu "
        "idle_slots=%zu idle_bytes=%zu",
        ioSlotPool->AllocationCount(),
        ioSlotPool->ReuseCount(),
        ioSlotPool->IdleCount(),
        ioSlotPool->IdleBytes());
    // The pool is a prefill-only workspace. Drop all idle PTE backing before
    // GGUF page discard, final memory accounting, and llama.cpp decode.
    pipeline.reset();
    rebuildBufferPool->Clear();
    ioSlotPool->Clear();
  }
  // File-mmap fallback may discard pages here. The joint-PD RAM store is shared
  // by Prefill and Decode, so DiscardMtkGgufSourcePages intentionally no-ops.
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
      "parallel_kv_handoff=%d "
      "embedding_ms=%.3f rotary_ms=%.3f initial_chunk_wait_ms=%.3f "
      "active_prefill_ms=%.3f kv_worker_ms=%.3f kv_boundary_ms=%.3f "
      "pure_npu_execute_ms=%.3f total_ms=%.3f final_rss_mib=%.2f peak_hwm_mib=%.2f",
      chunks.size(),
      blocks.size(),
      promptTokens.size(),
      static_cast<int>(enableThreeStagePipeline),
      static_cast<int>(enableThreeStagePipeline && enableAsyncRelease),
      static_cast<int>(persistentChunk0),
      static_cast<int>(parallelKvHandoff),
      result.embeddingMs,
      result.rotaryMs,
      result.initialChunkWaitMs,
      result.activePrefillMs,
      result.kvHandoffWorkerMs,
      result.kvHandoffBoundaryMs,
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
      nullptr,
      finalOutputLogits,
      enableThreeStagePipeline,
      enableAsyncRelease,
      false,
      false,
      nullptr);
  return session.Run(promptTokens);
}

} // namespace example
