/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <numeric>
#include <fstream>
#include <cmath>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <string>
#include <unordered_map>
#include <vector>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/evalue_util/print_evalue.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/log.h>
#include <executorch/runtime/platform/profiler.h>
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/core/portable_type/half.h>

#include "LlamaConfig.h"
#include "IncrementalKvCache.h"
#include "LlamaModelChunk.h"
#include "Utils.h"
#include "llm_helper/include/llm_types.h"

#include "llm_helper/include/mask_builder.h"
#include "llm_helper/include/rotary_embedding.h"

namespace example {

inline std::vector<size_t> getIndexRange(
    const size_t startIndex,
    const size_t count) {
  std::vector<size_t> indexes(count);
  size_t counter = startIndex;
  for (auto& idx : indexes) {
    idx = counter++;
  }
  return indexes;
}

LlamaModelChunk::LlamaModelChunk(
    const ModelPathMap& modelPathMap,
    const LlamaModelOptions& modelOptions,
    const bool useSharedWeights,
    const size_t initBatchSize,
    const size_t numCache,
    const size_t numRotEmbInputs,
    const size_t logitShardCount,
    const bool enableSWA,
    const size_t chunkIndex,
    const RotaryEmbeddingMasterLut* rotEmbMasterLut)
    : ModelChunk(modelPathMap, initBatchSize),
      kIsSharedWeightsUsed(useSharedWeights),
      kMaxTokenLength(modelOptions.max_token_length),
      kCacheLength(modelOptions.cache_size),
      kMaskType(modelOptions.mask_type),
      kRotEmbMasterLut(rotEmbMasterLut),
      kCacheType(modelOptions.cache_type),
      kWindowSize(modelOptions.window_size),

      kRotEmbInputCount(numRotEmbInputs),
      kCacheCount(numCache),
      kLogitShardCount(logitShardCount),
      kLayerDebugOutputCount(
          modelOptions.layer_debug_chunk_index < 0 ||
                  static_cast<size_t>(modelOptions.layer_debug_chunk_index) ==
                      chunkIndex
              ? modelOptions.layer_debug_output_count
              : 0),
      enableSWA(enableSWA),
      kCacheTypeSize(llm_helper::getLLMTypeSize(kCacheType)),
      mChunkIndex(chunkIndex) {}

LlamaModelChunk::~LlamaModelChunk() {}

std::string LlamaModelChunk::SelectMethod(
    const std::vector<std::string>& methodNames) const {
  const size_t curTokenSize = GetModelId();
  for (const auto& methodName : methodNames) {
    const auto matches = utils::extract_substr(methodName, "([0-9]+)t[0-9]+c");
    if (matches.empty()) {
      continue;
    }
    ET_CHECK_MSG(
        matches.size() == 2, "Invalid method name: %s", methodName.c_str());
    // Extract the first match group as token size
    const size_t methodTokenSize =
        static_cast<size_t>(std::atol(matches[1].c_str()));
    if (curTokenSize == methodTokenSize) {
      ET_LOG(
          Debug,
          "Selected method \"%s\" for token size %zu",
          methodName.c_str(),
          curTokenSize);
      return methodName;
    }
  }
  ET_LOG(
      Error,
      "Unable to find suitable method, fallback to use the first method.");
  return {};
}

void LlamaModelChunk::Initialize() {
  LoadModels();
  GetModelIoInfo();
  defineIOs();
  CheckIoCount();
  PrepareCacheIOs();
  AllocateIoBuffers();
  InitMaskBuilder();
  InitCache();

  SetBackendInputs();
  SetBackendOutputs();
  mIsInitialized = true;
}

void LlamaModelChunk::defineIOs() {
  // Inputs
  defineInput(IOKind::Embedding);
  defineInput(IOKind::Mask);
  defineInput(IOKind::SWAMask, enableSWA);
  defineInput(IOKind::RotEmb, kRotEmbInputCount);
  defineInput(IOKind::KVCache, kCacheCount);
  // Outputs
  defineOutput(IOKind::Logits, kLogitShardCount);
  defineOutput(IOKind::KVCache, kCacheCount);
  defineOutput(IOKind::LayerDebug, kLayerDebugOutputCount);
}

void LlamaModelChunk::defineInput(const IOKind kind, const size_t count) {
  ET_CHECK_MSG(
      !hasInput(kind),
      "Input kind has already been defined: %d",
      static_cast<int>(kind));
  const auto startIdx = mExpectedNumInputs;
  std::vector<size_t> indexes(count);
  std::iota(indexes.begin(), indexes.end(), startIdx);
  mInputIndexes[kind] = std::move(indexes);
  mExpectedNumInputs += count;
}

void LlamaModelChunk::defineOutput(const IOKind kind, const size_t count) {
  ET_CHECK_MSG(
      !hasOutput(kind),
      "Output kind has already been defined: %d",
      static_cast<int>(kind));
  const auto startIdx = mExpectedNumOutputs;
  std::vector<size_t> indexes(count);
  std::iota(indexes.begin(), indexes.end(), startIdx);
  mOutputIndexes[kind] = std::move(indexes);
  mExpectedNumOutputs += count;
}

bool LlamaModelChunk::hasInput(const IOKind kind) const {
  return (mInputIndexes.find(kind) != mInputIndexes.end()) &&
      !mInputIndexes.at(kind).empty();
}

bool LlamaModelChunk::hasOutput(const IOKind kind) const {
  return (mOutputIndexes.find(kind) != mOutputIndexes.end()) &&
      !mOutputIndexes.at(kind).empty();
}

const std::vector<size_t>& LlamaModelChunk::getInputIndexes(
    const IOKind kind) const {
  ET_CHECK_MSG(
      hasInput(kind), "Check failed for input kind %d", static_cast<int>(kind));
  return mInputIndexes.at(kind);
}

const std::vector<size_t>& LlamaModelChunk::getOutputIndexes(
    const IOKind kind) const {
  ET_CHECK_MSG(
      hasOutput(kind),
      "Check failed for output kind %d",
      static_cast<int>(kind));
  return mOutputIndexes.at(kind);
}

size_t LlamaModelChunk::getInputIndex(const IOKind kind, const size_t pos)
    const {
  const auto& inputIndexes = getInputIndexes(kind);
  ET_CHECK_MSG(
      pos < inputIndexes.size(), "getInputIndex(): Index out of range");
  return inputIndexes[pos];
}

size_t LlamaModelChunk::getOutputIndex(const IOKind kind, const size_t pos)
    const {
  const auto& outputIndexes = getOutputIndexes(kind);
  ET_CHECK_MSG(
      pos < outputIndexes.size(), "getOutputIndex(): Index out of range");
  return outputIndexes[pos];
}

size_t LlamaModelChunk::getNumInputsFor(const IOKind kind) const {
  if (!hasInput(kind)) {
    return 0;
  }
  return mInputIndexes.at(kind).size();
}

size_t LlamaModelChunk::getNumOutputsFor(const IOKind kind) const {
  if (!hasOutput(kind)) {
    return 0;
  }
  return mOutputIndexes.at(kind).size();
}

void LlamaModelChunk::CheckIoCount() {
  const auto& method = GetModelMethod();
  const size_t modelInputCount = method.inputs_size();
  const size_t modelOutputCount = method.outputs_size();
  ET_CHECK_MSG(
      modelInputCount == mExpectedNumInputs,
      "Number of inputs does not match (expected %zu but got %zu).",
      mExpectedNumInputs,
      modelInputCount);
  ET_CHECK_MSG(
      modelOutputCount == mExpectedNumOutputs,
      "Number of outputs does not match (expected %zu but got %zu).",
      mExpectedNumOutputs,
      modelOutputCount);
}

bool LlamaModelChunk::HotSwapModel(const size_t tokenBatchSize) {
  const auto status = ModelChunk::HotSwapModel(tokenBatchSize);

  // Force rebuild mask because different batch size values will produce
  // different mask shapes.
  mMaskBuilder->markMaskDirty();

  // Update mask size
  const auto newMaskSizeBytes =
      mInputBufferInfos[getInputIndex(IOKind::Mask)].nbytesUsed;
  mMaskBuilder->updateMaskSize(newMaskSizeBytes);

  return status;
}

void LlamaModelChunk::Reset() {
  mCurrentPadSize = 0;
  mCurrentTokenIndex = 0;
  InitCache(); // Reset cache to zeros
}

void LlamaModelChunk::SetLeftPadding(const size_t leftPadSize) {
  mCurrentPadSize = leftPadSize;
  mPaddingMode = PaddingMode::LEFT;

  // Notify mask builder about padding
  mMaskBuilder->notifyLeftPadding(leftPadSize);
}

void LlamaModelChunk::SetRightPadding(const size_t rightPadSize) {
  mCurrentPadSize = rightPadSize;
  mPaddingMode = PaddingMode::RIGHT;

  // Notify mask builder about padding
  mMaskBuilder->notifyRightPadding(rightPadSize);
}

size_t LlamaModelChunk::GetLeftPadding() const {
  return (mPaddingMode == PaddingMode::LEFT) ? mCurrentPadSize : 0;
}

size_t LlamaModelChunk::GetRightPadding() const {
  return (mPaddingMode == PaddingMode::RIGHT) ? mCurrentPadSize : 0;
}

void LlamaModelChunk::PaddingPostprocess() {
  if (mCurrentPadSize == 0) {
    return;
  }

  if (mPaddingMode == PaddingMode::RIGHT) {
    RightPaddingCachePostprocess();
  } else if (mPaddingMode == PaddingMode::LEFT) {
    LeftPaddingCachePostprocess();
  }
}

void LlamaModelChunk::LeftPaddingCachePostprocess() {
  PadHandoffCache(mCurrentPadSize, mCurrentTokenIndex + mTokenBatchSize, true);
  // NOTE: This part might not actually be needed

  // Stride size is same across caches
  const size_t strideSizeBytes = GetCacheStrideSize();
  const size_t rowSize = kCacheLength * strideSizeBytes;

  const size_t numRows = GetCacheNumRows();

  const size_t offset = (kCacheLength - mTokenBatchSize) * strideSizeBytes;
  const size_t zeroCount = mCurrentPadSize * strideSizeBytes;

  // Fill padded sections with zeros
  for (const auto cacheInputIdx : getInputIndexes(IOKind::KVCache)) {
    auto cacheBuffer =
        reinterpret_cast<char*>(mInputBufferInfos[cacheInputIdx].data);
    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
      // cacheBufRow points at the start of row
      auto cacheBufRow = cacheBuffer + rowIdx * rowSize;
      std::memset(cacheBufRow + offset, 0, zeroCount);
    }
  }
}

void LlamaModelChunk::RightPaddingCachePostprocess() {
  // NOTE: AdvanceTokenIndex() haven't been called for this inference step yet.
  const size_t numSeenToken = mCurrentTokenIndex + mTokenBatchSize;
  RollbackCache(mCurrentPadSize, numSeenToken);
}

void LlamaModelChunk::RollbackCache(
    const size_t rollbackTokCount,
    const size_t numSeenToken) {
  if (rollbackTokCount == 0) {
    return; // do nothing
  }

  PadHandoffCache(rollbackTokCount, numSeenToken, false);
  const size_t numSeenTokenAlive = std::min(numSeenToken, kCacheLength);
  const size_t firstNonEmptyIdx = kCacheLength - numSeenTokenAlive;
  const size_t preserveTokCount = (numSeenTokenAlive > rollbackTokCount)
      ? numSeenTokenAlive - rollbackTokCount
      : 0;

  if (!preserveTokCount) {
    // Clear cache to zeros
    InitCache();
    return;
  }

  const size_t strideSizeBytes = GetCacheStrideSize();
  const size_t rowSize = kCacheLength * strideSizeBytes;
  const size_t numRows = GetCacheNumRows();

  // Shift right and truncate rollbackTokCount, then fill left with zeros
  for (const auto cacheInputIdx : getInputIndexes(IOKind::KVCache)) {
    auto cacheBuffer =
        reinterpret_cast<char*>(mInputBufferInfos[cacheInputIdx].data);

    for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
      // Get the addr pointing to the start of row
      auto cacheBufRow = cacheBuffer + rowIdx * rowSize;

      // Move right for the section to be preserved
      const size_t dstOffset =
          strideSizeBytes * (firstNonEmptyIdx + rollbackTokCount);
      const size_t srcOffset = strideSizeBytes * firstNonEmptyIdx;
      const size_t preserveSize = strideSizeBytes * preserveTokCount;
      ET_DCHECK(dstOffset + preserveSize <= rowSize);
      std::memmove(
          cacheBufRow + dstOffset, cacheBufRow + srcOffset, preserveSize);

      // Then fill zeros to the section being moved out
      const size_t offset = firstNonEmptyIdx * strideSizeBytes;
      const size_t zeroCount = rollbackTokCount * strideSizeBytes;
      std::memset(cacheBufRow + offset, 0, zeroCount);
    }
  }
}

void LlamaModelChunk::UpdatePosEmbAndMask(const size_t numInputToken) {
  if (mCurrentTokenIndex + numInputToken > kMaxTokenLength) {
    ET_LOG(
        Fatal,
        "Attempting to generate tokens exceeding the supported max token length (%zu)",
        kMaxTokenLength);
  }
  if (mCurrentTokenIndex > 0 && GetLeftPadding() > 0) {
    ET_LOG(Fatal, "Left-padding is only allowed in the first prompt pass.");
  }
  auto isMaskUpdatable = mMaskBuilder->getMaskUpdateStatus();
  if (enableSWA) {
    const auto& swaMaskBufferInfo =
        mInputBufferInfos[getInputIndex(IOKind::SWAMask)];
    const auto swaMaskBuffer = swaMaskBufferInfo.data;
    const auto swaMaskSizeBytes = swaMaskBufferInfo.nbytesUsed;
    mMaskBuilder->setMaskBuffer(swaMaskBuffer, swaMaskSizeBytes);
    mMaskBuilder->enableSlidingWindow(kWindowSize);
    mMaskBuilder->buildMask(mTokenBatchSize, mCurrentTokenIndex);
  }
  // Pass same isMaskUpdatable to both mask
  mMaskBuilder->setIsMaskUpdatable(isMaskUpdatable);
  const auto& maskBufferInfo = mInputBufferInfos[getInputIndex(IOKind::Mask)];
  const auto maskBuffer = maskBufferInfo.data;
  const auto maskSizeBytes = maskBufferInfo.nbytesUsed;
  mMaskBuilder->setMaskBuffer(maskBuffer, maskSizeBytes);
  mMaskBuilder->disableSlidingWindow();
  mMaskBuilder->updateMask(mTokenBatchSize, mCurrentTokenIndex, numInputToken);

  mMaskBuilder->resetPadLength();
  SetPosEmbed(mCurrentTokenIndex);
}

void LlamaModelChunk::AdvanceTokenIndex() {
  // Exclude padded tokens
  const auto numValidInputToken = mTokenBatchSize - mCurrentPadSize;
  mCurrentTokenIndex += numValidInputToken;

  // Reset padding size
  mCurrentPadSize = 0;
}

size_t LlamaModelChunk::GetTokenIndex() const {
  return mCurrentTokenIndex;
}

size_t LlamaModelChunk::GetCacheLayerCount() const {
  return kCacheCount / 2;
}

size_t LlamaModelChunk::GetNumKVHeads() const {
  ET_CHECK_MSG(mCacheShape.size() == 4, "Expected a 4D KV cache");
  return static_cast<size_t>(mCacheShape[1]);
}

size_t LlamaModelChunk::GetCacheHeadDim() const {
  ET_CHECK_MSG(mCacheShape.size() == 4, "Expected a 4D KV cache");
  return static_cast<size_t>(mCacheShape[3]);
}

size_t LlamaModelChunk::GetCacheLength() const {
  return kCacheLength;
}

void LlamaModelChunk::CopyCacheToCanonicalFp16(
    const size_t validTokenCount,
    const size_t globalLayerOffset,
    const size_t totalLayers,
    std::vector<uint16_t>& destination) {
  ET_CHECK_MSG(mCacheShape.size() == 4, "Expected a 4D KV cache");
  ET_CHECK_MSG(mCacheShape[0] == 1, "PD export only supports cache batch 1");
  ET_CHECK_MSG(
      validTokenCount <= kCacheLength,
      "PD prompt length %zu exceeds MTK cache length %zu",
      validTokenCount,
      kCacheLength);
  ET_CHECK_MSG(
      kCacheType == LLMType::FP32 || kCacheType == LLMType::FP16,
      "PD export supports only FP32/FP16 MTK caches, got %s",
      llm_helper::getLLMTypeName(kCacheType));

  const size_t localLayers = GetCacheLayerCount();
  const size_t numKVHeads = GetNumKVHeads();
  const size_t headDim = GetCacheHeadDim();
  const size_t valuesPerKind =
      totalLayers * numKVHeads * validTokenCount * headDim;
  ET_CHECK_MSG(
      destination.size() == valuesPerKind * 2,
      "Canonical KV destination has %zu values, expected %zu",
      destination.size(),
      valuesPerKind * 2);
  ET_CHECK_MSG(
      globalLayerOffset + localLayers <= totalLayers,
      "Chunk layer range [%zu,%zu) exceeds %zu layers",
      globalLayerOffset,
      globalLayerOffset + localLayers,
      totalLayers);

  const size_t sourceTokenOffset = kCacheLength - validTokenCount;
  for (size_t kind = 0; kind < 2; ++kind) {
    for (size_t localLayer = 0; localLayer < localLayers; ++localLayer) {
      const auto cacheInput = getInputIndex(
          IOKind::KVCache, kind * localLayers + localLayer);
      const auto source = GetHandoffCache(kind * localLayers + localLayer);
      const size_t expectedSourceBytes =
          numKVHeads * kCacheLength * headDim * (mSharedKvIo ? sizeof(int16_t) : (mQuantizedKvIo ? sizeof(float) : kCacheTypeSize));
      ET_CHECK_MSG(
          source.nbytes >= expectedSourceBytes,
          "KV cache input %zu has %zu bytes, expected at least %zu",
          cacheInput,
          source.nbytes,
          expectedSourceBytes);

      const size_t globalLayer = globalLayerOffset + localLayer;
      for (size_t head = 0; head < numKVHeads; ++head) {
        const size_t sourceOffset =
            (head * kCacheLength + sourceTokenOffset) * headDim;
        const size_t destinationOffset =
            kind * valuesPerKind +
            ((globalLayer * numKVHeads + head) * validTokenCount * headDim);
        const size_t valueCount = validTokenCount * headDim;
        auto* output = destination.data() + destinationOffset;
        if (mSharedKvIo) {
          const auto* input = static_cast<const int16_t*>(source.data) + sourceOffset;
          const float scale = mKvIoScales.at(mTokenBatchSize).at(kind * localLayers + localLayer).first;
          for (size_t i=0;i<valueCount;++i)
            output[i] = executorch::runtime::etensor::internal::fp16_ieee_from_fp32_value(input[i]*scale);
        } else if (kCacheType == LLMType::FP16) {
          const auto* input =
              static_cast<const uint16_t*>(source.data) + sourceOffset;
          std::copy(input, input + valueCount, output);
        } else {
          const auto* input =
              static_cast<const float*>(source.data) + sourceOffset;
          for (size_t i = 0; i < valueCount; ++i) {
            output[i] = executorch::runtime::etensor::internal::
                fp16_ieee_from_fp32_value(input[i]);
          }
        }
      }
    }
  }
}

void LlamaModelChunk::CopyCacheToLocalCanonicalFp16(
    const size_t validTokenCount,
    std::vector<uint16_t>& destination) {
  const size_t localLayers = GetCacheLayerCount();
  const size_t numKVHeads = GetNumKVHeads();
  const size_t headDim = GetCacheHeadDim();
  const size_t valuesPerKind =
      localLayers * numKVHeads * validTokenCount * headDim;
  ET_CHECK_MSG(
      destination.size() == valuesPerKind * 2,
      "Local canonical KV destination has %zu values, expected %zu",
      destination.size(),
      valuesPerKind * 2);

  // Reuse the fully validated canonical copier. With totalLayers equal to the
  // local layer count and offset zero, its output is exactly chunk-local.
  CopyCacheToCanonicalFp16(
      validTokenCount, 0, localLayers, destination);
}

void LlamaModelChunk::CopyCacheToQnnU8(
    const size_t validTokenCount,
    const size_t globalLayerOffset,
    const QnnKvAbi& abi,
    uint8_t* output,
    const size_t outputBytes,
    QnnKvAbiStats* stats) {
  ET_CHECK_MSG(mCacheShape.size() == 4, "Expected a 4D KV cache");
  ET_CHECK_MSG(mCacheShape[0] == 1, "PD export only supports cache batch 1");
  ET_CHECK_MSG(
      kCacheType == LLMType::FP32 || kCacheType == LLMType::FP16,
      "Direct QNN KV handoff supports only FP32/FP16 MTK caches");
  const size_t localLayers = GetCacheLayerCount();
  const size_t numKVHeads = GetNumKVHeads();
  const size_t headDim = GetCacheHeadDim();
  ET_CHECK_MSG(
      abi.NumHeads() == numKVHeads && abi.HeadDim() == headDim &&
          globalLayerOffset + localLayers <= abi.NumLayers(),
      "MTK cache dimensions do not match QNN KV ABI");
  ET_CHECK_MSG(
      outputBytes ==
          2 * abi.NumLayers() * numKVHeads * validTokenCount * headDim,
      "Direct QNN KV handoff output size mismatch");
  const size_t sourceFirstToken = kCacheLength - validTokenCount;
  for (size_t kind = 0; kind < 2; ++kind) {
    for (size_t localLayer = 0; localLayer < localLayers; ++localLayer) {
      const auto cacheInput = getInputIndex(
          IOKind::KVCache, kind * localLayers + localLayer);
      const auto source = GetHandoffCache(kind * localLayers + localLayer);
      abi.ConvertCacheLayer(
          source.data,
          kCacheType == LLMType::FP16,
          kCacheLength,
          sourceFirstToken,
          validTokenCount,
          kind,
          globalLayerOffset + localLayer,
          output,
          outputBytes,
          stats,
          mSharedKvIo ? mKvIoScales.at(mTokenBatchSize).at(kind * localLayers + localLayer).first : 0.0f);
    }
  }
  // Explicit diagnostic only; exclude this reference conversion from speed runs.
  const char* verify = std::getenv("MTK_PD_VERIFY_KV_HANDOFF");
  if (verify && std::string(verify) == "1") {
    const size_t layerValues = numKVHeads * validTokenCount * headDim;
    std::vector<uint16_t> canonical(2 * localLayers * layerValues);
    CopyCacheToLocalCanonicalFp16(validTokenCount, canonical);
    std::vector<uint8_t> reference(outputBytes);
    QnnKvAbiStats referenceStats{};
    for (size_t kind = 0; kind < 2; ++kind) {
      for (size_t localLayer = 0; localLayer < localLayers; ++localLayer) {
        const size_t globalLayer = globalLayerOffset + localLayer;
        abi.ConvertCacheLayer(
            canonical.data() + (kind * localLayers + localLayer) * layerValues,
            true, validTokenCount, 0, validTokenCount, kind, globalLayer,
            reference.data(), reference.size(), &referenceStats);
        const size_t offset = (kind * abi.NumLayers() + globalLayer) * layerValues;
        ET_CHECK_MSG(std::memcmp(output + offset, reference.data() + offset, layerValues) == 0,
                     "QNN handoff reference mismatch: layer=%zu kind=%zu", globalLayer, kind);
      }
    }
    ET_CHECK_MSG(referenceStats.nonFinite == 0, "Non-finite canonical KV handoff");
    std::fprintf(stderr,
        "MTK_KV_HANDOFF_VERIFIED chunk=%zu tokens=%zu values=%zu shared_int16=%d\n",
        mChunkIndex, validTokenCount, canonical.size(), int(mSharedKvIo));
  }
}

void LlamaModelChunk::Run() {
  static const bool profile = [] {
    const char* value = std::getenv("MTK_PD_DETAIL_TIMING");
    return value && std::string(value) == "1";
  }();
  using Clock = std::chrono::steady_clock;
  const auto begin = profile ? Clock::now() : Clock::time_point{};
  // Validate every selected method before execution, including AR128 -> AR1
  // switches. Mixed legacy/incremental methods must not silently alias KV.
  const auto meta = GetModelMethod().method_meta();
  for (size_t i = 0; i < getNumOutputsFor(IOKind::KVCache); ++i) {
    const auto out = meta.output_tensor_meta(getOutputIndex(IOKind::KVCache, i));
    const auto shape = out->sizes();
    const auto output = GetOutputBuffer(getOutputIndex(IOKind::KVCache, i));
    ET_CHECK_MSG(shape.size() == 4 && static_cast<size_t>(shape[2]) ==
                     (mOutputNewCacheOnly ? mTokenBatchSize : kCacheLength) &&
                     output.nbytes >= out->nbytes(),
                 "Incompatible KV ABI or insufficient output storage after method switch");
  }
  if (mQuantizedKvIo) {
    ET_CHECK_MSG(mKvIoScales.count(mTokenBatchSize), "Missing KV method scales");
    const auto& selected = mKvIoScales.at(mTokenBatchSize);
    for (const auto& entry : mKvIoScales) {
      ET_CHECK_MSG(entry.second.size() == kCacheCount, "Incomplete KV scales");
      for (size_t i=0;i<kCacheCount;++i)
        ET_CHECK_MSG(selected[i].first > 0 && selected[i].second > 0 &&
                         selected[i].first == entry.second[i].first,
                     "KV history input scales must agree across methods");
    }
  }
  UpdatePosEmbAndMask(mTokenBatchSize);
  const auto prepared = profile ? Clock::now() : Clock::time_point{};
  ModelChunk::Run();
  const auto executed = profile ? Clock::now() : Clock::time_point{};
  if (mOutputNewCacheOnly) {
    UpdateCacheFromNewOutputs();
  }
  const auto updated = profile ? Clock::now() : Clock::time_point{};
  PaddingPostprocess();
  const auto padded = profile ? Clock::now() : Clock::time_point{};
  if (profile) {
    const auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::fprintf(stderr,
        "MTK_DETAIL_HOST token=%zu new_kv=%d prepare_us=%.3f method_us=%.3f kv_us=%.3f padding_us=%.3f\n",
        mCurrentTokenIndex, int(mOutputNewCacheOnly), us(begin, prepared),
        us(prepared, executed), us(executed, updated), us(updated, padded));
  }
  AdvanceTokenIndex();
}

void LlamaModelChunk::SetPosEmbed(const size_t tokenIndex) {
  if (tokenIndex >= kMaxTokenLength) {
    ET_LOG(
        Fatal,
        "Attempting to set rotaty embedding using index exceeding the supported max token length "
        "(%zu)",
        kMaxTokenLength);
  }

  auto getRotEmbInputs = [&]() {
    std::vector<void*> rotEmbInputs;
    const auto& RotEmbInputIndexes = getInputIndexes(IOKind::RotEmb);
    rotEmbInputs.reserve(RotEmbInputIndexes.size());
    for (const auto inputIdx : RotEmbInputIndexes)
      rotEmbInputs.push_back(mInputBufferInfos[inputIdx].data);
    return rotEmbInputs;
  };
  kRotEmbMasterLut->setEmbed(
      getRotEmbInputs(),
      tokenIndex,
      mTokenBatchSize,
      GetLeftPadding(),
      GetRightPadding());
}

void LlamaModelChunk::PrepareCacheIOs() {
  // Get cache shape
  const auto method_meta = GetModelMethod().method_meta();
  const auto firstInCacheIdx = getInputIndex(IOKind::KVCache);
  mCacheShape = method_meta.input_tensor_meta(firstInCacheIdx)->sizes();

  mQuantizedKvIo = method_meta.input_tensor_meta(firstInCacheIdx)->scalar_type() ==
      executorch::aten::ScalarType::Short;
  if (mQuantizedKvIo) {
    ET_CHECK_MSG(kCacheType == LLMType::FP32, "Int16 KV IO requires FP32 logical cache configuration");
    kCacheTypeSize = sizeof(int16_t);
    const char* root = std::getenv("MTK_PD_KV_IO_QPARAMS");
    ET_CHECK_MSG(root && *root, "Int16 KV IO requires MTK_PD_KV_IO_QPARAMS");
    char filename[64]; std::snprintf(filename, sizeof(filename), "/chunk_%02zu.txt", mChunkIndex);
    std::ifstream file(std::string(root) + filename);
    ET_CHECK_MSG(file.good(), "Missing KV IO qparams for chunk %zu", mChunkIndex);
    size_t ar, index; float inputScale, outputScale;
    while (file >> ar >> index >> inputScale >> outputScale) {
      ET_CHECK_MSG(index < kCacheCount && inputScale > 0 && outputScale > 0 &&
                       std::isfinite(inputScale) && std::isfinite(outputScale), "Invalid KV IO scale");
      auto& row = mKvIoScales[ar]; row.resize(kCacheCount);
      ET_CHECK_MSG(row[index].first == 0, "Duplicate KV IO scale");
      row[index] = {inputScale, outputScale};
    }
    ET_CHECK_MSG(file.eof(), "Malformed KV IO sidecar");
    ET_CHECK_MSG(mKvIoScales.count(mTokenBatchSize), "Missing selected KV method scales");
    mSharedKvIo = true;
    const auto& reference = mKvIoScales.at(mTokenBatchSize);
    for (const auto& entry : mKvIoScales) {
      for (size_t i=0;i<kCacheCount;++i) {
        ET_CHECK_MSG(entry.second[i].first > 0 && entry.second[i].second > 0,
                     "Missing KV scale entry");
        mSharedKvIo = mSharedKvIo && entry.second[i].first == reference[i].first &&
            entry.second[i].second == reference[i].first;
      }
    }
    ET_LOG(Info, "MTK Int16 KV storage: %s",
           mSharedKvIo ? "shared_scale_direct_append" : "requantize_with_fp32_handoff");
    // Unshared legacy scales need a full-range shadow for decode. Shared
    // scales keep only the Int16 history and append output codes directly.
    const size_t values = GetCacheNumRows() * kCacheLength * GetCacheHeadDim();
    if (!mSharedKvIo) {
      mHandoffCache.resize(kCacheCount);
      for (auto& cache : mHandoffCache) cache.resize(values);
    }
  }

  // Incremental outputs must have separate storage: aliasing them with the
  // history would overwrite KV still read by the NPU. Detect the ABI from PTE
  // metadata so legacy full-cache models remain usable.
  const auto firstOut = method_meta.output_tensor_meta(getOutputIndex(IOKind::KVCache));
  const auto outputShape = firstOut->sizes();
  ET_CHECK_MSG(outputShape.size() == 4, "Expected a 4D KV output");
  mOutputNewCacheOnly = static_cast<size_t>(outputShape[2]) != kCacheLength ||
      std::string(method_meta.name()).find("_new_kv_only") != std::string::npos;
  ET_CHECK_MSG(!mQuantizedKvIo || mOutputNewCacheOnly, "Int16 KV requires new-token outputs");
  if (mOutputNewCacheOnly) {
    ET_CHECK_MSG(static_cast<size_t>(outputShape[2]) == mTokenBatchSize,
                 "KV output must contain either full history or new tokens");
  }
  ET_LOG(Info, "MTK KV output ABI: %s",
         mOutputNewCacheOnly ? "new_tokens_only" : "full_cache");
  const size_t numCaches = getNumInputsFor(IOKind::KVCache);
  for (size_t i = 0; i < numCaches; i++) {
    const auto inMeta = method_meta.input_tensor_meta(getInputIndex(IOKind::KVCache, i));
    const auto outMeta = method_meta.output_tensor_meta(getOutputIndex(IOKind::KVCache, i));
    const auto inShape = inMeta->sizes();
    const auto outShape = outMeta->sizes();
    ET_CHECK_MSG(inShape.size() == 4 && outShape.size() == 4 &&
                     inShape[0] == outShape[0] && inShape[1] == outShape[1] &&
                     inShape[3] == outShape[3] &&
                     static_cast<size_t>(inShape[2]) == kCacheLength &&
                     static_cast<size_t>(outShape[2]) ==
                         (mOutputNewCacheOnly ? mTokenBatchSize : kCacheLength) &&
                     inMeta->scalar_type() == outMeta->scalar_type(),
                 "Inconsistent KV input/output ABI");
    if (!mOutputNewCacheOnly) {
      this->LinkModelIO(
          getInputIndex(IOKind::KVCache, i), getOutputIndex(IOKind::KVCache, i));
    }
  }
}

void LlamaModelChunk::UpdateCacheFromNewOutputs() {
  const auto meta = GetModelMethod().method_meta();
  for (size_t i = 0; i < getNumInputsFor(IOKind::KVCache); ++i) {
    const auto input = GetInputBuffer(getInputIndex(IOKind::KVCache, i));
    const auto output = GetOutputBuffer(getOutputIndex(IOKind::KVCache, i));
    const auto shape = meta.output_tensor_meta(getOutputIndex(IOKind::KVCache, i))->sizes();
    ET_CHECK_MSG(shape.size() == 4 &&
                     static_cast<size_t>(shape[2]) == mTokenBatchSize &&
                     input.data != output.data,
                 "Invalid incremental KV output after method switch");
    ET_CHECK_MSG(input.nbytes >= GetCacheNumRows() * kCacheLength * GetCacheStrideSize() &&
                     output.nbytesUsed >= GetCacheNumRows() * mTokenBatchSize * GetCacheStrideSize(),
                 "KV buffer is smaller than its contiguous tensor shape");
    if (mQuantizedKvIo && !mSharedKvIo) {
      const auto scales = mKvIoScales.at(mTokenBatchSize).at(i);
      const size_t values = GetCacheNumRows() * mTokenBatchSize * GetCacheHeadDim();
      mNewKvFloat.resize(values); mNewKvInput.resize(values);
      const auto* q = static_cast<const int16_t*>(output.data);
      size_t j = 0;
      #if defined(__aarch64__)
      const auto outScale = vdupq_n_f32(scales.second);
      const auto invScale = vdupq_n_f32(1.0f / scales.first);
      for (; j + 4 <= values; j += 4) {
        const auto v = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vld1_s16(q+j))), outScale);
        vst1q_f32(mNewKvFloat.data()+j, v);
        const auto clipped = vmaxq_f32(vdupq_n_f32(-32768.0f),
            vminq_f32(vdupq_n_f32(32767.0f), vmulq_f32(v, invScale)));
        vst1_s16(mNewKvInput.data()+j, vqmovn_s32(vcvtnq_s32_f32(clipped)));
      }
      #endif
      for (; j < values; ++j) {
        const float v = q[j] * scales.second; mNewKvFloat[j] = v;
        mNewKvInput[j] = static_cast<int16_t>(std::nearbyint(
            std::max(-32768.0f, std::min(32767.0f, v / scales.first))));
      }
      UpdateSlidingKvCache(input.data, mNewKvInput.data(), GetCacheNumRows(),
          kCacheLength, mTokenBatchSize, GetCacheStrideSize(), mCurrentTokenIndex);
      UpdateSlidingKvCache(mHandoffCache[i].data(), mNewKvFloat.data(), GetCacheNumRows(),
          kCacheLength, mTokenBatchSize, GetCacheHeadDim()*sizeof(float), mCurrentTokenIndex);
      continue;
    }
    UpdateSlidingKvCache(input.data, output.data, GetCacheNumRows(),
                         kCacheLength, mTokenBatchSize, GetCacheStrideSize(),
                         mCurrentTokenIndex);
  }
}

size_t LlamaModelChunk::GetCacheNumRows() const {
  return std::reduce(
      mCacheShape.begin(),
      mCacheShape.begin() + kCacheLengthDim,
      1,
      std::multiplies<>());
}

size_t LlamaModelChunk::GetCacheStrideSize() const {
  return std::reduce(
      mCacheShape.begin() + kCacheLengthDim + 1,
      mCacheShape.end(),
      kCacheTypeSize,
      std::multiplies<>());
}

void LlamaModelChunk::InitMaskBuilder() {
  mMaskBuilder = std::make_unique<MaskBuilder>(kMaskType, kCacheLength);
  // SWA Mask
  if (enableSWA) {
    const auto& swaMaskBufferInfo =
        mInputBufferInfos[getInputIndex(IOKind::SWAMask)];
    const auto swaMaskBuffer = swaMaskBufferInfo.data;
    const auto swaMaskSizeBytes = swaMaskBufferInfo.nbytesUsed;
    mMaskBuilder->setMaskBuffer(swaMaskBuffer, swaMaskSizeBytes);
    mMaskBuilder->enableSlidingWindow(kWindowSize);
    mMaskBuilder->buildMask(mTokenBatchSize, mCurrentTokenIndex);
  }
  // Global Mask
  const auto& maskBufferInfo = mInputBufferInfos[getInputIndex(IOKind::Mask)];
  const auto maskBuffer = maskBufferInfo.data;
  const auto maskSizeBytes = maskBufferInfo.nbytesUsed;
  mMaskBuilder->setMaskBuffer(maskBuffer, maskSizeBytes);
  mMaskBuilder->disableSlidingWindow();
  mMaskBuilder->buildMask(mTokenBatchSize, mCurrentTokenIndex);

  mMaskBuilder->resetPadLength();
}

void LlamaModelChunk::InitCache() {
  for (auto& cache : mHandoffCache) std::fill(cache.begin(), cache.end(), 0.0f);
  // Zero initialization
  for (const auto cacheIdx : getInputIndexes(IOKind::KVCache)) {
    const auto& inputCacheInfo = mInputBufferInfos[cacheIdx];
    char* cacheBuffer = reinterpret_cast<char*>(inputCacheInfo.data);
    const size_t cacheSizeBytes = inputCacheInfo.nbytes;
    std::memset(cacheBuffer, 0, cacheSizeBytes);
  }
}


BufferInfo LlamaModelChunk::GetHandoffCache(size_t localIndex) {
  if (!mQuantizedKvIo || mSharedKvIo) return GetInputBuffer(getInputIndex(IOKind::KVCache, localIndex));
  auto& v = mHandoffCache.at(localIndex);
  return {v.data(), v.size()*sizeof(float), v.size()*sizeof(float)};
}

void LlamaModelChunk::PadHandoffCache(size_t pad, size_t seen, bool left) {
  if (!mQuantizedKvIo || mSharedKvIo || !pad) return;
  const size_t dim = GetCacheHeadDim(), rows = GetCacheNumRows();
  const size_t alive = std::min(seen, kCacheLength);
  const size_t keep = alive > pad ? alive-pad : 0;
  for (auto& cache : mHandoffCache) {
    if (!left && !keep) { std::fill(cache.begin(), cache.end(), 0.0f); continue; }
    for (size_t row=0;row<rows;++row) {
      auto* data = cache.data()+row*kCacheLength*dim;
      if (left) {
        std::fill(data+(kCacheLength-mTokenBatchSize)*dim,
                  data+(kCacheLength-mTokenBatchSize+pad)*dim, 0.0f);
      } else {
        const size_t start=kCacheLength-alive;
        std::memmove(data+(start+pad)*dim, data+start*dim, keep*dim*sizeof(float));
        std::fill(data+start*dim, data+(start+pad)*dim, 0.0f);
      }
    }
  }
}

} // namespace example
