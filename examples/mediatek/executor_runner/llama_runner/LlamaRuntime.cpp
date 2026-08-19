/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include <string>
#include <thread>
#include <vector>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <executorch/runtime/platform/log.h>

#include "LlamaRuntime.h"
#include "Utils.h"

#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"

namespace example {

void LlamaRuntime::Initialize(
    const LlamaModelOptions& modelOptions,
    const LlamaModelPaths& modelPaths) {
  mModelOptions = modelOptions;

  // Initialize rotary embedding master lookup table
  const size_t headDim = modelOptions.head_dim
      ? modelOptions.head_dim
      : (modelOptions.hidden_size / modelOptions.num_head);
  const size_t rotEmbDim = headDim * modelOptions.partial_rotary_factor;
  mRotEmbMasterLut = std::make_unique<llm_helper::RotaryEmbeddingMasterLut>(
      modelOptions.rot_emb_type,
      modelOptions.max_token_length,
      rotEmbDim,
      modelOptions.rot_emb_base);
  mRotEmbMasterLut->generate();

  const bool useSharedWeights = !modelPaths.model_package_paths.empty();

  ET_CHECK_MSG(
      !useSharedWeights ||
          modelPaths.prompt_model_paths.empty() &&
              modelPaths.gen_model_paths.empty(),
      "The paths for both prompt and gen model paths should be empty when shared weights is used.");

  const size_t numChunk = useSharedWeights
      ? modelPaths.model_package_paths.size()
      : modelPaths.gen_model_paths.size();
  ET_CHECK_MSG(numChunk > 0, "No model to initialize");
  const size_t numCache = 2 * modelOptions.num_layer / numChunk;

  constexpr size_t numRotEmbInputs = 1;
  const bool usePromptModel = !modelPaths.prompt_model_paths.empty() ||
      !modelPaths.model_package_paths.empty();
  const size_t initBatchSize =
      usePromptModel ? modelOptions.prompt_token_batch_size : 1;
  mTokenBatchSize = initBatchSize;

  // Enable SWA if window size is not 0
  const bool enableSWA = (modelOptions.window_size != 0);

  // Get effective prompt and gen model paths
  const auto& [prompt_model_paths, gen_model_paths] = [&] {
    if (useSharedWeights) {
      return std::pair{
          modelPaths.model_package_paths, modelPaths.model_package_paths};
    }
    return std::pair{modelPaths.prompt_model_paths, modelPaths.gen_model_paths};
  }();

  for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
    ModelPathMap modelPathMap;
    auto addModelPath = [&](const auto& modelPaths, const size_t batchSize) {
      if (modelPaths.empty())
        return;
      modelPathMap[batchSize] = modelPaths[chunkIdx];
    };
    addModelPath(prompt_model_paths, modelOptions.prompt_token_batch_size);
    addModelPath(gen_model_paths, 1);
    auto llamaChunk = std::make_unique<LlamaModelChunk>(
        modelPathMap,
        modelOptions,
        useSharedWeights,
        initBatchSize,
        numCache,
        numRotEmbInputs,
        chunkIdx + 1 == numChunk ? modelOptions.logit_shard_count : 1,
        enableSWA,
        chunkIdx,
        mRotEmbMasterLut.get());
    mLlamaModelChunks.push_back(std::move(llamaChunk));
  }

  for (size_t i = 0; i < numChunk; i++) {
    auto& modelChunk = mLlamaModelChunks[i];
    if (i > 0 && !mModelOptions.copy_chunk_io) {
      const auto& prevModelChunk = mLlamaModelChunks[i - 1];
      modelChunk->SetInputBuffer(prevModelChunk->GetOutputBuffer());
    }
    modelChunk->Initialize();
    // modelChunk->LogIoSummary();
  }

  // NOTE: Token embedding type here is assumed to follow the model input
  // embedding type.
  mTokenEmbLut = std::make_unique<llm_helper::TokenEmbeddingLut>(
      modelPaths.token_embedding_path,
      modelOptions.model_input_type,
      modelOptions.hidden_size);

  // Link first chunk emb input to token emb lut output
  const auto& tokenEmbInput = mLlamaModelChunks.front()->GetInputBuffer();
  mTokenEmbLut->setOutput(tokenEmbInput.data, tokenEmbInput.nbytes);
}

void LlamaRuntime::Release() {
  for (auto& llamaChunk : mLlamaModelChunks) {
    llamaChunk->Release();
  }
  mLlamaModelChunks.clear();
  mRotEmbMasterLut.reset();
  mTokenEmbLut.reset();
}

void LlamaRuntime::SwapModel(const size_t batchSize) {
  auto hotSwapChunk = [&](const auto chunkIdx) {
    const auto status = mLlamaModelChunks[chunkIdx]->HotSwapModel(batchSize);
    if (!status)
      ET_LOG(Error, "Hot swapping failed on chunk %zu", chunkIdx);
  };

  // Use multi-threading to speedup model swapping
  std::vector<std::thread> threads;
  for (size_t i = 0; i < mLlamaModelChunks.size(); i++)
    threads.emplace_back(hotSwapChunk, i);
  for (size_t i = 0; i < mLlamaModelChunks.size(); i++)
    threads[i].join();

  mTokenBatchSize = batchSize;
}

void LlamaRuntime::Reset() {
  for (auto& modelChunk : mLlamaModelChunks) {
    static_cast<LlamaModelChunk*>(modelChunk.get())->Reset();
  }
  mTokenIndex = 0;
}

void* LlamaRuntime::Run(
    const std::vector<uint64_t>& inputTokens,
    const bool lastLogits,
    const ChunkCompleteCallback& chunkCompleteCallback) {
  const auto& firstLlamaChunk = mLlamaModelChunks.front();
  const auto tokenIndex =
      static_cast<LlamaModelChunk*>(firstLlamaChunk.get())->GetTokenIndex();
  const auto numNewInputToken = inputTokens.size();

  ET_CHECK_MSG(
      numNewInputToken <= mTokenBatchSize,
      "Input token length (%zu) > model token batch size (%zu)",
      numNewInputToken,
      mTokenBatchSize);

  // Handle padding
  auto curInputTokens = inputTokens; // Make a copy
  const size_t padSize = mTokenBatchSize - numNewInputToken;
  constexpr uint64_t padToken = 0;

  // Use left-padding if possible as it has lower overhead than right-padding.
  // Right-padding involves cache shifting which incurs additional overhead.
  const bool isLeftPadAllowed = (tokenIndex == 0);
  if (padSize > 0) {
    if (isLeftPadAllowed) {
      // Pad left since the cache is fresh new.
      curInputTokens.insert(curInputTokens.begin(), padSize, padToken);
    } else {
      // Pad right since left side of cache is occupied either by loaded cache
      // or previous inference pass.
      curInputTokens.insert(curInputTokens.end(), padSize, padToken);
    }
    ET_LOG(Debug, "Padding size = %zu", padSize);
  }

  // Begin inference flow

  // Lookup token embedding
  mTokenEmbLut->lookupEmbedding(curInputTokens);

  if (tokenIndex == 0 && !mModelOptions.first_chunk_input_path.empty()) {
    const auto inputBuffer = mLlamaModelChunks.front()->GetInputBuffer();
    std::ifstream stream(
        mModelOptions.first_chunk_input_path, std::ios::binary | std::ios::ate);
    ET_CHECK_MSG(
        stream.good(),
        "Unable to open first chunk input %s",
        mModelOptions.first_chunk_input_path.c_str());
    const auto fileSize = static_cast<size_t>(stream.tellg());
    ET_CHECK_MSG(
        fileSize == inputBuffer.nbytesUsed,
        "First chunk input size mismatch: file=%zu, model=%zu",
        fileSize,
        inputBuffer.nbytesUsed);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(inputBuffer.data), fileSize);
    ET_CHECK_MSG(
        stream.good(),
        "Unable to read first chunk input %s",
        mModelOptions.first_chunk_input_path.c_str());
    ET_LOG(
        Info,
        "Loaded first chunk input: %zu bytes <- %s",
        fileSize,
        mModelOptions.first_chunk_input_path.c_str());
  }

  // Decoder chunks
  for (size_t chunkIdx = 0; chunkIdx < mLlamaModelChunks.size(); ++chunkIdx) {
    auto& modelChunk = mLlamaModelChunks[chunkIdx];
    auto llamaChunk = static_cast<LlamaModelChunk*>(modelChunk.get());

    if (chunkIdx > 0 && mModelOptions.copy_chunk_io) {
      const auto& prevModelChunk = mLlamaModelChunks[chunkIdx - 1];
      modelChunk->SetInputBuffer(prevModelChunk->GetOutputBuffer());
    }

    // Set padding if needed.
    if (isLeftPadAllowed)
      llamaChunk->SetLeftPadding(padSize);
    else
      llamaChunk->SetRightPadding(padSize);

    // Run model chunk
    llamaChunk->Run();

    if (chunkCompleteCallback) {
      chunkCompleteCallback(chunkIdx, *llamaChunk);
    }

    // Dump the unmodified primary output before it is consumed by the next
    // chunk. This does not add graph outputs or change backend partitioning.
    if (tokenIndex == 0 && !mModelOptions.chunk_debug_dump_dir.empty()) {
      const auto buffer = modelChunk->GetOutputBuffer();
      std::ostringstream path;
      path << mModelOptions.chunk_debug_dump_dir << "/chunk_" << std::setfill('0')
           << std::setw(2) << chunkIdx << "_prefill_output_f32.bin";
      std::ofstream stream(path.str(), std::ios::binary);
      ET_CHECK_MSG(
          stream.good(), "Unable to open chunk dump %s", path.str().c_str());
      stream.write(reinterpret_cast<const char*>(buffer.data), buffer.nbytesUsed);
      ET_CHECK_MSG(
          stream.good(), "Unable to write chunk dump %s", path.str().c_str());
      ET_LOG(
          Info,
          "Chunk debug output %zu: %zu bytes -> %s",
          chunkIdx,
          buffer.nbytesUsed,
          path.str().c_str());
    }

    // Dump only the first (prefill) invocation. Extra outputs are ordered after
    // output 0 and all KV-cache outputs. The caller creates the destination dir.
    if (tokenIndex == 0 && !mModelOptions.layer_debug_dump_dir.empty()) {
      const size_t numCache =
          2 * mModelOptions.num_layer / mLlamaModelChunks.size();
      const size_t debugOutputCount =
          mModelOptions.layer_debug_chunk_index < 0 ||
              static_cast<size_t>(mModelOptions.layer_debug_chunk_index) ==
                  chunkIdx
          ? mModelOptions.layer_debug_output_count
          : 0;
      for (size_t localLayer = 0; localLayer < debugOutputCount; ++localLayer) {
        const size_t outputIndex = 1 + numCache + localLayer;
        const auto buffer = modelChunk->GetOutputBuffer(outputIndex);
        std::ostringstream path;
        if (mModelOptions.layer_debug_chunk_index >= 0) {
          path << mModelOptions.layer_debug_dump_dir << "/operator_"
               << std::setfill('0') << std::setw(2) << localLayer
               << "_prefill_f32.bin";
        } else {
          const size_t globalLayer =
              chunkIdx * mModelOptions.layer_debug_output_count + localLayer;
          path << mModelOptions.layer_debug_dump_dir << "/layer_"
               << std::setfill('0') << std::setw(2) << globalLayer
               << "_prefill_f32.bin";
        }
        std::ofstream stream(path.str(), std::ios::binary);
        ET_CHECK_MSG(stream.good(), "Unable to open layer dump %s", path.str().c_str());
        stream.write(
            reinterpret_cast<const char*>(buffer.data), buffer.nbytesUsed);
        ET_CHECK_MSG(stream.good(), "Unable to write layer dump %s", path.str().c_str());
        ET_LOG(
            Info,
            "Layer debug output %zu: %zu bytes -> %s",
            localLayer,
            buffer.nbytesUsed,
            path.str().c_str());
      }
    }

  }

  // Only consider valid tokens by ignoring padding
  mTokenIndex += inputTokens.size();

  // Return logits
  const auto& finalChunk = mLlamaModelChunks.back();
  if (mModelOptions.logit_shard_count > 1) {
    const size_t elementSize =
        llm_helper::getLLMTypeSize(mModelOptions.model_output_type);
    const size_t returnedTokens = lastLogits ? 1 : mTokenBatchSize;
    mShardedLogits.clear();
    mShardedLogits.reserve(
        returnedTokens * mModelOptions.vocab_size * elementSize);
    const size_t rightPadSize = !isLeftPadAllowed * padSize;
    const size_t firstToken =
        lastLogits ? mTokenBatchSize - 1 - rightPadSize : 0;
    for (size_t tokenOffset = 0; tokenOffset < returnedTokens; ++tokenOffset) {
      const size_t token = firstToken + tokenOffset;
      for (size_t shard = 0; shard < mModelOptions.logit_shard_count; ++shard) {
        const auto buffer = finalChunk->GetOutputBuffer(shard);
        ET_CHECK_MSG(
            buffer.nbytesUsed % mTokenBatchSize == 0,
            "Logit shard %zu size is not divisible by token batch size",
            shard);
        const size_t tokenBytes = buffer.nbytesUsed / mTokenBatchSize;
        const char* source = reinterpret_cast<const char*>(buffer.data) +
            token * tokenBytes;
        mShardedLogits.insert(
            mShardedLogits.end(), source, source + tokenBytes);
      }
    }
    ET_CHECK_MSG(
        mShardedLogits.size() ==
            returnedTokens * mModelOptions.vocab_size * elementSize,
        "Combined logits size mismatch: got %zu expected %zu",
        mShardedLogits.size(),
        returnedTokens * mModelOptions.vocab_size * elementSize);
    if (!mModelOptions.combined_logits_dump_dir.empty()) {
      std::ostringstream path;
      path << mModelOptions.combined_logits_dump_dir << "/logits_token_"
           << std::setfill('0') << std::setw(5) << mTokenIndex << ".bin";
      std::ofstream stream(path.str(), std::ios::binary);
      ET_CHECK_MSG(
          stream.good(), "Unable to open combined logits dump %s", path.str().c_str());
      stream.write(mShardedLogits.data(), mShardedLogits.size());
      ET_CHECK_MSG(
          stream.good(), "Unable to write combined logits dump %s", path.str().c_str());
    }
    return mShardedLogits.data();
  }
  const auto logitsBuffer = finalChunk->GetOutputBuffer();
  const auto logitsData = reinterpret_cast<char*>(logitsBuffer.data);
  const auto logitsSize = logitsBuffer.nbytesUsed;
  size_t offset = 0;
  const size_t rightPadSize = !isLeftPadAllowed * padSize;
  if (lastLogits && mTokenBatchSize > 1) {
    offset =
        (logitsSize / mTokenBatchSize) * (mTokenBatchSize - 1 - rightPadSize);
    ET_DCHECK(offset <= logitsSize);
  }
  return logitsData + offset;
}

size_t LlamaRuntime::GetTokenBatchSize() const {
  return mTokenBatchSize;
}

size_t LlamaRuntime::GetTokenIndex() const {
  return mTokenIndex;
}

const LlamaModelOptions& LlamaRuntime::GetModelOptions() const {
  return mModelOptions;
}

size_t LlamaRuntime::GetNumKVHeads() const {
  ET_CHECK_MSG(!mLlamaModelChunks.empty(), "Llama runtime is not initialized");
  return static_cast<LlamaModelChunk*>(mLlamaModelChunks.front().get())
      ->GetNumKVHeads();
}

size_t LlamaRuntime::GetCacheHeadDim() const {
  ET_CHECK_MSG(!mLlamaModelChunks.empty(), "Llama runtime is not initialized");
  return static_cast<LlamaModelChunk*>(mLlamaModelChunks.front().get())
      ->GetCacheHeadDim();
}

size_t LlamaRuntime::GetCacheLength() const {
  ET_CHECK_MSG(!mLlamaModelChunks.empty(), "Llama runtime is not initialized");
  return static_cast<LlamaModelChunk*>(mLlamaModelChunks.front().get())
      ->GetCacheLength();
}

} // namespace example
