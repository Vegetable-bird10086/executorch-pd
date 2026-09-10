/*
 * Copyright (c) 2024 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "NeuronBackend.h"
#include "NeuronBufferAllocator.h"
#include "NeuronLog.h"
#include "NeuronPayloadHeader.h"
#include "api/NeuronAdapter.h"

#include <executorch/runtime/executor/pte_data_map.h>
#include "executorch/runtime/core/error.h"

#include <algorithm>
#include <chrono>
#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <new>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace neuron {

using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

const char kHighAddrKey[] = "HighAddr";
const char kImportForeverKey[] = "ImportForever";
const char kSharedWeightsKey[] = "ExtractSharedBlobKey";

Result<DelegateHandle*> NeuronBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  NeuronDelegateSetting setting;
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  NeuronExecuTorchDelegate* delegate =
      runtime_allocator->allocateInstance<NeuronExecuTorchDelegate>();
  if (delegate == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  new (delegate) NeuronExecuTorchDelegate();

  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, kHighAddrKey) == 0) {
      setting.mHighAddr = *static_cast<char*>(compile_spec.value.buffer);
      LogInfo("NeuronBackend", "IsHighAddr Enable : %d", setting.mHighAddr);
    } else if (std::strcmp(compile_spec.key, kImportForeverKey) == 0) {
      setting.mImportForever = *static_cast<char*>(compile_spec.value.buffer);
      LogInfo(
          "NeuronBackend",
          "IsImportForever Enable : %d",
          setting.mImportForever);
    } else if (std::strcmp(compile_spec.key, kSharedWeightsKey) == 0) {
      setting.mSharedWeights = true;
      std::string shared_weights_key(
          static_cast<char*>(compile_spec.value.buffer),
          compile_spec.value.nbytes);
      LogInfo(
          "NeuronBackend",
          "SharedWeights Enabled for %s",
          shared_weights_key.c_str());
      std::shared_ptr<NeuronSharedWeights> neuron_shared_weights;
      if (neuron_shared_weights_cache_.find(shared_weights_key) !=
          neuron_shared_weights_cache_.end()) {
        neuron_shared_weights =
            neuron_shared_weights_cache_.at(shared_weights_key).lock();
        if (neuron_shared_weights) {
          LogInfo(
              "NeuronBackend",
              "Reusing cached shared weights with key %s",
              shared_weights_key.c_str());
          delegate->SetSharedWeights(neuron_shared_weights);
          continue;
        } else {
          LogInfo(
              "NeuronBackend",
              "Shared weights cache expired: %s",
              shared_weights_key.c_str());
          neuron_shared_weights_cache_.erase(shared_weights_key); // Expired
        }
      }
      const NamedDataMap* named_data_map = context.get_named_data_map();
      Result<FreeableBuffer> shared_weights =
          named_data_map->get_data(shared_weights_key.c_str());

      if (shared_weights.ok()) {
        LogInfo(
            "NeuronBackend",
            "Loaded shared weights from named_data_map. Size: %zu",
            shared_weights.get().size());
        FreeableBuffer& buffer = shared_weights.get();
        neuron_shared_weights =
            std::make_shared<NeuronSharedWeights>(std::move(buffer));
        delegate->SetSharedWeights(neuron_shared_weights);
        neuron_shared_weights_cache_[shared_weights_key] =
            neuron_shared_weights;
      } else {
        LogError(
            "NeuronBackend",
            "Failed to load shared weights from named_data_map.");
        return Error::Internal;
      }
    } else {
      LogWarn("NeuronBackend", "unknown compile spec: %s", compile_spec.key);
    }
  }
  auto Payload = NeuronPayload(processed->data(), processed->size());

  LogInfo(
      "NeuronBackend",
      "version %u, input %u, output %u, length %u, payload size: %zu",
      Payload.Header.Version,
      Payload.Header.InputCount,
      Payload.Header.OutputCount,
      Payload.Header.DataLen,
      processed->size());

  int res = delegate->LoadCompiledNetwork(Payload, setting);
  if (res == NEURON_NO_ERROR) {
    // NeuronExecutor has imported the compiled network into its own runtime
    // handle. Honor BackendInterface's processed-buffer contract so an owning
    // DataLoader can release this usually-large PTE segment immediately.
    processed->Free();
  }
  return res == NEURON_NO_ERROR ? delegate : nullptr;
}

Error NeuronBackend::execute(
    ET_UNUSED BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  NeuronExecuTorchDelegate* delegate =
      reinterpret_cast<NeuronExecuTorchDelegate*>(handle);
  return delegate->execute(context, args);
}

void NeuronBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    NeuronExecuTorchDelegate* delegate =
        reinterpret_cast<NeuronExecuTorchDelegate*>(handle);
    delegate->~NeuronExecuTorchDelegate();
  }
}

bool NeuronBackend::is_available() const {
  return true;
}

Error NeuronExecuTorchDelegate::execute(
    BackendExecutionContext& context,
    Span<EValue*> args) const {
  static const bool profile = [] {
    const char* value = std::getenv("MTK_PD_DETAIL_TIMING");
    return value && std::string(value) == "1";
  }();
  static const bool bindProfile = [] {
    const char* value = std::getenv("MTK_PD_BIND_TIMING");
    return value && std::string(value) == "1";
  }();
  using Clock = std::chrono::steady_clock;
  struct BindTiming {
    const char* kind;
    size_t index, bytes;
    bool hit = false, memory = false;
    double cache = 0, find = 0, lock = 0, lookup = 0, sdk = 0;
    int status = NEURON_NO_ERROR;
  };
  std::vector<BindTiming> bindTimings;
  if (bindProfile) {
    bindTimings.reserve(mInputSizes.size() + neuron_shared_weights_.size() + mOutputSizes.size());
  }
  const auto elapsedUs = [](auto a, auto b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
  };
  const auto begin = profile ? Clock::now() : Clock::time_point{};
  if (HintNeuronBackend(args) != NEURON_NO_ERROR) {
    return Error::InvalidState;
  };

  ET_CHECK_OR_RETURN_ERROR(
      CheckDimOrder(args) == NEURON_NO_ERROR,
      Internal,
      "Expecting default dim_order but got a non default dim_order tensor input");

  PrepareInputsOuputs(args);
  const auto prepared = profile ? Clock::now() : Clock::time_point{};

  auto allocator =
      dynamic_cast<neuron::BufferAllocator*>(context.get_temp_allocator());

  size_t inputCount = mInputSizes.size() + neuron_shared_weights_.size();
  size_t outputCount = mOutputSizes.size();

  const auto bind = [&](auto inputTag, size_t index, const InputOutputInfo& info) {
    constexpr bool isInput = decltype(inputTag)::value;
    auto data_ptr = info.data_ptr;
    auto data_size = info.size;
    BindTiming timing{isInput ? (index < mInputSizes.size() ? "input" : "weight") : "output",
                      index, data_size};
    const auto cacheBegin = bindProfile ? Clock::now() : Clock::time_point{};
    timing.hit = IsCached<isInput>(index, data_ptr);
    if (bindProfile) timing.cache = elapsedUs(cacheBegin, Clock::now());
    if (!timing.hit) {
      const auto findBegin = bindProfile ? Clock::now() : Clock::time_point{};
      auto unit = allocator != nullptr
          ? allocator->Find(data_ptr, bindProfile ? &timing.lock : nullptr,
                            bindProfile ? &timing.lookup : nullptr)
          : nullptr;
      if (bindProfile) timing.find = elapsedUs(findBegin, Clock::now());
      timing.memory = unit != nullptr;
      if (unit) {
        UpdateCache<isInput>(index, data_ptr);
        size_t offset = (char*)data_ptr - (char*)unit->GetAddress();
        auto memory = unit->GetNeuronMemory();
        const auto sdkBegin = bindProfile ? Clock::now() : Clock::time_point{};
        timing.status = mExecutor.SetInputOutputFromMemory<isInput>(
            index, memory, offset, data_size);
        if (bindProfile) timing.sdk = elapsedUs(sdkBegin, Clock::now());
      } else {
        const auto sdkBegin = bindProfile ? Clock::now() : Clock::time_point{};
        timing.status = mExecutor.SetInputOutput<isInput>(index, data_ptr, data_size);
        if (bindProfile) timing.sdk = elapsedUs(sdkBegin, Clock::now());
      }
    }
    if (bindProfile) bindTimings.push_back(timing);
  };
  for (size_t i = 0; i < inputCount; i++) {
    bind(std::true_type{}, i, mPreparedInputs[i]);
  }
  for (size_t o = 0; o < outputCount; o++) {
    bind(std::false_type{}, o, mPreparedOutputs[o]);
  }

  const auto bound = profile ? Clock::now() : Clock::time_point{};
  const int status = mExecutor.Compute();
  if (profile) {
    const auto done = Clock::now();
    const auto us = [](auto a, auto b) {
      return std::chrono::duration<double, std::micro>(b - a).count();
    };
    std::fprintf(stderr,
        "MTK_DETAIL_BACKEND prepare_us=%.3f bind_us=%.3f compute_us=%.3f status=%d\n",
        us(begin, prepared), us(prepared, bound), us(bound, done), status);
  }
  if (bindProfile) {
    for (const auto& t : bindTimings) {
      std::fprintf(stderr,
          "MTK_DETAIL_BIND delegate=%p kind=%s index=%zu bytes=%zu hit=%d memory=%d cache_us=%.3f find_us=%.3f lock_us=%.3f lookup_us=%.3f sdk_us=%.3f status=%d\n",
          static_cast<const void*>(this), t.kind, t.index, t.bytes, t.hit, t.memory,
          t.cache, t.find, t.lock, t.lookup, t.sdk, t.status);
    }
  }
  return status == NEURON_NO_ERROR ? Error::Ok : Error::InvalidState;
};

int NeuronExecuTorchDelegate::HintNeuronBackend(Span<EValue*> args) const {
  auto HintImportForever = [this](Span<EValue*> args) -> int {
    auto& allocator = GET_NEURON_ALLOCATOR;
    size_t inputCount = mInputSizes.size(), outputCount = mOutputSizes.size();
    for (int i = 0; i < inputCount; i++) {
      auto data_ptr = args[i]->toTensor().data_ptr();
      if (mHasImported.count(data_ptr)) {
        continue;
      }
      auto unit = allocator.Find(data_ptr);
      if (unit) {
        mExecutor.SetInputOutputFromMemory</*isInput*/ true>(
            i, unit->GetNeuronMemory(), 0, unit->GetSize());
        mHasImported.insert(data_ptr);
      }
    }
    for (int o = inputCount; o < inputCount + outputCount; o++) {
      auto data_ptr = args[o]->toTensor().data_ptr();
      if (mHasImported.count(data_ptr)) {
        continue;
      }
      auto output_index = o - inputCount;
      auto unit = allocator.Find(data_ptr);
      if (unit) {
        mExecutor.SetInputOutputFromMemory</*isInput*/ false>(
            output_index, unit->GetNeuronMemory(), 0, unit->GetSize());
        mHasImported.insert(data_ptr);
      }
    }
    return NEURON_NO_ERROR;
  };
  if (mSettings.mImportForever) {
    CHECK_NO_ERROR(HintImportForever(args));
  }
  return NEURON_NO_ERROR;
}

} // namespace neuron
} // namespace backends
} // namespace executorch

namespace {
auto cls = executorch::backends::neuron::NeuronBackend();
executorch::runtime::Backend backend{"NeuropilotBackend", &cls};
static auto success_with_compiler =
    executorch::runtime::register_backend(backend);
} // namespace
