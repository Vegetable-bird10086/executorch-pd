/*
 * Copyright (c) 2025 MediaTek Inc.
 *
 * Licensed under the BSD License (the "License"); you may not use this file
 * except in compliance with the License. See the license file in the root
 * directory of this source tree for more details.
 */

#include "include/NeuronBufferAllocator.h"

#include <algorithm>
#include <cstddef>
#include <chrono>
#include <mutex>

namespace executorch {
namespace backends {
namespace neuron {

bool loadLibrary() {
  handle = dlopen("libneuron_buffer_allocator.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Failed to load library: " << dlerror() << std::endl;
    return false;
  }

  create_func = (CreateFunc)dlsym(handle, "neuron_buffer_allocator_create");
  allocate_func = (AllocateFunc)dlsym(handle, "neuron_buffer_allocate");
  remove_func = (RemoveFunc)dlsym(handle, "neuron_buffer_remove");
  find_func = (FindFunc)dlsym(handle, "neuron_buffer_find");
  clear_func = (ClearFunc)dlsym(handle, "neuron_buffer_clear");

  if (!create_func || !allocate_func || !remove_func || !find_func ||
      !clear_func) {
    std::cerr << "Failed to retrieve symbols: " << dlerror() << std::endl;
    dlclose(handle);
    handle = nullptr;
    return false;
  }

  return true;
}

void unloadLibrary() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
}

BufferAllocator& BufferAllocator::GetInstance() {
  static BufferAllocator instance;

  if (handle == nullptr) {
    loadLibrary();
    if (allocatorHandle == nullptr) {
      allocatorHandle = create_func();
    }
  }

  return instance;
};

void* BufferAllocator::Allocate(size_t size) {
  std::scoped_lock Guard(mMutex);
  return allocate_func(allocatorHandle, size);
}

bool BufferAllocator::RemoveBuffer(void* address) {
  std::scoped_lock Guard(mMutex);
  return remove_func(allocatorHandle, address);
}

const MemoryUnit* BufferAllocator::Find(
    void* address, double* lockUs, double* lookupUs) {
  if (lockUs == nullptr || lookupUs == nullptr) {
    std::scoped_lock Guard(mMutex);
    return static_cast<const MemoryUnit*>(find_func(allocatorHandle, address));
  }
  using Clock = std::chrono::steady_clock;
  const auto begin = Clock::now();
  std::unique_lock<std::mutex> guard(mMutex);
  const auto locked = Clock::now();
  const auto* unit =
      static_cast<const MemoryUnit*>(find_func(allocatorHandle, address));
  const auto found = Clock::now();
  guard.unlock();
  *lockUs = std::chrono::duration<double, std::micro>(locked - begin).count();
  *lookupUs = std::chrono::duration<double, std::micro>(found - locked).count();
  return unit;
}

void BufferAllocator::Clear() {
  std::scoped_lock Guard(mMutex);
  clear_func(allocatorHandle);
}

} // namespace neuron
} // namespace backends
} // namespace executorch

namespace {
static auto& singletonInstance = GET_NEURON_ALLOCATOR;
} // namespace
