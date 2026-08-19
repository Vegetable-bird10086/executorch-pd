/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstddef>

#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace executorch {
namespace backends {
namespace qnn {

class QnnExecuTorchBackend;

// Minimal execution owner detached from an ExecuTorch Method/Program. It owns
// one external reference to the initialized QNN delegate and needs no PTE
// serialization data.
class QnnDetachedExecution final {
 public:
  ~QnnDetachedExecution();

  QnnDetachedExecution(const QnnDetachedExecution&) = delete;
  QnnDetachedExecution& operator=(const QnnDetachedExecution&) = delete;

  executorch::runtime::Error execute(
      executorch::runtime::Span<executorch::runtime::EValue*> args) const;

 private:
  friend std::unique_ptr<QnnDetachedExecution>
  AcquireLastInitializedQnnExecution(const char* method_name);

  QnnDetachedExecution(
      const QnnExecuTorchBackend* backend,
      executorch::runtime::DelegateHandle* handle,
      std::string method_name);

  const QnnExecuTorchBackend* backend_{nullptr};
  executorch::runtime::DelegateHandle* handle_{nullptr};
  std::string method_name_;
};

// Consume the most recent QNN delegate initialized on the calling thread and
// retain it independently of the ExecuTorch Method that created it.
std::unique_ptr<QnnDetachedExecution> AcquireLastInitializedQnnExecution(
    const char* method_name);

// Initializes the process-lifetime QNN backend/device bundle from an AOT
// compile spec. This deliberately creates no QNN context, graph, or tensors.
executorch::runtime::Error PrewarmQnnBackend(
    const void* qnn_compile_spec_data,
    size_t qnn_compile_spec_size);

// Release process-global backend/device resources after all delegate contexts
// have been destroyed and before switching to a CPU-only phase.
void ReleaseQnnBackendBundles();

// Release HTP performance votes but retain backend/device objects.
void ReleaseQnnPerformanceVotes();

class QnnExecuTorchBackend final
    : public ::executorch::runtime::BackendInterface {
 public:
  ~QnnExecuTorchBackend(){};

  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec>
          compile_specs) const override;

  executorch::runtime::Error execute(
      ET_UNUSED executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::Span<executorch::runtime::EValue*> args)
      const override;

  ET_NODISCARD executorch::runtime::Error set_option(
      executorch::runtime::BackendOptionContext& context,
      const executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  executorch::runtime::Error get_option(
      executorch::runtime::BackendOptionContext& context,
      executorch::runtime::Span<executorch::runtime::BackendOption>&
          backend_options) override;

  void destroy(executorch::runtime::DelegateHandle* handle) const override;

  bool is_available() const override;

 private:
  friend class QnnDetachedExecution;
  friend std::unique_ptr<QnnDetachedExecution>
  AcquireLastInitializedQnnExecution(const char* method_name);

  struct DelegateOwnership {
    std::int64_t signature{0};
    size_t method_refs{0};
    size_t detached_refs{0};
  };

  bool retain_detached_delegate(
      executorch::runtime::DelegateHandle* handle) const;
  void release_detached_delegate(
      executorch::runtime::DelegateHandle* handle) const;
  void add_cached_delegate(
      const std::int64_t& signature,
      executorch::runtime::DelegateHandle* handle) const;
  void retain_cached_method_delegate(
      executorch::runtime::DelegateHandle* handle) const;

  mutable std::mutex mutex_;
  mutable std::mutex runtime_option_mutex_;
  mutable std::unordered_map<int64_t, executorch::runtime::DelegateHandle*>
      delegate_map_;
  mutable std::unordered_map<executorch::runtime::DelegateHandle*, std::int64_t>
      delegate_map_rev_;
  mutable std::unordered_map<
      executorch::runtime::DelegateHandle*,
      DelegateOwnership>
      delegate_ownership_;

  RuntimeOption qnn_runtime_log_level_{false, 0};
  RuntimeOption qnn_runtime_performance_mode_{false, 0};
  RuntimeOption qnn_runtime_profile_level_{false, 0};
};

} // namespace qnn
} // namespace backends
} // namespace executorch
