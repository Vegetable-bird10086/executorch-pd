/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/aot/wrappers/TensorWrapper.h>
#include <executorch/backends/qualcomm/qc_compiler_spec_generated.h>
#include <executorch/backends/qualcomm/runtime/QnnBackendOptions.h>
#include <executorch/backends/qualcomm/runtime/QnnExecuTorchBackend.h>
#include <executorch/backends/qualcomm/runtime/QnnManager.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnCustomProtocol.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/backend/options.h>

#include <cstdlib>
namespace executorch {
namespace backends {
namespace qnn {

using namespace qnn_delegate;
using executorch::runtime::ArrayRef;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

namespace {

int ParseIndexedQnnTensorName(const std::string& name, const char* prefix) {
  const std::string prefix_str(prefix);
  if (name.rfind(prefix_str, 0) != 0) {
    return -1;
  }
  size_t pos = prefix_str.size();
  if (pos >= name.size() || name[pos] < '0' || name[pos] > '9') {
    return -1;
  }
  int value = 0;
  while (pos < name.size() && name[pos] >= '0' && name[pos] <= '9') {
    value = value * 10 + static_cast<int>(name[pos] - '0');
    ++pos;
  }
  if (pos >= name.size() || name[pos] != '_') {
    return -1;
  }
  return value;
}

} // namespace

// ========== Public method implementations =========================
constexpr const char* QNN_COMPILE_SPEC = "qnn_compile_spec";
Result<DelegateHandle*> QnnExecuTorchBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  // covert SizedBuffer to qnn ExecuTorch option
  QnnExecuTorchContextBinary qnn_context_blob;
  const qnn_delegate::QnnExecuTorchOptions* qnn_executorch_options = nullptr;
  auto [status, signature, ctx_size, ctx_bin] =
      QnnContextCustomProtocol().DeserializeContextCustomBuffer(
          const_cast<void*>(processed->data()));
  if (status == Error::Ok) {
    QNN_EXECUTORCH_LOG_INFO(
        "Deserializing processed data using QnnContextCustomProtocol");
    // After this stage, qnn_context_blob.nbytes & qnn_context_blob.buffer will
    // only store qnn_context_binary.
    qnn_context_blob.nbytes = ctx_size;
    qnn_context_blob.buffer = ctx_bin;
  } else {
    // This buffer will be verified again in QnnBackendCache.
    QNN_EXECUTORCH_LOG_INFO("Deserializing processed data using Dlc");
    qnn_context_blob.buffer = const_cast<void*>(processed->data());
    qnn_context_blob.nbytes = processed->size();
  }

  // convert CompileSpec to qnn ExecuTorch option
  for (auto& compile_spec : compile_specs) {
    if (std::strcmp(compile_spec.key, QNN_COMPILE_SPEC) == 0)
      qnn_executorch_options =
          GetQnnExecuTorchOptions(compile_spec.value.buffer);
    else
      QNN_EXECUTORCH_LOG_WARN("unknown argument: %s", compile_spec.key);
  }

  // Create QnnManager
  MemoryAllocator* runtime_allocator = context.get_runtime_allocator();
  QnnManager* qnn_manager = runtime_allocator->allocateInstance<QnnManager>();
  if (qnn_manager == nullptr) {
    return Error::MemoryAllocationFailed;
  }

  // NOTE: Since we use placement new and since this type is not trivially
  // destructible, we must call the destructor manually in destroy().
  new (qnn_manager) QnnManager(qnn_executorch_options, qnn_context_blob);
  // TODO: this is a temporal solution for multi-graph support, will be
  //       removed once framework starts to accept runtime configuration
  // ---
  // check if current context binary has already been initialized
  // return cached one for reducing memory footprint

  auto iter = delegate_map_.find(signature);
  if (iter != delegate_map_.end()) {
    QNN_EXECUTORCH_LOG_INFO(
        "Use cached delegate handle for current method: %s",
        context.get_method_name());
    return iter->second;
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitBackend() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->InitContext() == Error::Ok,
      Internal,
      "Fail to initialize Qnn Manager");

  if (qnn_manager->IsOnlinePrepare()) {
    ET_CHECK_OR_RETURN_ERROR(
        qnn_manager->CompileDlc() == Error::Ok,
        Internal,
        "Fail to compile binary in Dlc format");
  } else {
    for (const std::string& graph_name : qnn_manager->GetGraphNames()) {
      ET_CHECK_OR_RETURN_ERROR(
          qnn_manager->AllocateTensor(graph_name) == Error::Ok,
          Internal,
          "Fail to allocate tensor");
    }
  }
  add_cached_delegate(signature, qnn_manager);

#ifndef __hexagon__
  // This backend does not need its processed data after Init.
  processed->Free();
#endif

  return qnn_manager;
}

Error QnnExecuTorchBackend::execute(
    BackendExecutionContext& context,
    DelegateHandle* handle,
    Span<EValue*> args) const {
  ET_CHECK_OR_RETURN_ERROR(
      delegate_map_rev_.count(handle) != 0,
      Internal,
      "DelegateHandle has been deleted");
  QnnManager* qnn_manager = static_cast<QnnManager*>(handle);

  std::string method_name = context.get_method_name();
  const std::vector<std::string> graph_names = qnn_manager->GetGraphNames();
  bool graph_name_exists = false;
  for (const std::string& graph_name : graph_names) {
    if (graph_name == method_name) {
      graph_name_exists = true;
      break;
    }
  }
  if (!graph_name_exists && graph_names.size() == 1) {
    QNN_EXECUTORCH_LOG_WARN(
        "method name %s does not match QNN graph name %s; using graph name",
        method_name.c_str(),
        graph_names[0].c_str());
    method_name = graph_names[0];
  }
  std::vector<std::shared_ptr<TensorWrapper>> input_tensors =
      qnn_manager->GetGraphInputs(method_name);
  std::vector<std::shared_ptr<TensorWrapper>> output_tensors =
      qnn_manager->GetGraphOutputs(method_name);
  std::vector<Qnn_Tensor_t> input_tensor_structs;
  std::vector<Qnn_Tensor_t> output_tensor_structs;
  const bool binding_log_enabled =
      std::getenv("EXECUTORCH_QNN_BINDING_LOG") != nullptr;

  int args_index = 0;
  input_tensor_structs.reserve(input_tensors.size());
  for (const auto& input_tensor : input_tensors) {
    if (input_tensor->GetName().find("mutbuf_") == std::string::npos) {
      int matched_args_index = args_index;
      const int indexed_args_index =
          ParseIndexedQnnTensorName(input_tensor->GetName(), "input_");
      if (indexed_args_index >= 0 &&
          indexed_args_index < static_cast<int>(args.size())) {
        matched_args_index = indexed_args_index;
      } else if (indexed_args_index >= 0) {
        QNN_EXECUTORCH_LOG_WARN(
            "falling back to sequential input binding for graph %s tensor %s parsed_index=%d args=%zu",
            method_name.c_str(),
            input_tensor->GetName().c_str(),
            indexed_args_index,
            args.size());
      }
      if (binding_log_enabled && method_name == "prefill_forward") {
        const auto& arg_tensor = args[matched_args_index]->toTensor();
        QNN_EXECUTORCH_LOG_INFO(
            "qnn bind input graph=%s tensor=%s qnn_seq=%d parsed=%d arg=%d arg_rank=%d arg_dims=%d,%d,%d,%d qnn_rank=%u qnn_dims=%u,%u,%u,%u",
            method_name.c_str(),
            input_tensor->GetName().c_str(),
            args_index,
            indexed_args_index,
            matched_args_index,
            arg_tensor.dim(),
            arg_tensor.dim() > 0 ? static_cast<int>(arg_tensor.size(0)) : -1,
            arg_tensor.dim() > 1 ? static_cast<int>(arg_tensor.size(1)) : -1,
            arg_tensor.dim() > 2 ? static_cast<int>(arg_tensor.size(2)) : -1,
            arg_tensor.dim() > 3 ? static_cast<int>(arg_tensor.size(3)) : -1,
            input_tensor->GetRank(),
            input_tensor->GetRank() > 0 ? input_tensor->GetDims()[0] : 0,
            input_tensor->GetRank() > 1 ? input_tensor->GetDims()[1] : 0,
            input_tensor->GetRank() > 2 ? input_tensor->GetDims()[2] : 0,
            input_tensor->GetRank() > 3 ? input_tensor->GetDims()[3] : 0);
      }
      if (qnn_manager->RegisterMem(
              args[matched_args_index]->toTensor().mutable_data_ptr(), input_tensor) !=
          Error::Ok) {
        // update data ptr only should be fine
        input_tensor->FillDataBuffer(
            args[matched_args_index]->toTensor().const_data_ptr(),
            false /* copy_data */);
        // use the real input shape instead of nominal one to make sure
        // dynamic shape is functional
        auto dims = args[matched_args_index]->toTensor().sizes();
        input_tensor->SetDims(dims.data(), dims.size());
      }
      args_index++;
    }
    input_tensor_structs.emplace_back(input_tensor->CloneTensorStruct());
  }

  const int output_args_base = args_index;
  std::vector<bool> output_arg_used(args.size(), false);
  auto tensor_shape_matches = [](const EValue* value,
                                 const std::shared_ptr<TensorWrapper>& tensor) {
    const auto& et_tensor = value->toTensor();
    if (static_cast<std::uint32_t>(et_tensor.dim()) != tensor->GetRank()) {
      return false;
    }
    const std::uint32_t* qnn_dims = tensor->GetDims();
    for (int i = 0; i < et_tensor.dim(); ++i) {
      if (static_cast<std::uint32_t>(et_tensor.size(i)) != qnn_dims[i]) {
        return false;
      }
    }
    return true;
  };

  int output_value_index = 0;
  for (const auto& output_tensor : output_tensors) {
    // pos=0 limits the search to the prefix
    if (output_tensor->GetName().rfind("output_", 0) == 0 &&
        output_tensor->GetName().find("mutbuf_") == std::string::npos) {
      int matched_args_index = -1;
      for (int i = output_args_base; matched_args_index < 0 &&
           i < static_cast<int>(args.size()); ++i) {
        if (!output_arg_used[i] && tensor_shape_matches(args[i], output_tensor)) {
          matched_args_index = i;
          break;
        }
      }
      if (matched_args_index < 0) {
        matched_args_index = args_index;
        QNN_EXECUTORCH_LOG_WARN(
            "falling back to sequential output binding for graph %s tensor %s",
            method_name.c_str(),
            output_tensor->GetName().c_str());
      }
      output_arg_used[matched_args_index] = true;
      if (binding_log_enabled && method_name == "prefill_forward") {
        const auto& arg_tensor = args[matched_args_index]->toTensor();
        QNN_EXECUTORCH_LOG_INFO(
            "qnn bind output graph=%s tensor=%s qnn_seq=%d arg=%d arg_rank=%d arg_dims=%d,%d,%d,%d qnn_rank=%u qnn_dims=%u,%u,%u,%u",
            method_name.c_str(),
            output_tensor->GetName().c_str(),
            output_value_index,
            matched_args_index,
            arg_tensor.dim(),
            arg_tensor.dim() > 0 ? static_cast<int>(arg_tensor.size(0)) : -1,
            arg_tensor.dim() > 1 ? static_cast<int>(arg_tensor.size(1)) : -1,
            arg_tensor.dim() > 2 ? static_cast<int>(arg_tensor.size(2)) : -1,
            arg_tensor.dim() > 3 ? static_cast<int>(arg_tensor.size(3)) : -1,
            output_tensor->GetRank(),
            output_tensor->GetRank() > 0 ? output_tensor->GetDims()[0] : 0,
            output_tensor->GetRank() > 1 ? output_tensor->GetDims()[1] : 0,
            output_tensor->GetRank() > 2 ? output_tensor->GetDims()[2] : 0,
            output_tensor->GetRank() > 3 ? output_tensor->GetDims()[3] : 0);
      }
      void* mutable_data_ptr =
          args[matched_args_index]->toTensor().mutable_data_ptr();
      if (qnn_manager->RegisterMem(mutable_data_ptr, output_tensor) !=
          Error::Ok) {
        output_tensor->FillDataBuffer(mutable_data_ptr, false /* copy_data */);
      }
      args_index++;
      output_value_index++;
    }
    output_tensor_structs.push_back(output_tensor->CloneTensorStruct());
  }

  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->Execute(
          method_name,
          input_tensor_structs,
          output_tensor_structs,
          context.event_tracer()) == Error::Ok,
      Internal,
      "Fail to execute graph");
  ET_CHECK_OR_RETURN_ERROR(
      qnn_manager->ProfileExecuteData(method_name, context.event_tracer()) ==
          Error::Ok,
      Internal,
      "Fail to profile graph");

  return Error::Ok;
}

void QnnExecuTorchBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr && delegate_map_rev_.count(handle)) {
    QnnManager* qnn_manager = static_cast<QnnManager*>(handle);
    qnn_manager->Destroy();
    erase_cached_delegate(handle);
  }
}

executorch::runtime::Error QnnExecuTorchBackend::set_option(
    executorch::runtime::BackendOptionContext& context,
    const executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  std::lock_guard<std::mutex> guard(runtime_option_mutex_);
  size_t matches = backend_options.size();
  for (const auto& option : backend_options) {
    if (strcmp(option.key, QNN_RUNTIME_LOG_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_log_level_.value = *val;
        qnn_runtime_log_level_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_performance_mode_.value = *val;
        qnn_runtime_performance_mode_.is_set = true;
      }
    } else if (strcmp(option.key, QNN_RUNTIME_PROFILE_LEVEL) == 0) {
      if (auto* val = std::get_if<int>(&option.value)) {
        qnn_runtime_profile_level_.value = *val;
        qnn_runtime_profile_level_.is_set = true;
      }
    } else {
      ET_LOG(
          Error,
          "Unable to set the following runtime option for QnnExecuTorchBackend: %s.",
          option.key);
      matches--;
    }
  }

  ET_CHECK_OR_RETURN_ERROR(
      matches == backend_options.size(),
      Internal,
      "Some set options are not supported by QnnExecuTorchBackend. %zu options provided but only %zu is supported.",
      backend_options.size(),
      matches);

  return Error::Ok;
}

executorch::runtime::Error QnnExecuTorchBackend::get_option(
    executorch::runtime::BackendOptionContext& context,
    executorch::runtime::Span<executorch::runtime::BackendOption>&
        backend_options) {
  size_t matches = backend_options.size();
  for (size_t i = 0; i < backend_options.size(); ++i) {
    // Set the value to what was stored by set_option
    if (strcmp(backend_options[i].key, QNN_RUNTIME_LOG_LEVEL) == 0 &&
        qnn_runtime_log_level_.is_set) {
      backend_options[i].value = qnn_runtime_log_level_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_HTP_PERFORMANCE_MODE) == 0 &&
        qnn_runtime_performance_mode_.is_set) {
      backend_options[i].value = qnn_runtime_performance_mode_.value;
    } else if (
        strcmp(backend_options[i].key, QNN_RUNTIME_PROFILE_LEVEL) == 0 &&
        qnn_runtime_profile_level_.is_set) {
      backend_options[i].value = qnn_runtime_profile_level_.value;
    } else {
      // either runtime never called set_option or key does not exist
      matches--;
    }
  }

  if (matches != backend_options.size()) {
    return Error::Internal;
  }
  return Error::Ok;
}

bool QnnExecuTorchBackend::is_available() const {
  return true;
}

void QnnExecuTorchBackend::add_cached_delegate(
    const std::int64_t& signature,
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  delegate_map_[signature] = handle;
  delegate_map_rev_[handle] = signature;
}

void QnnExecuTorchBackend::erase_cached_delegate(
    executorch::runtime::DelegateHandle* handle) const {
  std::lock_guard<std::mutex> guard(mutex_);
  auto iter = delegate_map_rev_.find(handle);
  if (iter == delegate_map_rev_.end()) {
    return;
  }
  delegate_map_.erase(iter->second);
  delegate_map_rev_.erase(handle);
}

namespace {
auto cls = QnnExecuTorchBackend();
executorch::runtime::Backend backend{QNN_BACKEND, &cls};
static auto success_with_compiler = register_backend(backend);
} // namespace
} // namespace qnn
} // namespace backends
} // namespace executorch
