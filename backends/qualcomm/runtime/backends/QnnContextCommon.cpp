/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/backends/qualcomm/runtime/backends/QnnContextCommon.h>
#include <executorch/backends/qualcomm/runtime/backends/QnnDlcManager.h>

#include "HTP/QnnHtpContext.h"

#include <chrono>
#include <condition_variable>
#include <mutex>

namespace executorch {
namespace backends {
namespace qnn {

namespace {
using Clock = std::chrono::steady_clock;

double elapsed_ms(const Clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace

QnnContext::~QnnContext() {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;
  if (handle_ != nullptr) {
    QNN_EXECUTORCH_LOG_INFO("Destroy Qnn context");
    error = qnn_interface.qnn_context_free(handle_, /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to free QNN "
          "context_handle_. Backend "
          "ID %u, error %d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
    }
    handle_ = nullptr;
  }
}

Error QnnContext::Configure() {
  // create qnn context
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ErrorHandle_t error = QNN_SUCCESS;

  std::vector<const QnnContext_Config_t*> temp_context_config;
  ET_CHECK_OR_RETURN_ERROR(
      MakeConfig(temp_context_config) == Error::Ok,
      Internal,
      "Fail to make context config.");

  if (cache_->GetCacheState() == QnnBackendCache::DESERIALIZE) {
    const QnnExecuTorchContextBinary& qnn_context_blob =
        cache_->GetQnnContextBlob();

    const auto context_restore_start = Clock::now();
    error = qnn_interface.qnn_context_create_from_binary(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        static_cast<uint8_t*>(qnn_context_blob.buffer),
        qnn_context_blob.nbytes,
        &handle_,
        /*profile=*/nullptr);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't create context from "
          "binary. Error %d.",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
    QNN_EXECUTORCH_LOG_INFO(
        "QNN context timing: context_create_from_binary_ms=%.3f",
        elapsed_ms(context_restore_start));
  } else if (
      cache_->GetCacheState() == QnnBackendCache::SERIALIZE ||
      cache_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE ||
      cache_->GetCacheState() == QnnBackendCache::MULTI_GRAPH) {
    error = qnn_interface.qnn_context_create(
        backend_->GetHandle(),
        device_->GetHandle(),
        temp_context_config.empty() ? nullptr : temp_context_config.data(),
        &handle_);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Failed to create QNN context for Backend "
          "ID %u, error=%d",
          qnn_interface.GetBackendId(),
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR("QNN context cache is invalid.");
    return Error::Internal;
  }
  if (AfterConfigure() != Error::Ok) {
    return Error::Internal;
  }
  if (cache_->GetCacheState() == QnnBackendCache::ONLINE_PREPARE) {
    // Register graphs from DLC during online prepare for HTP/GPU/DSP backends
    return qnn_dlc_manager_->RegisterGraphsFromDLC(
        implementation_, backend_, this, cache_);
  }
  return Error::Ok;
}

Error QnnContext::ConfigureDeserializeBatch(
    const std::vector<QnnContext*>& contexts) {
  if (contexts.empty()) {
    return Error::InvalidArgument;
  }

  QnnContext* first = contexts.front();
  const QnnInterface& qnn_interface = first->implementation_->GetQnnInterface();
  if (qnn_interface.GetInterfaceVer().contextCreateFromBinaryListAsync ==
      nullptr) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN provider does not expose contextCreateFromBinaryListAsync");
    return Error::NotSupported;
  }

  struct BatchState {
    std::mutex mutex;
    std::condition_variable cv;
    size_t pending{0};
    Qnn_ErrorHandle_t first_error{QNN_SUCCESS};
  } state;
  struct NotifyState {
    QnnContext* context{nullptr};
    BatchState* batch{nullptr};
    bool context_notified{false};
  };

  auto notify = [](Qnn_ContextHandle_t context_handle,
                   Qnn_GraphHandle_t,
                   const char*,
                   QnnContext_createFromBinaryAsyncNotifyType_t notify_type,
                   void* notify_param,
                   Qnn_ErrorHandle_t status) {
    auto* item = static_cast<NotifyState*>(notify_param);
    if (notify_type != QNN_CONTEXT_NOTIFY_TYPE_CONTEXT_INIT) {
      if (status != QNN_SUCCESS) {
        std::lock_guard<std::mutex> guard(item->batch->mutex);
        if (item->batch->first_error == QNN_SUCCESS) {
          item->batch->first_error = status;
        }
      }
      return;
    }
    std::lock_guard<std::mutex> guard(item->batch->mutex);
    if (item->context_notified) {
      return;
    }
    item->context_notified = true;
    if (status == QNN_SUCCESS) {
      item->context->handle_ = context_handle;
    } else if (item->batch->first_error == QNN_SUCCESS) {
      item->batch->first_error = status;
    }
    if (item->batch->pending > 0) {
      --item->batch->pending;
    }
    item->batch->cv.notify_all();
  };

  std::vector<std::vector<const QnnContext_Config_t*>> per_context_configs(
      contexts.size());
  std::vector<std::unique_ptr<QnnContext_Params_t>> params;
  std::vector<const QnnContext_Params_t*> param_ptrs;
  std::vector<NotifyState> notify_states(contexts.size());
  params.reserve(contexts.size());
  param_ptrs.reserve(contexts.size() + 1);
  state.pending = contexts.size();

  for (size_t i = 0; i < contexts.size(); ++i) {
    QnnContext* context = contexts[i];
    if (context == nullptr || context->handle_ != nullptr ||
        context->implementation_ != first->implementation_ ||
        context->backend_->GetHandle() != first->backend_->GetHandle() ||
        context->device_->GetHandle() != first->device_->GetHandle() ||
        context->cache_->GetCacheState() != QnnBackendCache::DESERIALIZE) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Invalid or incompatible context in QNN batch restore at index %zu",
          i);
      return Error::InvalidArgument;
    }
    ET_CHECK_OR_RETURN_ERROR(
        context->MakeConfig(per_context_configs[i]) == Error::Ok,
        Internal,
        "Fail to make context config for batch index %zu",
        i);
    const QnnExecuTorchContextBinary& blob =
        context->cache_->GetQnnContextBlob();
    notify_states[i] = {context, &state, false};
    auto param = std::unique_ptr<QnnContext_Params_t>(
        new QnnContext_Params_t{QNN_CONTEXT_PARAMS_VERSION_1,
                                {{per_context_configs[i].empty()
                                      ? nullptr
                                      : per_context_configs[i].data(),
                                  blob.buffer,
                                  blob.nbytes,
                                  nullptr,
                                  notify,
                                  &notify_states[i]}}});
    param_ptrs.push_back(param.get());
    params.push_back(std::move(param));
  }
  param_ptrs.push_back(nullptr);

  const auto restore_start = Clock::now();
  const Qnn_ErrorHandle_t submit_error =
      qnn_interface.qnn_context_create_from_binary_list_async(
          first->backend_->GetHandle(),
          first->device_->GetHandle(),
          param_ptrs.data(),
          /*list_config=*/nullptr,
          /*signal=*/nullptr);
  if (submit_error != QNN_SUCCESS) {
    QNN_EXECUTORCH_LOG_ERROR(
        "QNN batch context restore submission failed. Error %d",
        QNN_GET_ERROR_CODE(submit_error));
    return Error::Internal;
  }

  {
    std::unique_lock<std::mutex> lock(state.mutex);
    const bool completed = state.cv.wait_for(
        lock, std::chrono::minutes(5), [&state] { return state.pending == 0; });
    if (!completed) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Timed out waiting for %zu QNN batch context callback(s)",
          state.pending);
      return Error::Internal;
    }
    if (state.first_error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "QNN batch context restore callback failed. Error %d",
          QNN_GET_ERROR_CODE(state.first_error));
      return Error::Internal;
    }
  }

  for (QnnContext* context : contexts) {
    ET_CHECK_OR_RETURN_ERROR(
        context->handle_ != nullptr && context->AfterConfigure() == Error::Ok,
        Internal,
        "Failed to finalize a QNN batch-restored context");
  }
  QNN_EXECUTORCH_LOG_INFO(
      "QNN batch context restore timing: contexts=%zu total_ms=%.3f "
      "resource_sharing=disabled",
      contexts.size(),
      elapsed_ms(restore_start));
  return Error::Ok;
}

Error QnnContext::GetContextBinary(
    QnnExecuTorchContextBinary& qnn_executorch_context_binary) {
  const QnnInterface& qnn_interface = implementation_->GetQnnInterface();
  Qnn_ContextBinarySize_t binary_size = 0;
  Qnn_ContextBinarySize_t bytes_written = 0;
  Qnn_ErrorHandle_t error =
      qnn_interface.qnn_context_get_binary_size(handle_, &binary_size);
  if (error == QNN_SUCCESS) {
    // create our own protocol here
    qnn_context_custom_protocol_ = QnnContextCustomProtocol(binary_size);
    qnn_context_custom_protocol_.BuildContextCustomBuffer();
    auto [context_buffer_ptr, context_buffer_size] =
        qnn_context_custom_protocol_.GetCustomProtocolBuffer();
    error = qnn_interface.qnn_context_get_binary(
        handle_,
        static_cast<uint8_t*>(context_buffer_ptr) +
            qnn_context_custom_protocol_.GetContextBinaryOffset(),
        binary_size,
        &bytes_written);
    if (error != QNN_SUCCESS) {
      QNN_EXECUTORCH_LOG_ERROR(
          "Can't get graph binary to be saved to "
          "cache. Error %d",
          QNN_GET_ERROR_CODE(error));
      return Error::Internal;
    } else {
      if (binary_size < bytes_written) {
        QNN_EXECUTORCH_LOG_ERROR(
            "Illegal written buffer size [%d] bytes. Cannot "
            "exceed allocated memory of [%d] bytes",
            bytes_written,
            binary_size);
        return Error::Internal;
      }

      qnn_executorch_context_binary.buffer = context_buffer_ptr;
      qnn_executorch_context_binary.nbytes = context_buffer_size;
    }
  } else {
    QNN_EXECUTORCH_LOG_ERROR(
        "Can't determine the size of "
        "graph binary to be saved to cache. Error %d",
        QNN_GET_ERROR_CODE(error));
    return Error::Internal;
  }
  return Error::Ok;
}
} // namespace qnn
} // namespace backends
} // namespace executorch
