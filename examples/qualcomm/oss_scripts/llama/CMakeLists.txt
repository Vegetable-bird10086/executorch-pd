# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# model sharding with custom op
set(CUSTOM_OP_SRCS_FILE
    "${EXECUTORCH_SOURCE_DIR}/extension/llm/custom_ops/op_fallback.cpp"
)
set(EXECUTORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../..)

add_library(custom_ops ${CUSTOM_OP_SRCS_FILE})
target_include_directories(custom_ops PUBLIC "${_common_include_directories}")
target_include_directories(
  custom_ops PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../include"
)
target_link_libraries(custom_ops PUBLIC executorch_core full_portable_ops_lib)
executorch_target_link_options_shared_lib(custom_ops)

# preprocess qnn runner src files for llama
set(_llama_runner__srcs ${_llama_runner__srcs})
list(TRANSFORM _llama_runner__srcs PREPEND "${EXECUTORCH_SOURCE_DIR}/")
list(FILTER _llama_runner__srcs EXCLUDE REGEX ".*(/runner/).*")
list(
  PREPEND
  _llama_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/cache_utils.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/decoder_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/decoder_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/prompt_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/prompt_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/separate_embed.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/separate_embed.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/pte_rebuilder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/pte_rebuilder.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/imem_alloc.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/client_mem.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/lhd_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/lhd_token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/attention_sink_rope_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/attention_sink_rope_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/rpc_mem.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/rpc_mem.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/kv_manager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/kv_manager.h
  ${EXECUTORCH_SOURCE_DIR}/examples/models/llama/runner/runner.cpp
  ${EXECUTORCH_SOURCE_DIR}/examples/models/llama/runner/runner.h
)

list(APPEND _llama_runner__srcs)

# build qnn llama runner
add_executable(qnn_llama_runner ${_llama_runner__srcs})
target_include_directories(
  qnn_llama_runner PUBLIC ${_common_include_directories}
)

executorch_target_link_options_shared_lib(quantized_ops_lib)

target_link_libraries(
  qnn_llama_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
  etdump
  flatccrt
)

target_include_directories(
  qnn_llama_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)

target_compile_options(qnn_llama_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# build qnn llama benchmark runner
set(_llama_bench__srcs ${_llama_runner__srcs})
list(FILTER _llama_bench__srcs EXCLUDE REGEX ".*qnn_llama_runner.*")
list(
  PREPEND
  _llama_bench__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_bench.cpp
)

add_executable(qnn_llama_bench ${_llama_bench__srcs})
target_include_directories(
  qnn_llama_bench PUBLIC ${_common_include_directories}
)
target_link_libraries(
  qnn_llama_bench
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
  etdump
  flatccrt
)
target_include_directories(
  qnn_llama_bench PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_compile_options(qnn_llama_bench PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_bench PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# build qnn llama pd runner without touching the original runner entrypoint
set(_llama_pd_runner__srcs ${_llama_runner__srcs})
list(FILTER _llama_pd_runner__srcs EXCLUDE REGEX ".*qnn_llama_runner.*")
list(FILTER _llama_pd_runner__srcs EXCLUDE REGEX ".*runner/runner\.(cpp|h)")
list(
  PREPEND
  _llama_pd_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_pd_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/pd_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/pd_runner.h
)

add_executable(qnn_llama_pd_runner ${_llama_pd_runner__srcs})
target_include_directories(
  qnn_llama_pd_runner PUBLIC ${_common_include_directories}
)
target_include_directories(
  qnn_llama_pd_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_include_directories(
  qnn_llama_pd_runner PUBLIC ${CMAKE_BINARY_DIR}/../../schema/include
)
target_include_directories(
  qnn_llama_pd_runner PUBLIC ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)
target_link_libraries(
  qnn_llama_pd_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
  etdump
  flatccrt
)
target_compile_options(qnn_llama_pd_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_pd_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# build qnn llama end-to-end pd runner that exports handoff then launches
# llama-pd-cli for decode
set(_llama_pd_e2e_runner__srcs ${_llama_pd_runner__srcs})
list(FILTER _llama_pd_e2e_runner__srcs EXCLUDE REGEX ".*qnn_llama_pd_runner.*")
list(
  PREPEND
  _llama_pd_e2e_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_pd_e2e_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/runner.h
)

add_executable(qnn_llama_pd_e2e_runner ${_llama_pd_e2e_runner__srcs})
target_include_directories(
  qnn_llama_pd_e2e_runner PUBLIC ${_common_include_directories}
)
target_include_directories(
  qnn_llama_pd_e2e_runner PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_include_directories(
  qnn_llama_pd_e2e_runner PUBLIC ${CMAKE_BINARY_DIR}/../../schema/include
)
target_include_directories(
  qnn_llama_pd_e2e_runner PUBLIC ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)
target_link_libraries(
  qnn_llama_pd_e2e_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
  etdump
  flatccrt
)
target_compile_options(qnn_llama_pd_e2e_runner PUBLIC ${_common_compile_options})
set_target_properties(
  qnn_llama_pd_e2e_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# Single-process PD runner. llama.cpp pd-cli is compiled as a library entry
# inside this target; no external Decode executable, fork, exec, socket, or FD
# handoff is used by this variant.
set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Path to the llama.cpp source tree")
set(EXECUTORCH_SCHEMA_INCLUDE_DIR "" CACHE PATH "Installed Executorch schema include directory")
if(EXECUTORCH_SCHEMA_INCLUDE_DIR)
  target_include_directories(
    qnn_llama_pd_runner PUBLIC ${EXECUTORCH_SCHEMA_INCLUDE_DIR}
  )
  target_include_directories(
    qnn_llama_pd_e2e_runner PUBLIC ${EXECUTORCH_SCHEMA_INCLUDE_DIR}
  )
endif()
if(LLAMA_CPP_SOURCE_DIR)
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
  set(_pd_joint_saved_build_shared_libs ${BUILD_SHARED_LIBS})
  set(BUILD_SHARED_LIBS OFF)
  add_subdirectory(
    ${LLAMA_CPP_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}/llama-pd-joint
    EXCLUDE_FROM_ALL
  )
  set(BUILD_SHARED_LIBS ${_pd_joint_saved_build_shared_libs})

  set(_llama_pd_joint_runner__srcs ${_llama_pd_e2e_runner__srcs})
  list(FILTER _llama_pd_joint_runner__srcs EXCLUDE REGEX ".*runner/runner\\.(cpp|h)")
  list(FILTER _llama_pd_joint_runner__srcs EXCLUDE REGEX ".*llama_tiktoken\\.cpp")
  list(
    PREPEND
    _llama_pd_joint_runner__srcs
    ${CMAKE_CURRENT_LIST_DIR}/qnn_llama_pd_joint_runner.cpp
    ${LLAMA_CPP_SOURCE_DIR}/tools/pd-cli/pd_cli.cpp
  )
  add_executable(qnn_llama_pd_joint_runner ${_llama_pd_joint_runner__srcs})
  target_compile_definitions(
    qnn_llama_pd_joint_runner PRIVATE
    QNN_LLAMA_PD_JOINT=1
    QNN_LLAMA_PD_E2E_NO_MAIN=1
    LLAMA_PD_CLI_NO_MAIN=1
  )
  target_include_directories(
    qnn_llama_pd_joint_runner PUBLIC
    ${_common_include_directories}
    ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
    ${CMAKE_BINARY_DIR}/../../schema/include
    ${EXECUTORCH_SCHEMA_INCLUDE_DIR}
    ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
    ${LLAMA_CPP_SOURCE_DIR}/include
    ${LLAMA_CPP_SOURCE_DIR}/common
    ${LLAMA_CPP_SOURCE_DIR}/src
    ${LLAMA_CPP_SOURCE_DIR}/tools/pd-cli
    ${LLAMA_CPP_SOURCE_DIR}/vendor/nlohmann
  )
  target_link_libraries(
    qnn_llama_pd_joint_runner
    qnn_executorch_backend
    executorch_core
    extension_data_loader
    extension_flat_tensor
    extension_llm_runner
    extension_module
    extension_tensor
    gflags
    custom_ops
    quantized_ops_lib
    quantized_kernels
    etdump
    flatccrt
    llama-common
    llama
    ${CMAKE_THREAD_LIBS_INIT}
  )
  target_compile_options(
    qnn_llama_pd_joint_runner PUBLIC ${_common_compile_options}
  )
  set_target_properties(
    qnn_llama_pd_joint_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
  )
endif()

# build PTE rebuild comparison tool
add_executable(
  pte_rebuild_compare
  ${CMAKE_CURRENT_LIST_DIR}/pte_rebuild_compare.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/pte_rebuilder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/pte_rebuilder.h
)
target_include_directories(
  pte_rebuild_compare PUBLIC ${_common_include_directories}
)
target_include_directories(
  pte_rebuild_compare PUBLIC ${EXECUTORCH_ROOT}/third-party/flatbuffers/include
)
target_link_libraries(
  pte_rebuild_compare
  executorch_core
  extension_data_loader
  extension_module
  extension_tensor
)
target_compile_options(pte_rebuild_compare PUBLIC ${_common_compile_options})
set_target_properties(
  pte_rebuild_compare PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)

# build qnn multimodal runner preprocess qnn runner src files for multimodal
set(_multimodal_runner__srcs ${_llama_runner__srcs})
list(FILTER _multimodal_runner__srcs EXCLUDE REGEX ".*qnn_llama_runner.*")
list(FILTER _multimodal_runner__srcs EXCLUDE REGEX ".*runner/runner\.(cpp|h)")
list(
  PREPEND
  _multimodal_runner__srcs
  ${CMAKE_CURRENT_LIST_DIR}/qnn_multimodal_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_embedding_merger.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_embedding_merger.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/utils.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/encoder.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/encoder.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_runner.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_runner.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/tok_embedding_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_prompt_processor.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_prompt_processor.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_token_generator.h
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_lhd_token_generator.cpp
  ${CMAKE_CURRENT_LIST_DIR}/runner/multimodal_runner/multimodal_lhd_token_generator.h
)

list(APPEND _multimodal_runner__srcs)

# build qnn multimodal runner
add_executable(qnn_multimodal_runner ${_multimodal_runner__srcs})
target_include_directories(
  qnn_multimodal_runner PUBLIC ${_common_include_directories}
)
target_include_directories(
  qnn_multimodal_runner
  PUBLIC ${EXECUTORCH_ROOT}/extension/llm/tokenizers/include
)
target_compile_options(qnn_multimodal_runner PUBLIC ${_common_compile_options})

target_link_libraries(
  qnn_multimodal_runner
  qnn_executorch_backend
  executorch_core
  extension_data_loader
  extension_flat_tensor
  extension_llm_runner
  extension_module
  extension_tensor
  gflags
  custom_ops
  quantized_ops_lib
  quantized_kernels
  tokenizers::tokenizers
)

set_target_properties(
  qnn_multimodal_runner PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'"
)
