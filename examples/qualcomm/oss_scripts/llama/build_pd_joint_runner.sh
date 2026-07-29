#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
executorch_root="$(cd -- "${script_dir}/../../../.." && pwd)"
llama_cpp_root="${LLAMA_CPP_ROOT:-${executorch_root}/../llama.cpp}"
executorch_install="${EXECUTORCH_INSTALL_DIR:-${executorch_root}/build-android}"
build_dir="${1:-${executorch_root}/build-android-pd-joint}"
android_ndk="${ANDROID_NDK:-${executorch_root}/../android-ndk-r27d}"
qnn_sdk_root="${QNN_SDK_ROOT:-${executorch_root}/../qairt/2.37.0.250724}"

if [[ ! -f "${executorch_install}/lib/cmake/ExecuTorch/executorch-config.cmake" ]]; then
  echo "error: ${executorch_install} does not contain an installed Executorch Android package" >&2
  echo "build/install the normal Qualcomm Executorch libraries first" >&2
  exit 2
fi
if [[ ! -f "${llama_cpp_root}/tools/pd-cli/pd_cli.cpp" ]]; then
  echo "error: LLAMA_CPP_ROOT does not point to the paired llama.cpp tree" >&2
  exit 2
fi

cmake -S "${executorch_root}/examples/qualcomm" -B "${build_dir}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${android_ndk}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=30 \
  -DBUILD_SHARED_LIBS=OFF \
  -DQNN_SDK_ROOT="${qnn_sdk_root}" \
  -DCMAKE_PREFIX_PATH="${executorch_install};${executorch_install}/lib/cmake/ExecuTorch;${executorch_install}/third-party/gflags" \
  -Dexecutorch_DIR="${executorch_install}/lib/cmake/ExecuTorch" \
  -Dgflags_DIR="${executorch_install}/third-party/gflags" \
  -Dabsl_DIR="${executorch_install}/lib/cmake/absl" \
  -Dtokenizers_DIR="${executorch_install}/lib/cmake/tokenizers" \
  -Dre2_DIR="${executorch_install}/lib/cmake/re2" \
  -Dpcre2_DIR="${executorch_install}/lib/cmake/pcre2" \
  -DEXECUTORCH_SCHEMA_INCLUDE_DIR="${executorch_install}/schema/include" \
  -DLLAMA_CPP_SOURCE_DIR="${llama_cpp_root}" \
  -DGGML_OPENMP=OFF \
  -DGGML_CPU_ALL_VARIANTS=OFF \
  -DGGML_CPU_ARM_ARCH=armv8.6-a+dotprod+i8mm \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_COMMON=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_CURL=OFF

cmake --build "${build_dir}" \
  --target qnn_llama_pd_joint_runner \
  --parallel "${BUILD_JOBS:-4}"

echo "${build_dir}/oss_scripts/llama/qnn_llama_pd_joint_runner"
