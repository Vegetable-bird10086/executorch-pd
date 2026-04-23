#!/usr/bin/env bash

# Source this file:
#   source executorch/examples/qualcomm/oss_scripts/llama/pd_stack_env.sh

set -euo pipefail

if ! (return 0 2>/dev/null); then
  echo "Please source this file instead of executing it:"
  echo "  source executorch/examples/qualcomm/oss_scripts/llama/pd_stack_env.sh"
  exit 1
fi

_pd_stack_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXECUTORCH_ROOT="$(cd "${_pd_stack_script_dir}/../../../../.." && pwd)"
export PD_STACK_ROOT="$(cd "${EXECUTORCH_ROOT}/.." && pwd)"

if [[ -z "${TMAC_ROOT:-}" ]]; then
  export TMAC_ROOT="${PD_STACK_ROOT}/T-MAC"
fi

export EXECUTORCH_LLAMA_ROOT="${EXECUTORCH_ROOT}/examples/qualcomm/oss_scripts/llama"
export TMAC_LLAMA_CPP_ROOT="${TMAC_ROOT}/3rdparty/llama.cpp"
export TMAC_LLAMA_PD_CLI_X86="${TMAC_LLAMA_CPP_ROOT}/build/bin/llama-pd-cli"
export TMAC_LLAMA_PD_CLI_ANDROID="${TMAC_LLAMA_CPP_ROOT}/build-android/bin/llama-pd-cli"

echo "PD stack environment ready:"
echo "  EXECUTORCH_ROOT=${EXECUTORCH_ROOT}"
echo "  TMAC_ROOT=${TMAC_ROOT}"
echo "  TMAC_LLAMA_CPP_ROOT=${TMAC_LLAMA_CPP_ROOT}"
echo "  TMAC_LLAMA_PD_CLI_X86=${TMAC_LLAMA_PD_CLI_X86}"
echo "  TMAC_LLAMA_PD_CLI_ANDROID=${TMAC_LLAMA_PD_CLI_ANDROID}"
