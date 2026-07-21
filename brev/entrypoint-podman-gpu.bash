#! /bin/bash
set -euo pipefail

link_driver_library() {
  local library=${1}
  local soname=${2}
  local lib
  local dir

  for lib in "/usr/lib/"*-linux-gnu/"${library}".so.* /usr/lib64/"${library}".so.*; do
    if [ -e "${lib}" ]; then
      dir=$(dirname "${lib}")
      ln -sf "$(basename "${lib}")" "${dir}/${soname}" 2>/dev/null || true
      break
    fi
  done
}

link_driver_library libcuda libcuda.so.1
link_driver_library libnvidia-ml libnvidia-ml.so.1
link_driver_library libnvidia-ptxjitcompiler libnvidia-ptxjitcompiler.so.1
link_driver_library libnvidia-nvvm libnvidia-nvvm.so.4

if [ -n "${ACH_NVIDIA_LIB_DIRS:-}" ]; then
  export LD_LIBRARY_PATH="${ACH_NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

exec /accelerated-computing-hub/brev/entrypoint.bash "$@"
