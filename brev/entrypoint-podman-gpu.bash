#! /bin/bash
set -euo pipefail

for lib in /usr/lib/x86_64-linux-gnu/libcuda.so.*; do
  if [ -e "${lib}" ]; then
    ln -sf "$(basename "${lib}")" /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null || true
    break
  fi
done

for lib in /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*; do
  if [ -e "${lib}" ]; then
    ln -sf "$(basename "${lib}")" /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1 2>/dev/null || true
    break
  fi
done

for lib in /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.*; do
  if [ -e "${lib}" ]; then
    ln -sf "$(basename "${lib}")" /usr/lib/x86_64-linux-gnu/libnvidia-ptxjitcompiler.so.1 2>/dev/null || true
    break
  fi
done

for lib in /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.*; do
  if [ -e "${lib}" ]; then
    ln -sf "$(basename "${lib}")" /usr/lib/x86_64-linux-gnu/libnvidia-nvvm.so.4 2>/dev/null || true
    break
  fi
done

exec /accelerated-computing-hub/brev/entrypoint.bash "$@"
