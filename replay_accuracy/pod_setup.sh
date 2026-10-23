#!/bin/bash
# Runs on the pod. Starts the checkpoint download and the DeepInfer build in parallel.
set -euo pipefail
cd /workspace
# ssh sessions do not inherit the container's environment; take RUNPOD_* from PID 1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(POD_ID|API_KEY)=' | sed 's/^/export /')"
export PATH=/usr/local/cuda/bin:$PATH

# 0. hard stop: the pod deletes itself after MAX_HOURS, whatever state the run is in
[ -f /workspace/watchdog.started ] || nohup bash -c "sleep $(( ${MAX_HOURS:-6} * 3600 )); runpodctl remove pod \$RUNPOD_POD_ID || \
  curl -s -X DELETE https://rest.runpod.io/v1/pods/\$RUNPOD_POD_ID -H \"Authorization: Bearer \$RUNPOD_API_KEY\"" \
  > /workspace/watchdog.log 2>&1 &
touch /workspace/watchdog.started
echo "watchdog: pod ${RUNPOD_POD_ID:-?} removes itself in ${MAX_HOURS:-6}h; runpodctl=$(command -v runpodctl || echo missing)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvcc --version | tail -2
python3 --version; gcc --version | head -1
python3 -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.device_count())"
free -g | head -2; df -h /workspace | tail -1; nproc
ls /dev/infiniband 2>&1 | head -3 || true

# 1. checkpoint, in its own venv so the build's pinned packages cannot disturb it
python3 -m venv /workspace/dlenv
/workspace/dlenv/bin/pip install -q huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 nohup /workspace/dlenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('deepseek-ai/DeepSeek-V4.1-Flash-1017', local_dir='/workspace/models/DeepSeek-V4.1-Flash-1017', max_workers=32)
print('DOWNLOAD DONE')" > /workspace/download.log 2>&1 &

# 2. DeepInfer at the commit the posts describe. DeepSelect's setup calls lspci, and
#    DeepEP probes ibstat, links libibverbs, and its DeepJIT needs elfutils headers; the
#    base image has none of them.
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq pciutils rdma-core libibverbs-dev infiniband-diags libdw-dev libelf-dev libnuma-dev libunwind-dev >/dev/null
git clone -q https://github.com/deepseek-ai/DeepInfer /workspace/DeepInfer
cd /workspace/DeepInfer && git checkout -q 9bd9cf2 && git log --oneline -1
# the image's torch lives in the system Python, which Ubuntu 24.04 marks externally managed
# Under CUDA 13.2 one FlashMLA SM100 decode kernel spills registers, which FlashMLA treats
# as a build failure. Spills cost speed, not correctness, and this run measures accuracy.
# DeepInfer's setup.py only checks that VIRTUAL_ENV or CONDA_PREFIX is set.
export DEEPINFER_ACCELERATOR=nvidia PIP_BREAK_SYSTEM_PACKAGES=1 FLASH_MLA_SKIP_REG_SPILL_CHECK=1 VIRTUAL_ENV=/usr
nohup bash -c "./build.sh && echo BUILD DONE" > /workspace/build.log 2>&1 &
echo "started download and build"
