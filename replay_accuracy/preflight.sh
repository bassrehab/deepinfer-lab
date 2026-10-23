#!/bin/bash
# Runs on the pod once DeepEP is built, while the rest of the build continues.
# V4.1-Flash reads its Engram tables through DeepEP's EngramBuffer, which starts NCCL
# Gin (GDAKI) even when every rank is on one node. If DeepEP's own Engram test fails
# here, the engine needs patch_single_node.py, which in turn needs MoE dispatch to work
# with Gin off. If that fails too, stop before paying for the rest.
set -uo pipefail
ls /dev/infiniband 2>&1 | head -3
cd /workspace/DeepInfer/vendor/DeepEP
N=$(nvidia-smi -L | wc -l)
timeout 900 python tests/engram/test_engram.py --num-processes "$N" --use-fp8 --allow-hybrid-mode 2>&1 | tail -25
echo "PREFLIGHT ENGRAM EXIT ${PIPESTATUS[0]}"
# If that failed for want of Gin, check that MoE dispatch at least runs over NVLink alone,
# which is what patch_single_node.py relies on.
EP_DISABLE_GIN=1 timeout 900 python tests/ep/test_ep.py --num-processes "$N" --test-first-only --skip-perf-test \
    --num-tokens 512 --hidden 4096 --num-experts 64 2>&1 | tail -5
echo "PREFLIGHT EP_NO_GIN EXIT ${PIPESTATUS[0]}"
