#!/bin/bash
# Runs on the pod after build and download. One engine run per keep_tokens value.
set -euo pipefail
cd /workspace/DeepInfer
export DEEPINFER_ACCELERATOR=nvidia REAL_PROMPTS=/workspace/prompts.json
MODEL=/workspace/models/DeepSeek-V4.1-Flash-1017
EP=${EP:-$(nvidia-smi -L | wc -l)}
# DeepEP reads Engram rows over RDMA (NCCL Gin), which needs an IB device even on one
# node. Without one: MoE dispatch over NVLink only, and the local Engram gather from
# patch_single_node.py, with every rank holding the whole table in host RAM.
if ! ls /dev/infiniband/uverbs* >/dev/null 2>&1; then
  python3 /workspace/patch_single_node.py /workspace/DeepInfer
  export EP_DISABLE_GIN=1 REAL_LOCAL_ENGRAM=1 REAL_ENGRAM_PARALLEL=1 DEEPINFER_ENFORCE_EAGER=1
  echo "no RDMA device: NVLink-only EP, local Engram gather, eager"
fi
mkdir -p var/real
for K in ${KEEPS:-0 140 512}; do
  echo "=== keep_tokens=$K  $(date -u +%H:%M:%S) ==="
  # The loader reads single-threaded; pull the shards into page cache with 16 readers.
  (cd "$MODEL" && ls *.safetensors | xargs -P 16 -I{} cat {} > /dev/null) &
  python examples/yoco/yoco_real.py logp --model_name "$MODEL" --keep-tokens "$K" --compact 1 \
      --output_len 256 --top_logprobs 20 --max_num_seqs 16 --ep "$EP" --cp 1 --out "var/real/keep${K}.json" \
      2>&1 | tee "/workspace/run_keep${K}.log" | grep -E '^\[logp\]|Error|error|Traceback' || true
  ls var/real/keep${K}.rank0.json >/dev/null 2>&1 || { echo "keep_tokens=$K produced no dump, stopping"; exit 1; }
done
ls -la var/real/
