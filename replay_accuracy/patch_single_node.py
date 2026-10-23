#!/usr/bin/env python3
"""Let DeepInfer run V4.1-Flash on one cloud node: no RDMA NIC, weights on a network volume.

DeepEP fetches Engram rows with NCCL Gin (GDAKI), and Gin needs an RDMA device even when
every rank sits on one node. MoE dispatch does not: with EP_DISABLE_GIN=1 it runs over
NVLink alone. This patch replaces only the Engram fetch. With engram_parallel_size=1
every rank holds the whole table, so the fetch becomes a plain gather: rows on the CPU,
scale factors on the GPU, returned in the same shapes and layouts DeepEP's hooks return.
It syncs the host on every fetch, so run with DEEPINFER_ENFORCE_EAGER=1.

It also raises the 30-minute checkpoint-load watchdog to three hours: each rank reads
its experts and the whole Engram table single-threaded, and a network volume is far
slower at that than local NVMe.

Nothing in the model math changes. Usage (from the DeepInfer repo root):
  python3 patch_single_node.py .
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])

LOCAL = '''"""Single-node stand-in for DeepEP's EngramBuffer (see patch_single_node.py)."""

import torch


class LocalEngramBuffer:
    is_local = True

    def __init__(self, group, num_entries_per_layer, hidden_size, num_entries_per_token, dtype, num_sf_packs) -> None:
        assert group.size() == 1, "local Engram needs engram_parallel_size=1"
        self.num_entries_per_layer = list(num_entries_per_layer)
        self.row_bytes = hidden_size * dtype.itemsize
        self.dtype = dtype
        self.tables: list[torch.Tensor] = []
        self.sfs: list[torch.Tensor] | None = None

    def set_config(self, *args, **kwargs) -> None:
        pass

    def write(self, storages, sfs=None) -> None:
        self.tables = [t.view(torch.uint8) for t in storages]
        for t, n in zip(self.tables, self.num_entries_per_layer, strict=True):
            assert t.device.type == "cpu" and tuple(t.shape) == (n, self.row_bytes), (t.device, t.shape)
        self.sfs = sfs

    def fetch(self, indices: torch.Tensor, num_qps: int = 0, use_tma_aligned_col_major_sf: bool = True):
        # The engine runs with torch's sync debug mode set to error; this gather has to sync.
        mode = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(0)
        try:
            return self._fetch(indices, use_tma_aligned_col_major_sf)
        finally:
            torch.cuda.set_sync_debug_mode(mode)

    def _fetch(self, indices: torch.Tensor, use_tma_aligned_col_major_sf: bool):
        num_layers, num_tokens, _ = indices.shape
        device = indices.device
        cpu_idx = indices.to("cpu", torch.int64)
        hooks = []
        for layer in range(num_layers):
            rows = self.tables[layer].index_select(0, cpu_idx[layer].reshape(-1))
            data = rows.view(num_tokens, -1).to(device).view(self.dtype)
            if self.sfs is None:
                hooks.append(lambda d=data: d)
                continue
            sf_rows = self.sfs[layer].index_select(0, indices[layer].reshape(-1).to(torch.int64)).view(num_tokens, -1)
            if use_tma_aligned_col_major_sf:
                step = 16 // sf_rows.element_size()
                aligned = (num_tokens + step - 1) // step * step
                buf = torch.empty((sf_rows.shape[1], aligned), dtype=sf_rows.dtype, device=device)
                buf[:, :num_tokens].copy_(sf_rows.t())
                sf = buf[:, :num_tokens].t()
            else:
                sf = sf_rows.contiguous()
            hooks.append(lambda d=data, s=sf: (d, s))
        return hooks

    def barrier(self) -> None:
        pass

    def destroy(self) -> None:
        self.tables, self.sfs = [], None
'''


def sub(path: str, old: str, new: str) -> None:
    p = root / path
    s = p.read_text()
    if new in s:
        return
    assert s.count(old) == 1, f"{path}: anchor not unique or missing: {old[:70]!r}"
    p.write_text(s.replace(old, new))


(root / "deepinfer/model_executor/parallel_utils/local_engram.py").write_text(LOCAL)

sub("deepinfer/model_executor/parallel_utils/userbuffer.py",
    '        assert self.engram is None, "allocate_engram can only be called once"\n',
    '        assert self.engram is None, "allocate_engram can only be called once"\n'
    '        import os as _os\n'
    '        if _os.environ.get("REAL_LOCAL_ENGRAM") == "1":\n'
    '            from deepinfer.model_executor.parallel_utils.local_engram import LocalEngramBuffer\n'
    '\n'
    '            self.engram = LocalEngramBuffer(engram_group, num_entries_per_layer, hidden_size, num_entries_per_token, dtype, num_sf_packs)\n'
    '            logger.info(f"Created local EngramBuffer ({len(num_entries_per_layer)} layers, {num_entries_per_layer=})")\n'
    '            return self.engram\n')

sub("deepinfer/model_executor/models/v4_modules/engram.py",
    "        weights = [cuda_view_for_write(embedding.weight) for embedding in embeddings]\n",
    "        if getattr(self._buffer, \"is_local\", False):\n"
    "            weights = [embedding.weight for embedding in embeddings]\n"
    "        else:\n"
    "            weights = [cuda_view_for_write(embedding.weight) for embedding in embeddings]\n")

sub("deepinfer/model_executor/models/v4_modules/engram.py",
    "        local_sf = _pack_ue8m0_to_int(self.sf)\n",
    "        local_sf = _pack_ue8m0_to_int(self.sf)\n"
    "        if self.engram_group.size() == 1:\n"
    "            return local_sf.contiguous()\n")

sub("deepinfer/model_executor/model_loader.py",
    "def checkpoint_load_watchdog(timeout: int = 30 * 60)",
    "def checkpoint_load_watchdog(timeout: int = 3 * 60 * 60)")

print("patched", root)
