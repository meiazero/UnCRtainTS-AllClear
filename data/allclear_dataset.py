"""Standardize AllClearDataset items into the batch the UnCRtainTS seam expects.

AllClearDataset (allclear.dataset) honors idx deterministically and carries real
per-frame timestamps. Its item_dict is channel-first ([C, T, H, W]); UnCRtainTS
wants time-first ([T, C, H, W]) plus a collapsed occlusion mask and target-relative
dates. This module does exactly that translation and nothing else.

Per-sample output (default_collate stacks the leading batch axis):
  A                 : [T, C, H, W]  input reflectance; C = 13, or 15 with --use_sar
  B                 : [1, 13, H, W] clean target (s2p, single frame)
  masks             : [T, H, W]     union cloud|shadow per input frame (1 = occluded)
  dates             : [T]           signed day-offset of each input to the target
  target_valid_mask : [1, H, W]     complement of the target's cloud|shadow
  sample_id         : str           AllClear data_id, for paired evaluation
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from itertools import pairwise
from pathlib import Path

import torch
from torch.utils.data import Dataset

SECONDS_PER_DAY = 86400.0

_IMPORT_HINT = (
    "Could not import `allclear`. Either install the AllClear repo into this venv\n"
    "  uv add --editable /path/to/allclear\n"
    "or pass the repo path as `dataset_repo`."
)


def _allclear_dataset_cls(dataset_repo: str = ""):
    """Load allclear/dataset.py by path: it needs only stdlib + numpy/rasterio/torch,
    while allclear/__init__.py drags in matplotlib and appends cwd to sys.path."""
    if dataset_repo:
        module_path = os.path.join(dataset_repo, "allclear", "dataset.py")
        if os.path.isfile(module_path):
            spec = importlib.util.spec_from_file_location("_allclear_dataset", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.AllClearDataset
        if dataset_repo not in sys.path:
            sys.path.insert(0, dataset_repo)
    try:
        from allclear.dataset import AllClearDataset
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    return AllClearDataset


def _union_cld_shdw(cld_shdw: torch.Tensor) -> torch.Tensor:
    """[2, T, H, W] (clouds_30, shadows_thres_30) -> [T, H, W] occlusion, nan->1."""
    m = torch.nan_to_num(cld_shdw, nan=1.0)
    return (m > 0.5).any(dim=0).to(torch.float32)


class _Standardize(Dataset):
    """Wrap an AllClearDataset so __getitem__ yields the standardized dict above."""

    def __init__(self, base):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, index: int) -> dict:
        it = self._base[index]
        a = it["input_images"].permute(1, 0, 2, 3).contiguous()  # [C,T,H,W] -> [T,C,H,W]
        b = it["target"].permute(1, 0, 2, 3).contiguous()  # [C,1,H,W] -> [1,C,H,W]
        masks = _union_cld_shdw(it["input_cld_shdw"])  # [T,H,W]
        # signed day-offset of each input frame to the target (target at 0)
        dates = (it["timestamps"] - it["target_timestamps"]) / SECONDS_PER_DAY  # [T]
        tgt_occ = _union_cld_shdw(it["target_cld_shdw"])  # [1,H,W]
        return {
            "A": a,
            "B": b,
            "masks": masks,
            "dates": dates.to(torch.float32),
            "target_valid_mask": (1.0 - tgt_occ),
            "sample_id": str(it["data_id"]),
        }


def _load_split_dict(path: Path) -> dict:
    """Split JSON as an order-frozen {sample_id: entry} (sorted keys => run-invariant idx)."""
    with Path(path).open() as fh:
        raw = json.load(fh)
    return {k: raw[k] for k in sorted(raw)}


def _assert_frame_order(split: dict, split_name: str) -> None:
    """Fail loud if any sample's s2_toa frames are not in ascending-timestamp order.

    AllClearDataset sorts the IMAGE frames by timestamp but stacks the cld_shdw masks
    in raw JSON order (dataset.py:238 vs :250). If a sample's s2_toa list isn't already
    ascending, mask frame k would correspond to a different date than image/date frame
    k — a silent mis-supervision. The timestamps are ISO "%Y-%m-%d %H:%M:%S", which sort
    lexicographically, so a plain string compare is exact and needs no image tree.
    """
    for sid, entry in split.items():
        ts = [t for t, _ in entry.get("s2_toa", [])]
        if any(a > b for a, b in pairwise(ts)):
            raise ValueError(
                f"{split_name} sample {sid}: s2_toa frames not ascending by timestamp "
                f"({ts}); mask/image frame order would misalign — re-sort the split JSON."
            )


def _assert_frame_count(split: dict, split_name: str, tx: int) -> None:
    """AllClearDataset sizes its tensor at tx rows but writes one row per s2_toa frame,
    so a tx12 split under tx=3 raises IndexError inside the first __getitem__."""
    steps = {len(entry["s2_toa"]) for entry in split.values()}
    if steps != {tx}:
        raise ValueError(
            f"{split_name} has {sorted(steps)} s2_toa time steps but tx is {tx}. "
            f"Use a tx{tx} split, or set tx accordingly."
        )


def _assert_sar_present(split: dict, split_name: str) -> None:
    """With use_sar the model reads 13 S2 + 2 SAR channels, and AllClearDataset leaves a
    missing auxiliary sensor at its torch.ones placeholder. A split whose `s1` lists are
    all empty (what make_s2_only.py produces) would feed a constant 1.0 — wrong numbers,
    no error. Samples that individually lack aligned SAR still get the placeholder; that
    is native AllClear behavior."""
    if not any(entry.get("s1") for entry in split.values()):
        raise ValueError(
            f"{split_name} has no SAR observations at all, so every sample would carry a "
            f"constant-1.0 placeholder in the 2 SAR channels. Use the matching `_s2-s1_` "
            f"split, or drop use_sar."
        )


def make_dataset(split_json, data_root, tx=3, use_sar=False, dataset_repo=""):
    """A standardized AllClear s2p dataset over one split, ready for DataLoader."""
    allclear_dataset = _allclear_dataset_cls(dataset_repo)
    split_json = Path(split_json)
    split = _load_split_dict(split_json)

    _assert_frame_count(split, split_json.name, tx)
    _assert_frame_order(split, split_json.name)  # guard mask/image frame alignment
    if use_sar:
        _assert_sar_present(split, split_json.name)

    base = allclear_dataset(
        dataset=split,
        selected_rois="all",
        main_sensor="s2_toa",
        aux_sensors=["s1"] if use_sar else [],
        aux_data=["cld_shdw"],  # need the occlusion mask, skip 'dw' (glob-heavy, unused)
        tx=tx,
        target_mode="s2p",  # seq -> one predefined clear target
        data_root=str(data_root),
    )
    return _Standardize(base)
