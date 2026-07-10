"""Adapter over the official `allclear.AllClearDataset`.

Emits the 4-tuple `(x, y, masks, dates)` that ACinterface_main_0515.py::iterate()
unpacks for `--dataset ALLCLEAR`. The tensor mapping mirrors the official inference
wrapper (allclear/baseline_wrappers.py::UnCRtainTS.preprocess/forward), so training
and the AllClear benchmark feed the network identically.
"""

import importlib.util
import json
import os
import sys

import torch
from torch.utils.data import Dataset

_IMPORT_HINT = (
    "Could not import `allclear`. Either install the AllClear repo into this venv\n"
    "  uv add --editable /path/to/allclear\n"
    "or pass --allclear_repo /path/to/allclear"
)


def _allclear_dataset_cls(repo_path=""):
    # load allclear/dataset.py directly: it imports only stdlib + numpy/rasterio/torch,
    # while allclear/__init__.py drags in matplotlib and appends cwd to sys.path.
    if repo_path:
        module_path = os.path.join(repo_path, "allclear", "dataset.py")
        if os.path.isfile(module_path):
            spec = importlib.util.spec_from_file_location("_allclear_dataset", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.AllClearDataset
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        from allclear.dataset import AllClearDataset
    except ImportError as exc:
        raise ImportError(_IMPORT_HINT) from exc
    return AllClearDataset


class AllClearReconstruct(Dataset):
    """Wraps AllClearDataset for UnCRtainTS reconstruction training.

    Args:
        split_json: path to a metadata/datasets/*.json split (must be a tx{N} split
            whose N matches `tx`, and must carry a "target" key, i.e. seq2point).
        data_root: directory holding the extracted roiXXXX/ trees.
        tx: number of input time steps; must equal config.input_t.
        repo_path: path to the AllClear repo, if `allclear` is not already importable.
    """

    def __init__(self, split_json, data_root, tx=3, repo_path="", selected_rois="all"):
        allclear_dataset = _allclear_dataset_cls(repo_path)
        with open(split_json) as f:
            dataset = json.load(f)

        name = os.path.basename(split_json)

        # AllClearDataset sizes its tensor at tx rows but writes one row per s2_toa
        # timestamp, so a tx12 split under --input_t 3 dies with IndexError on the
        # first __getitem__. Fail here instead, with the split named.
        steps = {len(sample["s2_toa"]) for sample in dataset.values()}
        if steps != {tx}:
            raise ValueError(
                f"{name} has {sorted(steps)} s2_toa time steps but --input_t is {tx}. "
                f"Use a tx{tx} split, or set --input_t accordingly."
            )

        # Missing SAR is not an error to AllClearDataset: it leaves those channels at
        # the torch.ones placeholder. The `_s2_` splits carry an s1 key that is empty
        # for every sample, so the model would silently see constant-1.0 SAR. That is
        # a wrong number, not a crash, so refuse it here.
        with_sar = sum(1 for sample in dataset.values() if sample.get("s1"))
        if not with_sar:
            raise ValueError(
                f"{name} has no SAR observations at all; UnCRtainTS reads 13 S2 + 2 SAR "
                f"channels, so all {len(dataset)} samples would carry a constant-1.0 "
                f"placeholder. Use the matching `_s2-s1_` split instead."
            )
        print(f"{name}: {len(dataset)} samples, {100*with_sar/len(dataset):.1f}% with real SAR")

        self.dataset = allclear_dataset(
            dataset=dataset,
            selected_rois=selected_rois,
            aux_sensors=["s1"],      # -> input_images has 13 S2 + 2 SAR = 15 channels
            aux_data=["cld_shdw"],   # ponytail: skip "dw", nothing downstream reads it
            tx=tx,
            target_mode="s2p",
            data_root=data_root,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        x = sample["input_images"].permute(1, 0, 2, 3)              # (T, 15, H, W)
        y = sample["target"].permute(1, 0, 2, 3)                    # (1, 13, H, W)
        masks = torch.clip(sample["input_cld_shdw"].sum(dim=0), 0, 1)  # (T, H, W)
        dates = sample["time_differences"]                          # (T,) days since start
        return x, y, masks, dates
