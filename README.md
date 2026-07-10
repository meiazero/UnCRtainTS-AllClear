# UnCRtainTS on AllClear

![banner](architecture.png)

Fork of [UnCRtainTS](https://github.com/PatrickTUM/UnCRtainTS) trimmed down to train and evaluate on
the [AllClear](https://github.com/Zhou-Hangyu/allclear) dataset. The network, the losses and the
metrics are unchanged from upstream; only the data pipeline and the entry point differ.

> P. Ebel, V. Garnot, M. Schmitt, J. Wegner and X. X. Zhu. UnCRtainTS: Uncertainty Quantification for
> Cloud Removal in Optical Satellite Time Series. CVPRW, 2023.

---

## Installation

```bash
git clone https://github.com/meiazero/UnCRtainTS-AllClear.git
cd UnCRtainTS-AllClear
uv sync                    # creates .venv from pyproject.toml + uv.lock
uv sync --extra profile    # additionally installs fvcore (FLOP counting, --profile)
```

Python 3.12, PyTorch >= 2.13. CUDA strongly recommended.

## Dataset

Get the AllClear repo and download the ROIs you need:

```bash
git clone https://github.com/Zhou-Hangyu/allclear.git
cd allclear
python download.py --from-json metadata/datasets/train_tx3_s2-s1_10pct.json --dry-run
```

Splits live in `metadata/datasets/`. UnCRtainTS uses `input_t=3` time steps with S2 + S1
(13 + 2 = 15 input channels), so the matching splits are the `tx3_s2-s1` family:

| Split | File | Samples | With real SAR |
|---|---|---|---|
| train | `train_tx3_s2-s1_10pct.json` (also `_1pct`, `_3.4pct`, `_100pct`) | 27861 | 60.5% |
| val   | `val_tx3_s2-s1-landsat_100pct.json` | 14212 | 62.8% |
| test  | `test_tx3_s2-s1_100pct.json` | 55317 | 59.5% |

Take the `_s2-s1_` splits, **not** the `_s2_` ones. `_s2_` means "100% of the S2 sequences", not
"S2 only": those files cover the very same samples but with an empty `s1` list, and `AllClearDataset`
fills missing auxiliary sensors with a constant-1.0 placeholder. Training on real SAR and evaluating
against a fabricated one produces no error, only wrong numbers. `data/allclear_dataset.py` rejects a
split with zero SAR coverage for this reason.

The remaining ~40% of samples have no temporally aligned SAR and get the same placeholder. That is
native AllClear behavior, identical to the official inference wrapper.

There is no `val_tx3_s2-s1_100pct.json`; the landsat variant is the tx3 val split. That is fine —
the adapter requests only `s2_toa` and `s1`, so the landsat entries are never read.

`data/allclear_dataset.py` wraps `allclear.dataset.AllClearDataset` and emits the
`(input, target, masks, dates)` tuple the training loop expects. The tensor mapping mirrors the
official inference wrapper (`allclear/baseline_wrappers.py::UnCRtainTS`), so training and the
AllClear benchmark feed the network identically.

## Training

```bash
uv run python train_allclear.py \
    --experiment_name allclear_v1 \
    --allclear_repo        /path/to/allclear \
    --allclear_root        /path/to/allclear/data \
    --allclear_train_split /path/to/allclear/metadata/datasets/train_tx3_s2-s1_10pct.json \
    --allclear_val_split   /path/to/allclear/metadata/datasets/val_tx3_s2-s1-landsat_100pct.json \
    --allclear_test_split  /path/to/allclear/metadata/datasets/test_tx3_s2-s1_100pct.json \
    --num_workers 4
```

`--allclear_repo` is only needed when `allclear` is not importable from the active environment; it
puts `allclear/dataset.py` on the import path without pulling in the rest of the package.

Every flag not passed keeps the default UnCRtainTS-on-AllClear setting (`--lr 0.001`,
`--scale_by 10.0`, `--use_sar`, `--loss MGNLL`, `--covmode diag`, `--batch_size 4`, `--epochs 20`).
See `model/parse_args.py` for the full list.

Results, checkpoints and `conf.json` land under `--res_dir` (default `./results`). The `conf.json` is
what the official AllClear benchmark reads back to reconstruct the model, so keep it with the weights.

---

## References

```bibtex
@inproceedings{UnCRtainTS,
        title = {{UnCRtainTS: Uncertainty Quantification for Cloud Removal in Optical Satellite Time Series}},
        author = {Ebel, Patrick and Garnot, Vivien Sainte Fare and Schmitt, Michael and Wegner, Jan and Zhu, Xiao Xiang},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
        year = {2023},
        organization = {IEEE},
        url = {"https://openaccess.thecvf.com/content/CVPR2023W/EarthVision/papers/Ebel_UnCRtainTS_Uncertainty_Quantification_for_Cloud_Removal_in_Optical_Satellite_Time_CVPRW_2023_paper.pdf"}
}
```

## Credits

Originally based on [UTAE](https://github.com/VSainteuf/utae-paps) and
[SEN12MS-CR-TS](https://github.com/PatrickTUM/SEN12MS-CR-TS). AllClear integration follows
[Zhou-Hangyu/allclear](https://github.com/Zhou-Hangyu/allclear).
