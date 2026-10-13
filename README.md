# EDFA Gain Modelling with Soft-Threshold Denoising

Research code associated with **Accurate EDFA Gain Modelling using Convolutional Neural Network with Denoising Layers enabled by Soft-thresholding**, Hongbo Chen, Guo-Wei Lu and Kunping Luo, ECOC 2023, pp. 282-285. [Paper DOI](https://doi.org/10.1049/icp.2023.2060).

This repository provides single-EDFA modelling code with adaptive soft-threshold denoising, including wide-channel and residual-attention CNN variants and a fully connected baseline. It preserves the model definitions and provides a portable training/evaluation entry point. See [model descriptions](docs/MODELS.md) and [source provenance](docs/SOURCE_MAP.md).

## Contents

| File | Purpose |
| --- | --- |
| `models/wide_cnn.py` | Wide-channel soft-threshold CNN |
| `models/residual_attention_cnn.py` | Residual CNN with soft thresholding and additional attention |
| `models/mlp.py` | 85-256-128-83 fully connected baseline |
| `run.py` | New path-configurable training, independent cross-validation and evaluation runner |
| `data.py` | Reader for prepared 169-column numerical CSVs |
| `data/README.md` | Dataset source and input column schema |

Cascade/system experiments, semantic embeddings, wavelength-augmented inputs, domain adaptation, duplicate plotting scripts, existing checkpoints and generated figures are not included in the main release. Their source files remain in the original working directory.

## Environment

Use Python 3.10 or newer in a separate environment:

```shell
python -m venv .venv
```

Activate it using `.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on Linux/macOS, then:

```shell
python -m pip install -r requirements.txt
python run.py --help
```

Dependency ranges are listed in `requirements.txt`. Select the runtime device with `--device cpu`, `cuda`, or `auto`.

## Data

The paper uses the DTU experimental dataset by M. P. Yankov and F. Da Ros: [dataset DOI, version 1](https://doi.org/10.11583/DTU.13135754.v1). Raw data and local processed CSVs are not bundled or relicensed by this code repository.

The runner uses a **prepared, numeric, normally headerless CSV with exactly 169 columns**. The first column is ignored; the next 85 are model inputs; the last 83 are targets. Prepare raw data in this layout using a consistent preprocessing pipeline. See [data instructions](data/README.md).

## Usage

The following commands show example settings for user-supplied, consistently preprocessed data. Adjust the hyperparameters to your experiment. Each output directory must be new.

Train on an A1 dataset:

```shell
python run.py train --data data/processed/A1_15dBm.csv --model wide-cnn --epochs 500 --lr 0.001 --output runs/A1-15dBm
```

Five-fold evaluation on A1, with a fresh model and optimizer for each fold:

```shell
python run.py cross-validate --data data/processed/A1_15dBm.csv --model wide-cnn --folds 5 --epochs 500 --output runs/cv-A1-15dBm
```

Evaluate the same A1-trained checkpoint on A2 without retraining:

```shell
python run.py evaluate --data data/processed/A2_15dBm.csv --checkpoint runs/A1-15dBm/model.pt --output runs/test-A2-15dBm
```

Repeat evaluation on A3 using that same checkpoint. For other output-power levels, prepare matching data and separate runs. `15/16/17/18 dBm` denote output-power settings.

Use `--model residual-attention-cnn` or `--model mlp` to select the other models. Use `--skip-header` for a file with a textual header. Add `--save-predictions` to evaluation to export predictions in the same row/channel order as the input CSV.

## Runner Details

- The model files are copied unchanged; their original state-dictionary names are retained.
- The new runner uses Adam and `sqrt(MSE + 1e-6)` as its training loss, with no learning-rate scheduler, augmentation or normalization.
- The final epoch is saved and used for held-out evaluation.
- Each fold saves its train/test row indices, configuration, history and final checkpoint. The input CSV is identified by SHA-256.
- Evaluation computes `sqrt(sum(error^2) / (samples * 83))` and global MAE.
- Metrics use the units of the supplied target values.
- Each run records its seed, device and library versions.
- Evaluation loads the labelled checkpoints produced by this runner.

## Citation and Licence

```bibtex
@inproceedings{chen2023edfa,
  author = {Chen, Hongbo and Lu, Guo-Wei and Luo, Kunping},
  title = {Accurate EDFA Gain Modelling using Convolutional Neural Network with Denoising Layers enabled by Soft-thresholding},
  booktitle = {49th European Conference on Optical Communications (ECOC 2023)},
  year = {2023},
  pages = {282--285},
  doi = {10.1049/icp.2023.2060}
}
```

Code: [MIT licence](LICENSE), following the author's selected licensing preference. The paper and third-party dataset retain their own terms. The publisher PDF is not distributed here.
