# Source Provenance and Cleanup Scope

All paths in the source column are relative to the original `EDFA1` directory. The three recovered model files were copied byte-for-byte; their hashes are recorded in `SOURCE_MANIFEST.json`.

| Release path | Original source | Role |
| --- | --- | --- |
| `models/wide_cnn.py` | `EDFA_network/result/transfor.py` | Wide soft-threshold CNN accepting 85 features |
| `models/residual_attention_cnn.py` | `EDFA_network/model_NN.py` | Residual attention soft-threshold CNN accepting 85 features |
| `models/mlp.py` | `EDFA_network/result/model_NN.py` | Fully connected network accepting 85 features |

The original directory contains different models named `model_NN.py` in separate folders. The model factory exposes descriptive names for each implementation.

The model files retain their original definitions, parameter names and initialization order. See `MODELS.md` for their active forward paths.

## Newly Added Files

`run.py`, `data.py`, `models/__init__.py`, documentation and repository metadata were added during repository preparation. Model definitions are preserved from the source paths listed above.

The runner supports configurable paths and devices, initializes each fold independently, saves final-epoch checkpoints and aggregates global prediction errors. It uses Adam with a configurable constant learning rate and consumes prepared data.

## Left in the Original Working Directory

| Original area | Reason not included in the primary release |
| --- | --- |
| `pool.py`, root `test.py` | Earlier experiment entry points; the release uses `run.py` |
| `EDFA_network/result/train_EDFA_non.py` | Earlier MLP experiment entry point; the release uses `run.py` |
| Other `EDFA_network/` scripts and system datasets | Cascade/system or embedding experiments outside the paper's single-EDFA scope |
| `NEW/` | Wavelength augmentation, Mixup, domain adaptation and transfer-learning experiments |
| Root processing/plotting scripts | Local data-processing and plotting utilities |
| CSV/NPY data, PTH/PT weights, result figures | Local datasets, model checkpoints and generated outputs |
| IDE directories and caches | Machine-specific generated files |

Exclusion from the release does not delete or modify these local materials. The publisher PDF is a comparison source, not a redistributed repository file.
