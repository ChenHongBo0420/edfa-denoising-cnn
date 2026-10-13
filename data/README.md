# Data Source and Prepared CSV Contract

## Original Experimental Dataset

The paper cites the following reused dataset in reference [6]:

M. P. Yankov and F. Da Ros (2020), *Input-output power spectral densities for three C-band EDFAs and four multi-span inline EDFA'd fiber optic systems of different lengths*, DTU, version 1. DOI: [10.11583/DTU.13135754.v1](https://doi.org/10.11583/DTU.13135754.v1).

The ECOC paper's experiments use three individual booster EDFAs, A1/A2/A3, rather than the multi-span system experiments. The paper describes 83 equally spaced channels and output-power settings of 15, 16, 17 and 18 dBm. See paper page 2 (printed page 283).

Obtain data and its metadata from the original repository and follow its terms. Code and data have separate licences. Prepare data in the following layout before using the runner.

## Accepted Layout

The recovered training scripts use zero-based slices `X = array[:, 1:86]` and `y = array[:, 86:]`. The new reader requires exactly 169 columns:

| Zero-based columns | Number | Role |
| --- | --- | --- |
| `0` | 1 | Auxiliary/index column; ignored, not assumed to be physical gain |
| `1:3` | 2 | Total input/output power fields used by the recovered code |
| `3:86` | 83 | Prepared input-spectrum features |
| `86:169` | 83 | Prepared output-spectrum targets |

The complete 85-dimensional input is passed to the model unchanged. Values must be finite numerical values. Rows and spectral channels must be consistently aligned. A header is skipped only when the user explicitly passes `--skip-header`.

The original working directory contains headerless examples such as `intra-EDFA/TEST/EDFA_1_17dB.csv` and `intra-EDFA/TEST/Gain/EDFA1_cleaned.csv`.

## Raw Data Is Different

The local `monolithDatabaseRawEDFA.csv` has 208 columns, including a textual header beginning with `EDFAId, profileId, targetPout, pInIdx, measuredTotalPin, measuredTotalPout`. It is not a direct input to this runner.

The paper refers to preprocessing from its reference [5]. This runner starts from prepared CSVs. When preparing raw data, use consistent spectral channels, normalization and units for training and evaluation, and retain any transformation parameters needed to interpret the outputs.

Metrics are reported in prepared target units. Apply the corresponding inverse transformation when reporting results in physical units.

## Experiment Separation

- Intra-EDFA: five independent folds within A1, grouped by output-power setting as in Fig. 3(a-d).
- Inter-EDFA: train on A1 and evaluate the unchanged checkpoint on A2/A3 using the same preprocessing parameters.
- Gain-resolved plots use total-power/gain metadata; output-power settings define the experiment groups.

Place local inputs under `data/processed/` or pass any external path. Data files, weights and run outputs are excluded by `.gitignore`.
