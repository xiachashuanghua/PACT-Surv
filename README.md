# PACT-Surv

**Knowledge-guided survival prediction with clinical–RNA integration and constrained cross-cancer transfer.**

PACT-Surv integrates language-model-derived gene knowledge with clinical and transcriptomic data for cancer prognosis. It connects explicit gene selection to patient-level risk estimation, while drawing on shared representations learned across cancer types.

## Method overview

- **Knowledge-guided RNA selection.** Cancer-specific gene preferences guide both survival-informed feature screening and coefficient regularization.
- **Joint clinical–RNA estimation.** Clinical and selected RNA features are fitted together in a Cox model, with an agreement penalty linking their risk contributions.
- **Constrained cross-cancer transfer.** Neural networks pretrained on other cancer types provide a correction to the clinical–RNA risk. Adaptation is regularized toward the source networks and selected using training-only cross-validation.
- **Patient-level interpretation.** Predictions can be decomposed into clinical, selected-RNA and neural contributions, with integrated gradients identifying gene contributions to the transfer change.

The study evaluates LUAD, LGG, KIRC, BRCA and LIHC using ten outer partitions per cohort. Source pretraining draws on 33 TCGA cancer types and excludes the target cancer for each analysis. Further details are provided in the [method and evaluation protocol](docs/protocol.md).

## Installation

Requires Python 3.10 and PyTorch. Install the dependencies in a Python environment:

```bash
git clone https://github.com/xiachashuanghua/PACT-Surv.git
cd PACT-Surv
python -m pip install -r requirements.txt
```

Use a PyTorch build compatible with your CPU or CUDA environment.

## Data preparation

Inputs consist of clinical features, candidate RNA expression, gene preference scores, high-dimensional RNA expression and survival outcomes. Prepare one file per target cohort at `data/{cancer}.npz`, following the [data format guide](docs/data_format.md).

Supply the pretraining source pool as `sources/pancan_all.npz`. Source and target RNA matrices must share the same high-dimensional gene coordinates and ordering. Patient data and pretrained weights are not included in this repository.

## Training and evaluation

The following example pretrains source networks with LUAD excluded, then fits the LUAD models across ten outer partitions:

```bash
python train.py --prepare-sources --cancer luad --data-dir data --source-dir sources
python train.py --train --cancer luad --data-dir data --source-dir sources --out-dir results
```

Repeat for `lgg`, `kirc`, `brca` and `lihc`. After all five cohorts have been fitted, evaluate their test predictions:

```bash
python train.py --evaluate --data-dir data --source-dir sources --out-dir results
```

Evaluation reports C-index and pooled horizon AUC, with means and sample standard deviations across the ten partitions. The [selected model configurations](configs/paper_selected_models.json) are provided for reference.

## Prediction and interpretation

Use a trained model together with its `selection.json` and corresponding source-network weights:

```bash
python predict.py --cancer luad --split 1 --data-dir data --source-dir sources \
  --checkpoint results/luad/split_01/model.pkl --out outputs/luad_split01.csv --ig
```

This exports test-patient risk predictions, evaluation metrics and gene-level integrated gradients. Risk decomposition preserves the sign of each contribution and distinguishes the neural reference from the patient-specific transfer change. Omit `--ig` for prediction and evaluation alone.

For input specifications, model definitions and validation details, see the [data guide](docs/data_format.md), [protocol](docs/protocol.md) and [validation report](docs/validation.md).
