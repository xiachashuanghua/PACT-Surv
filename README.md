# PACT-Surv

PACT-Surv combines language-model-derived gene preferences, a jointly estimated clinical–RNA Cox risk, and constrained cross-cancer neural transfer for cancer prognosis.

The model uses gene preferences in both survival-informed RNA screening and coefficient shrinkage. A clinical–RNA agreement penalty links the two linear components. Five source networks provide a transfer correction; training-only cross-validation selects the source ensemble, adaptation scope, and regularization. Source pretraining excludes the target cancer.

This repository contains the standalone code corresponding to the paper's **target-specific exclusion protocol**, with five target cohorts and ten outer partitions per cohort. The core numerical functions are retained from the frozen research implementation. File loading and command-line entry points have been separated from the surrounding research workspace.

## Installation

Use Python 3.10 in an isolated environment:

```bash
git clone https://github.com/xiachashuanghua/PACT-Surv.git
cd PACT-Surv
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest -v test_core
```

`requirements.txt` records the reference inference versions. Install the appropriate PyTorch CPU/CUDA wheel for your system. CUDA accelerates source and target training; the prediction CLI uses CPU for compatibility with the numerical audit. Set `CUDA_VISIBLE_DEVICES` before starting Python when choosing a GPU.

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Source pretraining, training-only cross-validation, target fitting, and frozen test evaluation |
| `pipeline.py`, `linear.py` | Cohort validation, fold-specific preprocessing, gene screening, and penalized clinical–RNA estimation |
| `pretrain19.py`, `pretrain19b.py` | Source-network architecture, stratified Cox loss, and fitting primitives; use the `train.py` entry point |
| `model.py` | Cox loss and standardized neural head |
| `predict.py` | Frozen-checkpoint inference and optional patient-level risk decomposition |
| `interpret.py` | Gauss–Legendre integrated gradients with a zero-input reference |
| `metrics.py` | Horizon-specific and pooled pair-weighted AUC |
| `prepare_inputs.py` | Conversion of the frozen research input layout into portable cohort files |
| `configs/paper_selected_models.json` | The 50 selected configurations and checkpoint hashes |

See [data format](docs/data_format.md), [method and evaluation protocol](docs/protocol.md), and [code provenance](docs/source_provenance.json).

## Prepare inputs

The repository contains **code and model configurations**. Patient matrices, source-network weights, target checkpoints, private paths, and API credentials are not bundled. Training starts from prepared expression/clinical matrices and fixed gene preference scores; it does not call an LLM API.

For an existing copy of the archived research inputs:

```bash
python prepare_inputs.py --legacy-root /path/to/research_inputs --out-dir data
```

This preserves the original target cohort eligibility, ordered gene coordinates, and ten outer partitions. It does not refit preprocessing or regenerate gene scores. The source input layout is documented in [data_format.md](docs/data_format.md). Alternatively, prepare the same NPZ schema directly.

Each target uses `data/{cancer}.npz`, where `cancer` is `luad`, `lgg`, `kirc`, `brca`, or `lihc`. The pretrained source pool is supplied separately as `sources/pancan_all.npz`. All source and target high-dimensional gene coordinates must have the same order.

## Source pretraining

```bash
python train.py --prepare-sources --cancer luad --data-dir data --source-dir sources
```

Repeat for each target. The code excludes that target from the source pool, verifies patient disjointness, and fits five networks for each of three weight decays. The source pool used for the paper contains 33 TCGA cancer types; each target-specific pool contains the other 32.

Weights are saved under:

```text
sources/{cancer}/p0.5_wd{0.001|0.01|0.1}_lr1e-05/s{0..4}.npz
```

If the matching frozen weights already exist, supply them at those paths. Target training does not implicitly launch source pretraining.

## Train and evaluate

```bash
python train.py --train --cancer luad --data-dir data --source-dir sources --out-dir results
```

The default is all ten outer partitions. `--splits 1` restricts fitting to the first partition; `--linear-only` computes and saves its training-CV linear search. Repeat fitting for the five target cohorts. Training writes configurations, checkpoints, and train/validation predictions. Test predictions are exported after all 50 models are frozen:

```bash
python train.py --evaluate --data-dir data --source-dir sources --out-dir results
```

This evaluation entry point validates the code/input signatures of models trained by this repository. For the archived paper checkpoints, use the inference entry point below: their historical code signatures are different after packaging, while their model parameter format is supported. Do not reuse old CV caches with new signatures.

## Use a frozen checkpoint

Place the trusted `model.pkl` together with its `selection.json`, or supply its trusted `--sha256` explicitly:

```bash
python predict.py --cancer luad --split 1 --data-dir data --source-dir sources \
  --checkpoint /path/to/luad/split_01/model.pkl --out outputs/luad_split01.csv
```

The command verifies checkpoint/source-weight hashes and outer partition indices before predicting. Source files provide each network's fixed output mean/SD; prediction uses the **adapted** network parameters saved in the target checkpoint. An enabled transfer term requires both artifacts. Pickle checkpoints must come from a trusted source.

For patient-level decomposition and gene integrated gradients:

```bash
python predict.py --cancer luad --split 1 --data-dir data --source-dir sources \
  --checkpoint /path/to/luad/split_01/model.pkl --out outputs/luad_split01.csv --ig
```

Use a new output filename for each invocation. IG exports are restricted to outer-test patients and include signed clinical, sparse-RNA, neural-reference and transfer contributions, gene symbols, patient identifiers, and reconstruction residuals. They retain the original FP32 neural endpoints and report quadrature/reconstruction discrepancies without assigning residuals to genes. Across multiple partitions, repeated patients must be deduplicated before patient-level distribution summaries.

## Validation and scope

`test_core.py` checks Breslow gradients, held-out isolation, adaptation/reload, source fitting, IG completeness, metric edge cases, and hash/split validation using synthetic algorithm-test inputs. These arrays are not used as scientific results. Frozen-model replay checks are reported in [validation.md](docs/validation.md).

The repository covers PACT-Surv itself. It does not include the other thirteen comparison methods, manuscript files, or the clinical deployment of a single combined model. The paper evaluates 50 selected predictors. No separate software license is assigned in this initial repository.
