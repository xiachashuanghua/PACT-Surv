# Input and checkpoint contract

## Prepared target cohort

One `data/{cancer}.npz` contains:

| Field | Shape / meaning |
| --- | --- |
| `x` | `(N, 1, G)` raw candidate RNA matrix; the single modality is RNA |
| `clinical` | `(N, C)` numerical/encoded clinical features in the frozen feature order |
| `genes` | `(G,)` ordered candidate gene symbols |
| `prior` | `(G,)` raw LLM penalty-factor means, each in `[0.1, 1]` |
| `full_rna` | `(N, D)` high-dimensional RNA in the shared source-network coordinate order |
| `broad_genes` | `(D,)` corresponding gene symbols; `D=17,689` in the paper |
| `patients` | `(N,)` unique patient strings; not sample filenames |
| `time` | `(N,)` overall-survival follow-up in months for target evaluation |
| `event` | `(N,)` death indicator, 1=event and 0=censored |
| `train_01`, `val_01`, `test_01`, …, `test_10` | Integer row indices of the ten disjoint outer partitions |

Each outer partition must cover every cohort row exactly once. Across partitions the same patient can recur in the test set. Missing clinical/RNA entries use NaN; infinity is invalid. Imputation and scaling are fitted within the appropriate training fold. Do not globally standardize the target arrays again before passing them to this code.

Raw priors are means of three fixed LLM ratings. A smaller value means a stronger prior preference, not higher expression. The selected exponent `eta` determines screening factors `p**eta`; actual RNA penalty weights are normalized by their mean over the screened genes. They are not the raw priors.

The archived target cohort sizes are LUAD 442, LGG 509, KIRC 311, BRCA 768, and LIHC 358. LUAD retains two zero-time censored observations under its frozen protocol; the other targets retain finite positive follow-up times.

## Original-layout conversion

`prepare_inputs.py --legacy-root DIR` reads the following relative paths without modifying them:

```text
DIR/
  data/cohort/cohort.npz
  data/cohort/split_01/preprocessing.npz   # through split_10
  data/extra.npz
  data/extra2.npz
  data/prior15.npz
  pancan19.npz                           # ordered LUAD broad-RNA coordinate map
  data/{lgg,kirc,brca,lihc}/
    cohort.npz
    clin.npz
    rnapan.npz
    prior.npz
```

The converter preserves original arrays and eligibility rules, selects the RNA modality, aligns high-dimensional coordinates, and carries the LUAD outer split indices forward. The other four cohorts use the original stratified seeds: an 80/20 split at seed `split`, then a 50/50 validation/test split of the remainder at seed `100 + split`. Generated manifests record input hashes. Object-containing archives are loaded only by this trusted-input conversion utility; portable cohort files use non-object arrays.

## Pretraining source pool

`sources/pancan_all.npz` requires `X`, `time`, `event`, `type`, `patients`, and `genes`. `X` is the already-prepared source expression matrix in the same gene order as `broad_genes`; `type` holds upper-case TCGA abbreviations. The paper's source pool contains 9,890 patients from 33 cancer types. The frozen source preprocessing and ordered source pool are prerequisites; this code does not replace them with a new download or a newly normalized matrix.

Source survival times retain their archived units. Source Cox fitting uses within-cancer risk ordering; target AUC horizons require months. A target's source pool must exclude all patients of that target cancer. It is partitioned 90/10 using cancer/event stratification and seed 19000.

## Checkpoints

`model.pkl` stores the clinical–RNA coefficients, selected RNA indices, imputation/scaling parameters, outer row indices, selected source weight decay, adapted network weights, shared transfer coefficient, and protocol/source fingerprints. `selection.json` stores the selected configuration and model hash.

For enabled transfer, the corresponding five source NPZ files contain `w1`, `b1`, `w2`, `b2`, `w3`, `risk_mean`, and `risk_sd`. Source output statistics stay fixed after adaptation. `ft=None` means the neural correction is disabled. All local array, checkpoint, and output directories are excluded by `.gitignore`.
