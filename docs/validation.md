# Validation

The release is checked against the target-specific exclusion implementation used by the paper. The archived checkpoint hashes are listed in `configs/paper_selected_models.json`.

- Python syntax and CLI import checks passed.
- Seven algorithm checks passed: Cox/Breslow gradient, training-only fitting, adapted-parameter reload, source fitting, zero-reference IG completeness, AUC edge cases, and input/hash rejection paths.
- The five archived cohorts converted successfully with 442, 509, 311, 768 and 358 eligible patients, preserving their ten outer partitions.

Frozen-checkpoint replay completed on 2026-11-15 using Python 3.10.18, NumPy 2.1.2, SciPy 1.15.3, scikit-learn 1.7.1 and PyTorch 2.4.1+cu124, with inference on CPU:

- All **50** predictors passed, covering **2,410** outer-test patient–predictor records. These are records, not 2,410 unique patients.
- Patient identifiers, survival outcomes and all outer partition indices matched the archived records.
- C-index and all horizon AUC values matched exactly.
- Maximum absolute prediction difference was **6.011687219142914e-7**, within the unchanged `atol=1e-6, rtol=1e-6` criteria.
- IG/reconstruction spot checks used one test patient from each cohort's first predictor and one from the disabled-transfer predictor: six records, maximum absolute reconstruction residual **2.723484737732207e-8**. This is not a full-patient IG replay.
- Twenty-five code comparisons confirmed unchanged numerical syntax trees, including the complete `linear.py` module (whitespace normalized). Local converted inputs also matched all 50 archived test patient lists.

Details are in [frozen_replay_validation.json](frozen_replay_validation.json) and [numerical_code_validation.json](numerical_code_validation.json). No patient identifiers or patient-level outputs are included in these reports. Source pretraining and the full configuration searches were not rerun as part of packaging; a three-step source-fitting smoke test was run on synthetic arrays.
