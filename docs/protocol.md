# PACT-Surv model and evaluation protocol

## Model

The final signed log-risk is

`R = R_clinical + R_sparseRNA + gamma * mean(f_j(X))`.

Each neural output `f_j` is standardized using that source network's fixed training-output mean and standard deviation. The same patient-shared `gamma` multiplies the five-network mean; it is initialized at zero and is unrestricted in sign.

Gene preferences enter both univariate survival-score ranking and normalized RNA shrinkage. Clinical and screened-RNA coefficients are jointly fitted using Cox partial likelihood, an elastic-net penalty, and the event-scaled clinical–RNA agreement term. The linear anchor remains fixed during neural adaptation.

Source pretraining uses a 256–64-hidden-unit GELU network, input dropout 0.2, hidden dropout 0.5, and cancer-stratified Cox partial likelihood with Breslow ties. Five seeds are fitted for each of weight decays 0.001, 0.01 and 0.1, at learning rate 1e-5. Validation is checked every step, with at most 5,000 steps and patience 500.

## Selection

Each of the five targets has ten outer train/validation/test partitions. Within outer training, selection uses five-fold cross-validation repeated twice, with seed `12000 + split`.

1. Select among 972 clinical–RNA configurations using mean training-CV C-index.
2. Compare three five-network source ensembles using prediction-head adaptation and candidate step counts 25, 50, 100, 200, 400 and 800.
3. On the selected ensemble, compare no transfer with 36 adaptation configurations. `frozen` freezes the first layer and updates subsequent layers; `full` updates the whole network. Source-centred quadratic penalties constrain the adapted parameters.

The outer validation set is reported but does not select the PACT-Surv configuration in this protocol. Staged selection reuses the same CV folds. Test evaluation begins after all requested paper predictors have been frozen. The archived selection file contains 23 prediction-head, 26 full-network and one disabled-transfer configuration; nine predictors use uniform gene factors (`eta=0`).

## Evaluation

C-index follows comparable event-time pairs, with tied risk scores receiving half credit. Horizon AUC uses event cases with `time < h` and observed controls with `time > h`, for `h=12,60,120` months. The pooled AUC sums credit and comparable-pair counts across horizons; it is not the arithmetic mean of three AUCs and is not an IPCW estimator. A horizon with no comparable pairs remains undefined. Cohort summaries report means and sample standard deviations across the ten model partitions.

## Interpretation

The neural reference is `gamma * mean(f_j(0))`, with zero defined in target-standardized and clipped input coordinates. The transfer change is `gamma * mean(f_j(X) - f_j(0))`. Gauss–Legendre IG attributes this change to high-dimensional RNA genes. The four additive terms are clinical risk, sparse-RNA risk, neural reference, and summed gene IG.

Numerical integration starts with 32 nodes and doubles up to 1,024. Its checks retain the archived tolerances: quadrature `atol=1e-6, rtol=1e-4`; native endpoint completeness `atol=1e-4, rtol=1e-3`. The native FP32 neural outputs remain the prediction reference. Reference contributions are model-level offsets; they are not independent patient-varying factors within one predictor.
