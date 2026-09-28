# Best models summary

## Final model selection

The final models were selected by grouped cross-validation (GroupKFold on `(ticker, level_id)`) within the training period, using PR AUC (average precision) as the selection criterion. Each horizon was assigned its own hyperparameter grid, reflecting differences in class balance: a relatively high `scale_pos_weight` for the 1-hour horizon (approximately 10:1 imbalance), a moderate value for 4 hours (approximately 4:1), and a minimal value for 24 hours (approximately 1:1).

## Best hyperparameters

| target       |   cv_pr_auc_mean |   cv_pr_auc_std |   train_pr_auc |    gap |   param_subsample |   param_scale_pos_weight |   param_reg_lambda |   param_reg_alpha |   param_num_leaves |   param_n_estimators |   param_min_child_samples |   param_max_depth |   param_learning_rate |   param_colsample_bytree |
|:-------------|-----------------:|----------------:|---------------:|-------:|------------------:|-------------------------:|-------------------:|------------------:|-------------------:|---------------------:|--------------------------:|------------------:|----------------------:|-------------------------:|
| breakout_1h  |           0.2117 |          0.0081 |         0.39   | 0.1782 |              0.85 |                      3   |                0.5 |               0   |                 31 |                  800 |                       200 |                 3 |                  0.03 |                     0.7  |
| breakout_4h  |           0.4274 |          0.0083 |         0.4797 | 0.0524 |              0.7  |                      5   |                0.5 |               0.1 |                 31 |                  800 |                        20 |                 3 |                  0.01 |                     0.85 |
| breakout_24h |           0.7244 |          0.0139 |         0.7629 | 0.0385 |              0.85 |                      1.5 |                1   |               0.1 |                 15 |                  400 |                       100 |                 3 |                  0.03 |                     0.7  |

## Metrics on the hold-out test set

| target       | model      | split   |   threshold |   roc_auc |   pr_auc |   brier |   precision |   recall |     f1 |
|:-------------|:-----------|:--------|------------:|----------:|---------:|--------:|------------:|---------:|-------:|
| breakout_1h  | calibrated | test    |        0.5  |    0.8074 |   0.204  |  0.0607 |      0.2857 |   0.0053 | 0.0104 |
| breakout_1h  | calibrated | test    |        0.14 |    0.8074 |   0.204  |  0.0607 |      0.2107 |   0.6148 | 0.3138 |
| breakout_4h  | calibrated | test    |        0.5  |    0.793  |   0.4054 |  0.1214 |      0.4839 |   0.1109 | 0.1804 |
| breakout_4h  | calibrated | test    |        0.24 |    0.793  |   0.4054 |  0.1214 |      0.405  |   0.6505 | 0.4992 |
| breakout_24h | calibrated | test    |        0.5  |    0.7341 |   0.6795 |  0.2099 |      0.655  |   0.6707 | 0.6628 |
| breakout_24h | calibrated | test    |        0.34 |    0.7341 |   0.6795 |  0.2099 |      0.5554 |   0.9108 | 0.69   |

## Notes

- ROC AUC and PR AUC are threshold-independent. PR AUC is treated as the primary metric because of class imbalance.
- `precision`, `recall`, and `f1` are reported at two thresholds: the default value of 0.5 and the tuned threshold selected on the validation set by maximising F1.
- For the 1-hour horizon, the default threshold of 0.5 is not appropriate; the tuned threshold yields non-zero recall and F1.
- Isotonic calibration reduces the Brier score for the 1-hour and 4-hour horizons. For the 24-hour horizon the raw model is already reasonably calibrated, so the improvement is marginal.
