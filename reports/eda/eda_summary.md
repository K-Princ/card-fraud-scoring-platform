# Exploratory Data Analysis — IEEE-CIS Fraud Detection

## 1. Project Overview

This report documents the exploratory data analysis (EDA) performed for the Real-Time Card Fraud Scoring Platform with Drift Monitoring.

The objective is to understand transaction behavior, identify data-quality issues, investigate fraud patterns, and establish decisions for the upcoming feature-engineering phase.

**Dataset:** IEEE-CIS Fraud Detection  
**Problem type:** Binary classification  
**Target variable:** `isFraud`

Target values:
- `0` — Legitimate transaction
- `1` — Fraudulent transaction

## 2. Dataset Structure

The project uses two primary training datasets:

| Dataset | Rows | Columns |
|---|---:|---:|
| Transaction | 590,540 | 395 |
| Identity | 144,233* | 41* |

*Identity dimensions should be confirmed against the original file inspection; the merged dataset gained 40 columns.*

The datasets are connected using `TransactionID`.

After performing a left join:

| Metric | Before Join | After Join |
|---|---:|---:|
| Rows | 590,540 | 590,540 |
| Columns | 395 | 435 |

Both transaction and identity `TransactionID` columns were verified as unique.

The left join preserved all transaction rows without multiplying records.

## 3. Class Distribution

The target distribution is:

| Transaction Type | Count | Percentage |
|---|---:|---:|
| Legitimate | 569,877 | 96.50% |
| Fraudulent | 20,663 | 3.50% |
| Total | 590,540 | 100% |

### Observation

The dataset exhibits significant class imbalance.

Fraudulent transactions represent approximately 3.5% of the training data.

### Implication

Accuracy alone is not sufficient for evaluating fraud detection models.

Future model evaluation should include:
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

## 4. Missing Value Analysis

The following features had the highest observed missing percentages:

| Feature | Missing Percentage |
|---|---:|
| dist2 | 93.63% |
| D7 | 93.41% |
| D13 | 89.51% |
| D14 | 89.47% |
| D12 | 89.04% |
| D6 | 87.61% |
| D9 | 87.31% |
| D8 | 87.31% |
| V153 | 86.12% |
| V149 | 86.12% |

### Missingness by Fraud Label

For `dist2`:

| Class | Missing Percentage |
|---|---:|
| Legitimate | 94.05% |
| Fraudulent | 81.94% |

For `D7`:

| Class | Missing Percentage |
|---|---:|
| Legitimate | 94.19% |
| Fraudulent | 71.98% |

### Observation

Missingness differs between legitimate and fraudulent transactions for both features.

### Implication

Missingness may contain predictive information.

High-missingness columns should not automatically be removed. Their usefulness must be evaluated during feature engineering and model validation.

## 5. Transaction Amount Analysis

The average and median transaction amounts were:

| Statistic | Legitimate | Fraudulent |
|---|---:|---:|
| Mean | 134.51 | 149.24 |
| Median | 68.50 | 75.00 |

### Observations

- Fraudulent transactions have a higher mean transaction amount.
- Fraudulent transactions also have a higher median.
- Both classes exhibit substantial right skewness.
- Extreme transaction amounts occur in both classes.
- The distributions overlap considerably.

### Implication

Transaction amount may be a useful predictive feature, but it cannot independently distinguish fraudulent transactions.

Outliers should not automatically be deleted.

A logarithmic transformation may be worth evaluating during feature engineering.

## 6. ProductCD Analysis

Transaction counts by product category:

| ProductCD | Transaction Count |
|---|---:|
| W | 439,670 |
| C | 68,519 |
| R | 37,699 |
| H | 33,024 |
| S | 11,628 |

Observed fraud rates:

| ProductCD | Fraud Rate |
|---|---:|
| C | 11.69% |
| S | 5.90% |
| H | 4.77% |
| R | 3.78% |
| W | 2.04% |

### Observation

Category C has the highest observed fraud rate, while W has the lowest.

Although W accounts for most transactions, it does not have the highest fraud rate.

### Implication

ProductCD may contain useful predictive information.

These historical associations do not establish causation.

## 7. Card Network Analysis

Transaction counts:

| Card Network | Count |
|---|---:|
| Visa | 384,767 |
| Mastercard | 189,217 |
| American Express | 8,328 |
| Discover | 6,651 |

Observed fraud rates:

| Card Network | Fraud Rate |
|---|---:|
| Discover | 7.73% |
| Visa | 3.48% |
| Mastercard | 3.43% |
| American Express | 2.87% |

### Observation

Discover has the highest observed fraud rate.

Visa has the largest transaction volume but not the highest fraud rate.

### Implication

Card network may provide useful categorical information.

Rates for categories with smaller sample sizes require cautious interpretation.

## 8. Card Type Analysis

Transaction counts:

| Card Type | Count |
|---|---:|
| Debit | 439,938 |
| Credit | 148,986 |
| Debit or credit | 30 |
| Charge card | 15 |

Observed fraud rates:

| Card Type | Fraud Rate |
|---|---:|
| Credit | 6.68% |
| Debit | 2.43% |
| Charge card | 0.00% |
| Debit or credit | 0.00% |

### Observation

Credit card transactions have a higher observed fraud rate than debit card transactions.

The two rare categories have extremely small sample sizes.

### Implication

Card type may be a useful predictive feature.

Rare categories should be handled carefully during categorical preprocessing.

## 9. Temporal Analysis

The `TransactionDT` feature was examined across fraud and legitimate transactions.

Fraud rates across ten relative-time quantile bins varied approximately between:

- Minimum: 2.0%
- Maximum: 4.3%

### Observation

Fraud prevalence changes across the relative transaction timeline.

### Implication

Temporal variation supports the need for time-aware model validation.

`TransactionDT` is a relative time offset, not an actual calendar timestamp.

The observed variation alone does not prove model drift or performance degradation.

## 10. Identity Dataset Analysis

Only 24.42% of transactions had matching identity information.

Approximately 75.58% did not have a matching identity record.

The `id_01` feature had approximately 75.58% missingness after the left join.

### Join Validation

| Validation | Result |
|---|---|
| Transaction IDs unique | Passed |
| Identity IDs unique | Passed |
| Row count preserved | Passed |
| Unexpected row multiplication | Not observed |

### Observation

Identity information is available only for a subset of transactions.

### Engineering Implication

The scoring system must support transactions without identity information.

Identity features must be optional in the online inference pipeline.

All transaction records must be preserved during joining.

## 11. Key Findings

1. The dataset contains 590,540 transactions, with a fraud prevalence of approximately 3.50%.
2. Significant class imbalance requires metrics beyond accuracy.
3. Several transaction features have very high missingness.
4. Missingness differs between fraud and legitimate classes for selected features.
5. Fraudulent transactions have higher average and median transaction amounts in this dataset.
6. ProductCD category C has the highest observed fraud rate.
7. Discover has the highest observed fraud rate among the card networks examined.
8. Credit card transactions have a higher observed fraud rate than debit card transactions.
9. Fraud prevalence varies across relative transaction-time bins.
10. Identity information is available for only 24.42% of transactions.

## 12. Data Quality and Modeling Considerations

The following decisions will guide the next phase:

- Preserve all transaction rows.
- Do not automatically remove high-missingness features.
- Investigate missingness indicators.
- Treat identity features as optional.
- Validate all joins using transaction identifiers.
- Avoid using identifiers as predictive features without justification.
- Investigate feature availability at prediction time.
- Use time-aware validation to evaluate temporal generalization.
- Avoid target leakage during preprocessing and feature engineering.
- Evaluate categorical features with appropriate encoding strategies.
- Do not interpret historical fraud-rate differences as causal relationships.

## 13. Limitations

- This analysis is exploratory and does not establish causation.
- Some categorical groups have small sample sizes.
- Relative transaction time does not provide an actual calendar date.
- No model performance conclusions can be drawn from EDA alone.
- Feature usefulness must be established through leakage-safe validation.

## 14. Conclusion

The IEEE-CIS dataset contains useful transaction, card, product, and identity information for fraud classification.

However, significant class imbalance, high missingness, partial identity coverage, and temporal variation introduce important engineering and modeling challenges.

The findings from this analysis will guide the development of a reusable data-cleaning and feature-engineering pipeline in Phase 2.

**Phase 1 status:** Core exploratory analysis completed. Findings documented; review and reproducibility checks remain part of the phase approval process.

## Data Validation — Phase 2

- Transaction records: 590,540
- Transaction columns: 395
- Duplicate TransactionIDs: 0
- Missing target values: 0
- Required columns: All present
- Negative transaction amounts: 0
- Negative transaction times: 0
- Target values: 0 and 1
- Missing values in key columns: 0

Result: Initial data validation passed.

### Missingness Feature Analysis

- Created missing-value indicators for dist2, D7, and D13.
- D7 missing: 2.70% fraud rate.
- D7 present: 14.88% fraud rate.
- D7 missingness may provide predictive information.

Decision: Preserve missingness indicators for further model evaluation.

### Missingness Indicator Feature Engineering

Created a reusable function:
- File: src/features/build_features.py
- Function: add_missing_indicators()
- Features: dist2_missing, D7_missing, D13_missing

Validation:
- Tested on a 1,000-row sample.
- All three indicators were created successfully.
- Original input DataFrame remained unchanged.
- Different missingness combinations were observed.

Result: Feature function test passed.

### Reusable Preprocessing Pipeline

Created in src/features/build_features.py:

Functions:
- add_missing_indicators()
- fill_numeric_missing()
- get_medians()
- apply_medians()
- preprocess_features()

Validation:
- Tested on 1,000 training rows.
- Tested on 1,000 validation rows.
- Both produced 6 features.
- Zero missing values after preprocessing.
- Training medians reused for validation data.

Result: Initial reusable preprocessing workflow passed.

### Categorical Feature Engineering

Categorical columns:
- ProductCD
- card4
- card6

Implemented:
- Missing categories replaced with "Unknown".
- One-hot encoding tested using Pandas.
- Reusable scikit-learn OneHotEncoder created.
- handle_unknown="ignore" configured.

Validation:
- Encoder fitted on a training sample.
- Unseen category tested successfully.
- Output retained 11 columns.
- No encoding error occurred.

Result: Initial categorical encoding tests passed.

### Reusable Categorical Encoder

Implemented in src/features/build_features.py:

- create_encoder()
- apply_encoder()

Validation:
- Encoder fitted on training sample.
- Unseen category handled without errors.
- Transformed 1,000 rows successfully.
- Output contained 11 encoded columns.
- Feature column consistency check passed.

Result: Categorical transformation passed.

### Temporal Train-Validation Split

Split method: TransactionDT 80th percentile.

Time cutoff: 12192853.6

Training:
- Rows: 472,432
- Fraud rate: 3.51%

Validation:
- Rows: 118,108
- Fraud rate: 3.44%

Validation checks:
- Training transactions occur before validation.
- No chronological overlap.
- Fraud rates are similar across both sets.

Result: Temporal split validation passed.

### Full Dataset Preprocessing Results

Temporal split:
- Training rows: 472,432
- Validation rows: 118,108

Training-fitted numeric medians:
- dist2: 38.0
- D7: 0.0
- D13: 0.0

Feature engineering:
- 3 numerical features
- 3 missingness indicators
- 15 one-hot encoded categorical features
- Total: 21 features

Validation:
- Training shape: 472,432 x 21
- Validation shape: 118,108 x 21
- Missing values: 0 in both datasets
- Feature columns consistent: True
- All features numeric: True

Result: Full training and validation preprocessing passed.