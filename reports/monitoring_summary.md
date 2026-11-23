# Data Drift Monitoring — Initial Results

## Dataset Overview

- Reference dataset: 472,432 transactions
- Current dataset: 118,108 transactions
- Features monitored: TransactionAmt, ProductCD, card4, card6

## Drift Detection Results

Evidently detected drift in 0 out of 4 monitored features (0.0%).

| Feature | Drift Score | Drift Status |
|---|---:|---|
| ProductCD | 0.078504 | Not Detected |
| card6 | 0.021941 | Not Detected |
| TransactionAmt | 0.018890 | Not Detected |
| card4 | 0.018746 | Not Detected |

## Conclusion

No data drift was detected between the reference and current datasets using the configured Evidently drift tests.

This is an initial monitoring experiment using a chronological split of the IEEE-CIS dataset. It does not establish that future production data will remain stable or that model performance is unaffected.