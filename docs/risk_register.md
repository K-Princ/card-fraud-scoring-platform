
# Risk Register

## Project: Card Fraud Scoring Platform

| ID | Risk | Probability | Impact | Mitigation Strategy |
|---|---|---|---|---|
| R01 | Severe class imbalance | High | High | Use PR-AUC, class weights and threshold optimization |
| R02 | Data leakage | Medium | Critical | Use chronological validation and leakage audits |
| R03 | Missing transaction features | High | Medium | Implement explicit missing-value handling |
| R04 | Concept drift | Medium | High | Monitor delayed-label model performance |
| R05 | Data drift | High | Medium | Compare production and reference distributions |
| R06 | Training-serving skew | Medium | Critical | Reuse the same feature transformation pipeline |
| R07 | Model artifact corruption | Low | High | Validate artifacts before deployment |
| R08 | API latency | Medium | High | Benchmark inference and optimize bottlenecks |
| R09 | Sensitive information exposure | Low | Critical | Avoid real cardholder data and secure credentials |
| R10 | Delayed fraud labels | High | High | Implement delayed-label evaluation |

## Monitoring Strategy

- Monitor feature distributions.
- Monitor prediction probability distributions.
- Monitor actual model performance when labels arrive.
- Record model versions.
- Generate alerts for significant changes.

## Review Policy

Review this register after every major development milestone.
Update probability, impact and mitigation strategies
when new risks are discovered.
