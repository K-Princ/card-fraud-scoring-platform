# API Testing

## Test Framework
- pytest
- FastAPI TestClient
- httpx

## Test Coverage

| Test | Expected Behavior | Result |
|---|---|---|
| Home endpoint | Returns API welcome message | PASS |
| Health endpoint | Returns healthy status | PASS |
| Model status | Confirms model artifacts are loaded | PASS |
| Valid prediction | Returns fraud prediction and probability | PASS |
| Invalid amount | Rejects negative transaction amount | PASS |
| Invalid category | Rejects unsupported category | PASS |
| Missing categories | Uses Unknown defaults | PASS |

## Test Execution

Command:

python -m pytest -v

Result: 7 tests passed.

## Conclusion

The FastAPI service passed all current automated tests for
endpoint availability, model loading, prediction, and input validation.