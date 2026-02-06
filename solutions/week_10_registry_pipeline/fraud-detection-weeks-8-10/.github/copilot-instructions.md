# Fraud Detection Project - Copilot Instructions

## Project Context
This is a fraud detection ML pipeline for financial transactions.
- Data: Transaction records with amount, merchant, time features
- Model: XGBoost classifier for binary classification
- Tracking: MLflow for experiment tracking
- Environment: Runs locally and on AWS SageMaker

## Code Style Requirements

### Type Hints
- ALWAYS include type hints for function parameters and returns
- Use `Optional[]` for parameters that can be None
- Import types from `typing` module

### Docstrings
- Use Google-style docstrings for ALL public functions
- Include: Brief description, Args, Returns, Raises

### Naming Conventions
- Functions: snake_case
- Variables: snake_case
- Constants: UPPER_CASE
- No single-letter variables except loop indices

## Error Handling
- Validate inputs at function boundaries
- Use logging module, not print()
- Raise descriptive exceptions
- Never silently catch exceptions

## ML Conventions
- Follow sklearn API style (fit, predict, transform)
- Always set random_state for reproducibility
- Log experiments with MLflow
