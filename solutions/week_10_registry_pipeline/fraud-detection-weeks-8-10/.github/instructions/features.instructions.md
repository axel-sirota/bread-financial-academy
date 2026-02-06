---
applyTo: "**/features*.py"
---
# Feature Engineering Instructions

## Function Naming
- Feature functions start with `create_` or `extract_`
- Column names are descriptive: `amount_log`, `is_weekend`, NOT `f1`

## Function Pattern
All feature functions MUST:
1. Accept DataFrame as first parameter
2. Return NEW DataFrame (never modify input)
3. Validate required columns exist
4. Include docstring with example

## Example
```python
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features.

    Args:
        df: DataFrame with 'hour' and 'day_of_week' columns

    Returns:
        DataFrame with new time features added
    """
    result = df.copy()  # Never modify input
    result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
    return result
```
