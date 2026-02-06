---
applyTo: "**/test_*.py,**/tests/**/*.py"
---
# Testing Instructions

## Structure
- Use pytest
- Group tests in classes: `class TestFunctionName:`
- Test names: `test_<what_is_being_tested>`

## Pattern (Arrange-Act-Assert)
```python
def test_example(self):
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    assert result == expected
```

## What to Test
1. Happy path (normal operation)
2. Edge cases (empty, boundary)
3. Error cases (invalid input)

## Assertions
- Specific: `assert x == 5` not just `assert x`
- Floats: `pytest.approx()`
- Exceptions: `pytest.raises()`
