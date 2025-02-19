# Linear Regression Model

This project implements a simple Linear Regression model in Python using NumPy and Pandas. It includes a flexible class structure for training and making predictions with the model, and a separate utility for calculating the coefficients using the Normal Equation approach.

## Structure
The project consists of two main components:

1. **LinearRegressionModel Class**: This class handles training and predictions.
2. **RSS Class**: A helper class that calculates the weights and bias based on input features and target values.

## Files
- `LinearRegressionModel.py`: Contains the `LinearRegressionModel` class.
- `RSS_min.py`: Contains the `RSS` class for coefficient calculation.

## Requirements
- Python 3.x
- NumPy
- Pandas
- Joblib

## Installation
```bash
pip install numpy pandas joblib
```

## Usage

### LinearRegressionModel Class

```python
from LinearRegressionModel import LinearRegressionModel

model = LinearRegressionModel()
X = [[1, 2], [2, 3], [3, 4]]  # Example features
y = [3, 5, 7]                 # Example targets

model.fit(X, y)
predictions = model.predict([[4, 5], [5, 6]])
print(predictions)
```

### RSS Class (Coefficient Calculation)

```python
from RSS_min import RSS

X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])

weights, bias = RSS.coeffecients(X, y)
print(f"Weights: {weights}, Bias: {bias}")
```

## Explanation
### 1. LinearRegressionModel Class
- **Initialization (`__init__`)**: Initializes `weights` and `bias`.
- **fit(X, y)**: Converts input to NumPy arrays, calls `RSS.coeffecients()` to compute weights and bias.
- **predict(X)**: Predicts target values based on input features and learned weights and bias.

### 2. RSS Class
- **coeffecients(X, y)**: Computes the slope (weights) and intercept (bias) for each feature using the Normal Equation approach.

## Key Equations
### Weights (Slope):
```
w = Cov(X, y) / Var(X)
```
### Bias (Intercept):
```
b = mean(y) - sum(w[i] * mean(X[:, i]))
```

## Notes
- Handles multiple features.
- Skips columns with NaN values.
- Returns 0 weight if variance is 0.

## Limitations
- Assumes linear relationship between input and target variables.
- NaN handling is simplistic (skips columns).


