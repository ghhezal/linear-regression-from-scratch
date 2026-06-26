# Linear Regression from Scratch

A linear regression model built **from scratch in Python** using gradient descent — no scikit-learn or ML libraries used.

## What it does

- Loads a dataset of student study time vs exam scores
- Implements the **cost function (MSE)** manually
- Implements **gradient descent** manually using partial derivatives of MSE
- Plots the final regression line and the loss curve over epochs

## Math behind it

The model fits a line `y = mx + b` by minimizing the Mean Squared Error:

$$E = \frac{1}{n} \sum_{i=0}^{n} (y_i - (mx_i + b))^2$$

Using these gradients derived by hand:

$$\frac{\partial E}{\partial m} = -\frac{2}{n} \sum x_i(y_i - (mx_i + b))$$

$$\frac{\partial E}{\partial b} = -\frac{2}{n} \sum (y_i - (mx_i + b))$$

## Project structure

```
linear-regression-from-scratch/
├── data/
│   └── studytime_score.csv   # 230 training examples
├── linear_regression.py      # main script
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python linear_regression.py
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Epochs | 10000 |
| Initial m | 0 |
| Initial b | 0 |

## Results

The model converges to approximately:
- `m ≈ 6–7` — each extra study hour adds ~6–7 points
- `b ≈ 45–50` — baseline score with 0 study time

Two plots are produced side by side:

| Regression line | Loss curve |
|---|---|
| Scatter plot of data with the fitted line | MSE decreasing over 10,000 epochs |

## Dataset

230 synthetic but realistic training examples with intentional noise:
- Some students studied a lot and still failed
- Some students studied little and still passed
- Study time distribution weighted toward 1–5 hours (realistic)

## What I learned

- Derived the partial derivatives of MSE with respect to `m` and `b` by hand using the chain rule
- Implemented gradient descent from scratch without any ML library
- Debugged convergence issues by analyzing the effect of learning rate and epoch count on the loss curve
- Understood why the cost function averages over all examples (vs single-example loss)
