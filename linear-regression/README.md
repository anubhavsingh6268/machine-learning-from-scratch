# Linear Regression From Scratch

A simple implementation of Linear Regression from scratch using Python and Pandas.

The purpose of this project is to understand how Linear Regression works internally instead of relying on machine learning libraries such as Scikit-learn.

## How It Works

The model calculates the slope (`m`) and intercept (`b`) using the Ordinary Least Squares method.

The prediction is calculated using:

y = mx + b

The slope is calculated from the relationship between the input feature and target values, and the intercept is then calculated from their means.

## Project Structure
```text
linear-regression/
├── linear_regression.py
├── example.py
└── results/
    └── regression_plot.png
