import pandas as pd
import matplotlib.pyplot as plt

from linear_regression import LinearRegression

# Load dataset
data = pd.read_csv("data.csv")

# Select feature and target
X = data["feature"]
y = data["target"]

# Train/test split
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]


# Create and train model
model = LinearRegression()

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

print("Slope:", model.m)
print("Intercept:", model.b)
print("\nPredictions:")
print(predictions)

# Plot results
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, predictions, label="Predicted")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression From Scratch")
plt.legend()

plt.savefig("results/regression_plot.png")
plt.show()
