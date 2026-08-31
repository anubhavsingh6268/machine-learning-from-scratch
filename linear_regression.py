class LinearRegression:
  def __init__(self):
    self.m = 0
    self.b = 0
    
  def fit(self, X_train, y_train):
    x_mean = X_train.mean()
    y_mean = y_train.mean()

    self.numerator=0
    self.denominator=0
    for i in range(len(X_train)):
      self.numerator += (X_train.iloc[i] - x_mean) * (y_train.iloc[i] - y_mean)
      self.denominator += (X_train.iloc[i] - x_mean) ** 2

    self.m = self.numerator/self.denominator
    self.b = y_mean - (self.m * x_mean)

  def predict(self, X_test):
    return self.m * X_test + self.b
