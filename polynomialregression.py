import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.random.rand(100, 1)
Y = 4 + 5*X + np.random.randn(100, 1)

reg = LinearRegression()
reg.fit(X, Y)

X_values = np.linspace(0, 1, 100).reshape(-1, 1)
Y_values = reg.predict(X_values)
plt.scatter(X, Y)
plt.plot(X_values, Y_values)
plt.show()