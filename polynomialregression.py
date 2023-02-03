import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

X = 4 * np.random.rand(100, 1) - 2
Y = 4 + 2 * X + 5 * X ** 2 + np.random.randn(100, 1)    #degree set garne

poly_features = PolynomialFeatures(degree=2, include_bias=False)   #degree set garne
X_poly = poly_features.fit_transform(X)

reg = LinearRegression()
reg.fit(X_poly, Y)

X_values = np.linspace(-2, 2, 100).reshape(-1, 1)
X_values_poly = poly_features.transform(X_values)

Y_values = reg.predict(X_values_poly)


plt.scatter(X, Y)
plt.plot(X_values, Y_values, color="red")
plt.show()