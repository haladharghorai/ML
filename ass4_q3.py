import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("house.csv")

x = df[["area"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)

linear_prediction = linear_model.predict(x_test)

linear_r2 = r2_score(y_test, linear_prediction)

poly = PolynomialFeatures(degree=2)

x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()

poly_model.fit(x_train_poly, y_train)

poly_prediction = poly_model.predict(x_test_poly)

poly_r2 = r2_score(y_test, poly_prediction)

print("linear regression r2:", linear_r2)
print("polynomial regression r2:", poly_r2)