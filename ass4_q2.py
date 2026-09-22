import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house.csv")

x = df[["area", "bedrooms"]]
y = df["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print(prediction)