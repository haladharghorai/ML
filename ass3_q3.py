import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

data = {
    "Age": [22, 25, 30, 27, 28],
    "Salary": [25000, 30000, 35000, 45000, 50000],
    "Years_of_Experience": [1, 2, 5, 4, 6]
}

df = pd.DataFrame(data)

num_columns = ["Age", "Salary", "Years_of_Experience"]

data = df[num_columns]

standard_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

standard_data = standard_scaler.fit_transform(data)
minmax_data = minmax_scaler.fit_transform(data)

print("standard scaler:")
print(standard_data)
print("minmax scaler:")
print(minmax_data)

print("\n standard scaler range:")
print(standard_data.min(), standard_data.max())

print("\n minmax scaler range:")
print(minmax_data.min(), minmax_data.max())