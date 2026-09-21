import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


df = pd.DataFrame({
    'Age': [22, 25, 28, 30, 35],
    'Salary': [25000, 30000, 40000, 45000, 55000],
    'Years_of_Experience': [1, 2, 4, 5, 10]
})


standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)


minmax_scaler = MinMaxScaler()
minmax_scaled = 
print("StandardScaler Transformed Array:")
print(standard_scaled)

print("\nStandardScaler Numerical Range:")
print("Minimum:", standard_scaled.min(axis=0))
print("Maximum:", standard_scaled.max(axis=0))


print("\nMinMaxScaler Transformed Array:")
print(minmax_scaled)

print("\nMinMaxScaler Numerical Range:")
print("Minimum:", minmax_scaled.min(axis=0))
print("Maximum:", minmax_scaled.max(axis=0))
