import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score


data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "pass": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[["study_hours", "attendance"]]
y = df["pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()

model.fit(x_train, y_train)

y_probability = model.predict_proba(x_test)[:, 1]

for threshold in [0.3, 0.5, 0.7]:

    y_pred = (y_probability >= threshold).astype(int)

    print("threshold :", threshold)
    print("precision :",precision_score(y_test, y_pred))
    print("recall :",recall_score(y_test, y_pred))