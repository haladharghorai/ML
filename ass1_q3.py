import pandas as pd

data = {
    "Student Name": ["Rahul", "Amit", "Priya", "Neha", "Riya"],
    "Roll Number": [1, 2, 3, 4, 5],
    "Marks": [85, 72, 91, 78, 88],
    "Attendance": [90, 85, 95, 80, 92]
}

df = pd.DataFrame(data)

def grade(marks):
    if marks >= 90:
        return "A"
    elif marks >= 80:
        return "B"
    elif marks >= 70:
        return "C"
    else:
        return "D"

df["Grade"] = df["Marks"].apply(grade)

print(df)