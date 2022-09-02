import pandas as pd
import os


def f(x):
    if (x >= 1.24) & (x <= 1.36):
        return "正常"
    return "異常"


data = pd.read_csv("data.csv")
data = data.fillna(method="bfill")
data["label"] = data["final thickness"].apply(f)
shift = 60
print(data.head(10))
data = data.sort_index(ascending=False)
print(data.head(10))
print(data.shape)
data = data.shift(shift)
data = data.dropna(how="all")
print(data.shape)
data.to_csv("data_.csv", index=False)



