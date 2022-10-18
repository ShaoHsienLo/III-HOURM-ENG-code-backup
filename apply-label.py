import pandas as pd
import os


def f(x):
    if (x >= 1.2) & (x <= 1.4):
        return "正常"
    return "異常"


data = pd.read_csv("./data/unlabeled-data.csv")
data = data.fillna(method="bfill")
data["label"] = data["final thickness"].apply(f)
shift = 40
data = data.sort_index(ascending=False)
data = data.shift(shift)
data = data.dropna(how="all")
print(data)
data.to_csv("./data/labeled-data.csv", index=False)



