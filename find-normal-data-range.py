import pandas as pd
import numpy as np

# 讀檔
data = pd.read_csv("data.csv")
data = data.drop(columns=["Timestamp", "thickness"])

# 計算每個欄位的第25百分位數至第75百分位數的範圍，依此範圍訂定為正常值區間
for col in data.columns:
    quan_10 = round(np.quantile(data[col], 0.1), 2)
    quan_90 = round(np.quantile(data[col], 0.9), 2)
    diff = round(quan_90 - quan_10, 2)
    print("{}: Math.random() * {} + {}".format(col, diff, quan_10))

    