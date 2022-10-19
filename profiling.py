from pandas_profiling import ProfileReport
import sweetviz as sv
import pandas as pd

# pd.set_option("display.max_columns", 100)


data = pd.read_csv("./data/data.csv")
# data = data[
#     (data["width"] > 50) & (data["width"] < 150) &
#     (data["M_tem"] > 20) & (data["M_tem"] < 90)
# ]
# print(len(data))
# data = data[(data["final thickness"] >= 1.0) & (data["final thickness"] <= 1.6)]
# print(len(data))
# print(data.describe())
# exit(0)

# 擷取時間段
# data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y-%m-%d %H:%M:%S")
# data = data[
#     (data["Timestamp"] > pd.to_datetime("2022-08-14", format="%Y-%m-%d")) &
#     (data["Timestamp"] < pd.to_datetime("2022-08-15", format="%Y-%m-%d"))
# ]

# data_normal = data[data["label"] == "正常"]
# data_abnormal = data[data["label"] == "異常"]

# print(data["label"].value_counts())

# rate = 0.86
# extra_length = int(1 / rate * float(len(data_normal))) - len(data_normal)
# data = pd.concat([data_normal, data_abnormal.sample(n=extra_length)])

# print(data["label"].value_counts())


# normal_data = data[data["label"] == "正常"]
# abnormal_data = data[data["label"] == "異常"]
# normal_rate = 0.82
# normal_data = normal_data[:int(len(normal_data) * normal_rate)]
# abnormal_data = abnormal_data[:int(len(abnormal_data) * (1 - normal_rate))]
# data = pd.concat([normal_data, abnormal_data], axis=0)
# print(data.shape)
# print("正常數量: ", len(data[data["label"] == "正常"]))
# print("異常數量: ", len(data[data["label"] == "異常"]))

# data = data.drop(columns=["Timestamp", "thickness", "final thickness"])
# data.to_csv("./data/data.csv", index=False)
# print(data.shape)

# Pandas Profiling
report = ProfileReport(data, title="Pandas Profiling", minimal=True)
report.to_file("./eda/0814的/Pandas-Profiling.html")

# Sweetviz
# report = sv.analyze(data)
# report.show_html(filepath='./eda/撈取資料範圍1.0到1.6的/Sweetviz-Profiling.html', open_browser=False)

