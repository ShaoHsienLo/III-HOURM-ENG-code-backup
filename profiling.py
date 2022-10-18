from pandas_profiling import ProfileReport
import sweetviz as sv
import pandas as pd

pd.set_option("display.max_columns", 100)


data = pd.read_csv("./data/labeled-data.csv")
print(len(data))
data = data[(data["final thickness"] >= 1.0) & (data["final thickness"] <= 1.6)]
print(len(data))
print(data.describe())
exit(0)

# 擷取時間段：8/13-8/14
# data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y-%m-%d %H:%M:%S")
# data = data[
#     (data["Timestamp"] > pd.to_datetime("2022-08-13", format="%Y-%m-%d")) &
#     (data["Timestamp"] < pd.to_datetime("2022-08-15", format="%Y-%m-%d"))
# ]

# data_in_range = data[(data["final thickness"] >= 1.2) & (data["final thickness"] <= 1.4)]
# data_out_range = data[(data["final thickness"] < 1.2) | (data["final thickness"] > 1.4)]
# print(data_in_range)
# print(data_out_range)

# rate = 0.86
# extra_length = int(1 / rate * float(len(data_in_range))) - len(data_in_range)
# data = pd.concat([data_in_range, data_out_range.sample(n=extra_length)])

# normal_data = data[data["label"] == "正常"]
# abnormal_data = data[data["label"] == "異常"]
# normal_rate = 0.82
# normal_data = normal_data[:int(len(normal_data) * normal_rate)]
# abnormal_data = abnormal_data[:int(len(abnormal_data) * (1 - normal_rate))]
# data = pd.concat([normal_data, abnormal_data], axis=0)
# print(data.shape)
# print("正常數量: ", len(data[data["label"] == "正常"]))
# print("異常數量: ", len(data[data["label"] == "異常"]))

data = data.drop(columns=["Timestamp", "thickness", "final thickness"])
data.to_csv("./data/data.csv", index=False)
print(data.shape)

# Pandas Profiling
# report = ProfileReport(data, title="Pandas Profiling", minimal=True)
# report.to_file("./eda/0813-0814共11多萬筆的/Pandas-Profiling.html")

# Sweetviz
# report = sv.analyze(data)
# report.show_html(filepath='./eda/0813-0814共11多萬筆的/Sweetviz-Profiling.html', open_browser=False)

