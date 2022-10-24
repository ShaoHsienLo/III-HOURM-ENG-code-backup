import plotly.express as px
import pandas as pd
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt


# df = pd.read_csv("./data/unlabeled-data.csv")
# df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
# df = df[(df["final thickness"] > 0) & (df["final thickness"] < 20)]
# ts_lst = ["2022-06-30", "2022-07-01", "2022-07-02", "2022-07-03", "2022-08-12", "2022-08-13", "2022-08-14"]
# ts = pd.to_datetime(ts_lst, format="%Y-%m-%d")
# for t_str, t in zip(ts_lst, ts):
#     df_ = df[(df["Timestamp"] > t) & (df["Timestamp"] < t + timedelta(days=1))]
#     fig = px.scatter(x=list(df_.index), y=df_["final thickness"])
#     # fig.show()
#     fig.write_html("./plot/{}.html".format(t_str))

df = pd.read_csv("./data/labeled-processed-data.csv")
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()


