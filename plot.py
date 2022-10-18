import plotly.express as px
import pandas as pd


df = pd.read_csv("./data/data.csv")
fig = px.scatter(x=list(df.index), y=df["In_temperature"])
fig.show()
# fig.write_html("plot.html")
