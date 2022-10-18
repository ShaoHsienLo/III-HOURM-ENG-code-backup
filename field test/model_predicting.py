import json

import joblib
import numpy as np
import paho.mqtt.client as mqtt
import random
import time
import datetime

import pandas as pd
from sqlalchemy import create_engine


pd.set_option('display.max_columns', 100)


def get_ipc_data_from_potgres(ipc_name="ipc1"):
    engine = create_engine('postgresql://postgres:iii05076416@localhost:5432/postgres')
    result = engine.execute('SELECT * FROM public.{}_history order by "Timestamp" desc limit 1'.format(ipc_name))
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=rows[0].keys())
    return df


def load_model():
    return joblib.load("rf.model")


# mapping = {
#     "M_tem": "材料溫度",
#     "In_temperature": "In_temperature",
#     "RPM": "RPM",
#     "A_tem": "環境溫度",
#     "width": "width",
#     "M_yaw": "材料偏擺度"
# }

cols = ["材料溫度", "In_temperature", "RPM", "環境溫度", "width", "材料偏擺度"]
quality_mapping = {0: "正常", 1: "異常"}

df_ipc1 = get_ipc_data_from_potgres(ipc_name="ipc1")
df_ipc2 = get_ipc_data_from_potgres(ipc_name="ipc2")

df = pd.concat([df_ipc1, df_ipc2], axis=1)
df = df[cols]

model = load_model()
y_prob = model.predict_proba(df)
y_prob = y_prob[0]
max_index = np.argmax(y_prob).item()
quality = quality_mapping[max_index]
score = y_prob[max_index] * 100

print(quality)
print(score)

