import json
import time

import pandas as pd
import paho.mqtt.client as mqtt
from sqlalchemy import create_engine
from datetime import timedelta, datetime


def get_ipc_data_from_potgres(ipc_name="ipc1"):
    engine = create_engine('postgresql://postgres:iii05076416@localhost:5432/postgres')
    result = engine.execute('SELECT "Timestamp", thickness, "材料偏擺度" '
                            'FROM public.ipc1_history '
                            'where "Timestamp" > current_timestamp - interval \'1 hour\'')
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=rows[0].keys())
    return df


ISOTIMEFORMAT = "%Y-%m-%d %H:%M:%S"

client = mqtt.Client(transport="websockets")
# transport="websockets"
client.username_pw_set("iii", "iii05076416")
client.connect("192.168.1.39", 8087, 60)

while True:
    df = get_ipc_data_from_potgres()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=ISOTIMEFORMAT)
    now_time = df["Timestamp"].iloc[-1]

    payload = {
        "timestamp": datetime.now().timestamp(),
        "topic": "mytopic",
        "thickness_30s": round((df.loc[df["Timestamp"] > now_time - timedelta(seconds=30), "thickness"]).mean(), 2),
        "thickness_5m": round((df.loc[df["Timestamp"] > now_time - timedelta(minutes=5), "thickness"]).mean(), 2),
        "thickness_30m": round((df.loc[df["Timestamp"] > now_time - timedelta(minutes=30), "thickness"]).mean(), 2),
        "thickness_1hr": round((df.loc[df["Timestamp"] > now_time - timedelta(hours=1), "thickness"]).mean(), 2),
        "yaw_30s": round((df.loc[df["Timestamp"] > now_time - timedelta(seconds=30), "材料偏擺度"]).mean(), 2),
        "yaw_5m": round((df.loc[df["Timestamp"] > now_time - timedelta(minutes=5), "材料偏擺度"]).mean(), 2),
        "yaw_30m": round((df.loc[df["Timestamp"] > now_time - timedelta(minutes=30), "材料偏擺度"]).mean(), 2),
        "yaw_1hr": round((df.loc[df["Timestamp"] > now_time - timedelta(hours=1), "材料偏擺度"]).mean(), 2)
    }
    print(payload)

    client.publish("mytopic", json.dumps(payload))
    time.sleep(5)

