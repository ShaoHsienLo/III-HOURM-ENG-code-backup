import json

import paho.mqtt.client as mqtt
import pandas as pd
from sqlalchemy import create_engine, types


def insert_data_to_postgres(df):
    engine = create_engine('postgresql://postgres:iii05076416@localhost:5432/postgres')
    sql_types = {
        "Timestamp": types.DateTime, "M_thickness": types.FLOAT, "M_tem": types.FLOAT, "A_tem": types.FLOAT,
        "A_hum": types.FLOAT, "M_yaw": types.FLOAT, "alarm": types.FLOAT, "mark": types.FLOAT
    }
    try:
        df.to_sql('ipc1_history', engine, index=False, dtype=sql_types)
    except ValueError as e:
        df.to_sql('ipc1_history', engine, if_exists="append", index=False, dtype=sql_types)
    print("Insert data successed.")


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # If we lose connection or reconnect, the terminal will resubscribe
    client.subscribe("new_ipc")


def on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode("utf-8"))
    df = pd.DataFrame.from_records(data)
    insert_data_to_postgres(df)


client = mqtt.Client(transport="websockets")
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("iii", "iii05076416")
client.connect("192.168.1.39", 8087, 60)
client.loop_forever()







