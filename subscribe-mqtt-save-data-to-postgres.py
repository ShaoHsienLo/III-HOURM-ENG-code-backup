import json

import paho.mqtt.client as mqtt
import pandas as pd
import psycopg2


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))

    # If we lose connection or reconnect, the terminal will resubscribe
    client.subscribe("hourmeng1")


def on_message(client, userdata, msg):
    # print(msg.topic + " " + msg.payload.decode("utf-8"))
    json_data = json.loads(msg.payload.decode("utf-8"))
    # print(json_data, type(json_data))

    conn = psycopg2.connect(database="postgres", user="postgres",
                            password='postgres', host="localhost",
                            port=5432)
    cur = conn.cursor()
    query = """
        insert into public.ipc1 values
        (current_timestamp, {}, {}, {}, {}, {})
    """.format(json_data["材料偏擺度"], json_data["材料厚度"], json_data["材料溫度"], json_data["環境溫度"],
               json_data["環境濕度"])
    cur.execute(query)
    # result = pd.DataFrame(cur.fetchall())
    # result.columns = [col.name for col in cur.description]

    conn.commit()
    conn.close()


# def postgres_conn():
#     conn = psycopg2.connect(database="postgres", user="postgres",
#                             password='postgres', host="localhost",
#                             port=5432)
#     cur = conn.cursor()
#     return cur


client = mqtt.Client(transport="websockets")
client.on_connect = on_connect
client.on_message = on_message
client.username_pw_set("iii", "iii05076416")
client.connect("139.162.96.124", 8087, 60)
client.loop_forever()







