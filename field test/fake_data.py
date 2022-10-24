import paho.mqtt.client as mqtt
import datetime
import random
import json
import time

ISOTIMEFORMAT = "%Y-%m-%d %H:%M:%S"
client = mqtt.Client(transport="websockets")
# client = mqtt.Client()
client.username_pw_set("iii", "iii05076416")
client.connect("192.168.1.39", 8087, 60)
while True:
    payload = {
        "Timestamp": datetime.datetime.now().timestamp(),
        "topic": "samuello-test-topic",
        "quality": round(random.randint(0, 1), 2),
        "30s_thickness": round(random.uniform(1.11, 1.33), 2),
        "5m_thickness": round(random.uniform(1.11, 1.33), 2),
        "30m_thickness": round(random.uniform(1.11, 1.33), 2),
        "1hr_thickness": round(random.uniform(1.11, 1.33), 2),
        "30s_yaw": round(random.uniform(1.11, 1.33), 2),
        "5m_yaw": round(random.uniform(1.11, 1.33), 2),
        "30m_yaw": round(random.uniform(1.11, 1.33), 2),
        "1hr_yaw": round(random.uniform(1.11, 1.33), 2),
        "adjustment": round(random.randint(0, 1), 2)
    }
    # for key in ["ingot", "discharge", "oil_pressure", "mould", "bucket"]:
    #     payload[key] = round(random.uniform(1, 10), 2)
    print(json.dumps(payload))
    client.publish("samuello-test-topic", json.dumps(payload))
    time.sleep(5)
