import json
import paho.mqtt.client as mqtt
import random
import time
import datetime

import pandas as pd
from sqlalchemy import create_engine


def get_quality_from_potgres():
    engine = create_engine('postgresql://postgres:iii05076416@localhost:5432/postgres')
    result = engine.execute('SELECT result FROM public.ipc1_history order by ingot_start_time desc limit 6')
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=rows[0].keys())
    return df

