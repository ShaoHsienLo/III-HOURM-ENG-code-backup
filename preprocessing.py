import pandas as pd
import os
import codecs
from modules import process_json_files
from loguru import logger


# process_json_files()
#
# exit(0)

# 資料儲存路徑
old_ipc_path = r"C:\Users\samuello\Downloads\III\宏英\code\data\DATA\1\data"
new_ipc_path = r"C:\Users\samuello\Downloads\III\宏英\code\data\DATA\2\data"

# 所有資料檔案
old_ipc_files = os.listdir(old_ipc_path)
new_ipc_files = os.listdir(new_ipc_path)

# 以新ipc的資料為準，看重複且有連續時間生產的檔案有哪些
duplicate_ipc_files = []
for new_ipc_file in new_ipc_files:
    if new_ipc_file in old_ipc_files:
        duplicate_ipc_files.append(new_ipc_file)
drop_files = ["2022032213.json", "2022081808.json"]
duplicate_ipc_files = [file for file in duplicate_ipc_files if file not in drop_files]

# 舊ipc的資料欄位：Timestamp、thickness、材料溫度、環境溫度、環境濕度、材料偏擺度、alarm、mark
# 新ipc的資料欄位：Timestamp、width、A_temperature、A_Hum、In_temperature、roller_temperature_1、PID_temperature_1、CT
# 、RPM、thickness

for file in duplicate_ipc_files:
    print(f"File name: {file}")
    # 置換舊ipc資料欄位
    #     - 刪除alarm、mark
    # old_ipc_column_name_mapping = {
    #     "材料溫度": "M_tem",
    #     "環境溫度": "A_tem",
    #     "環境濕度": "A_hum",
    #     "thickness": "final thickness"
    # }
    old_ipc_drop_columns = ["alarm", "mark"]
    old_ipc_data = pd.read_json(os.path.join(old_ipc_path, file), lines=True)
    # old_ipc_data.rename(columns=old_ipc_column_name_mapping)
    old_ipc_data = old_ipc_data.drop(columns=old_ipc_drop_columns)

    # 置換新ipc資料欄位
    #     - thickness->extrusion thickness
    #     - 刪除A_temperature、A_Hum
    new_ipc_column_name_mapping = {
        "thickness": "extrusion thickness"
    }
    new_ipc_drop_columns = ["A_temperature", "A_Hum"]
    new_ipc_data = pd.read_json(os.path.join(new_ipc_path, file), lines=True)
    new_ipc_data.rename(columns=new_ipc_column_name_mapping)
    new_ipc_data = new_ipc_data.drop(columns=new_ipc_drop_columns)

    # 組合資料表
    old_ipc_data["Timestamp"] = pd.to_datetime(old_ipc_data["Timestamp"], format="%Y-%m-%d %H:%M:%S")
    new_ipc_data["Timestamp"] = pd.to_datetime(new_ipc_data["Timestamp"], format="%Y-%m-%d %H:%M:%S")

    df = pd.merge(old_ipc_data, new_ipc_data, how="right")

    # 儲存成csv檔案
    try:
        path = r"C:\Users\samuello\Downloads\III\宏英\code\model-input-data"
        df.to_csv(os.path.join(path, file[:-5] + ".csv"), index=False)
    except Exception as e:
        logger.error(e)
        exit(0)




