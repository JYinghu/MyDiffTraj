from datetime import datetime

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


# Haversine公式计算两点之间的距离（单位：米）
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # 地球半径，单位：米
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


input_csv_path = './zgdzdx/dataset/zgdzdx_data.csv'
output_csv_path = './zgdzdx/dataset/zgdzdx_data_class.csv'

# 读取CSV文件
df = pd.read_csv(input_csv_path)  # 替换为你的CSV文件路径

# 确保timestamp列为字符串
df['timestamp'] = df['timestamp'].astype(str)

# 按id和timestamp排序，确保轨迹点按时间顺序
df = df.sort_values(by=['id', 'timestamp'])

# 创建结果DataFrame
result = []

# 按id分组，处理每条轨迹
for id_, group in df.groupby('id'):
    group = group.reset_index(drop=True)
    for i in range(len(group) - 1):
        slon = group.loc[i, 'lon']
        slat = group.loc[i, 'lat']
        stime = group.loc[i, 'timestamp']
        elon = group.loc[i + 1, 'lon']
        elat = group.loc[i + 1, 'lat']
        etime = group.loc[i + 1, 'timestamp']

        # 将字符串时间戳转换为datetime以计算时间间隔
        stime_dt = datetime.strptime(stime, '%Y%m%d%H%M%S')
        etime_dt = datetime.strptime(etime, '%Y%m%d%H%M%S')

        # 计算时间间隔（秒）
        interval = (etime_dt - stime_dt).total_seconds()

        # 计算距离（米）
        distance = haversine(slon, slat, elon, elat)

        # 添加到结果，保留原始字符串时间戳
        result.append({
            'id': id_,
            'Slon': slon,
            'Slat': slat,
            'Stime': stime,
            'Elon': elon,
            'Elat': elat,
            'Etime': etime,
            'interval': interval,
            'distance': distance
        })

# 转换为DataFrame
result_df = pd.DataFrame(result)

# 保存结果到新的CSV文件
result_df.to_csv(output_csv_path, index=False)

print("处理完成，结果已保存")