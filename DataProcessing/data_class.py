import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

""" 预处理轨迹数据 """
class LimitClass:
    def __init__(self, init_data_csv, save_limit_dir):
        self.init_data_csv = init_data_csv
        self.init_data_df = pd.read_csv(init_data_csv, sep=',', header=0)

        self.save_limit_dir = save_limit_dir
        os.makedirs(self.save_limit_dir, exist_ok=True)

        self.extracted_df = None
        self.limit_dis_df = None
        self.limit_dis_time_df = None
        self.limit_df = None

    """ 从初始文件提取数据，并保证轨迹连续 """
    def extract_data(self, columns):
        self.extracted_df = self.init_data_df[columns].drop_duplicates()

        # 先按照原始 id 排序，以保证顺序
        self.extracted_df.sort_values(by=["id", "Stime"], inplace=True)

        # 存储新的 id
        current_new_id = 1  # 初始编号
        new_ids = []  # 存储最终的编号

        # 按 id 分组
        for original_id, group in self.extracted_df.groupby("id"):
            prev_row = None  # 记录上一行数据
            for i, row in group.iterrows():
                if prev_row is not None: # 不是第一行
                    # 判断前一行的 Elon/Elat 是否与当前行的 Slon/Slat 不一致
                    if prev_row["Elon"] != row["Slon"] or prev_row["Elat"] != row["Slat"]:
                        current_new_id += 1  # 轨迹段不同，创建新 id
                new_ids.append(current_new_id)
                prev_row = row  # 更新prev_row

            current_new_id += 1  # 一个原始 id 处理完后，编号递增，保证不同原始 id 之间不混淆

        # 添加新的 id 列
        self.extracted_df["id"] = new_ids

        print("提取完成，并保证轨迹连续")

    """ 辅助函数，计算某列（排除 0）的平均值和中位数 """
    def column_mean_median(self, columns_name):
        non_zero_distances = self.extracted_df[self.extracted_df[columns_name] > 0][columns_name]
        avg = non_zero_distances.mean()  # 计算平均数
        median = non_zero_distances.median()  # 计算中位数

        print(f'除零外的{columns_name}，平均数：{avg}，中位数：{median}')
        return avg, median

    """ 根据limit_dis和limit_value，按id分组并分割轨迹 """
    def limit_by_value(self, limit_dis, limit_time):
        """ 辅助函数，根据limit_value，按id分组并分割轨迹 """
        def limit_column(data_df, column_name, limit_value):
            limit_df = data_df.copy()
            num = limit_df['id'].nunique()
            print(f'原始有{num}段')

            limit_df['new_id'] = 0  # 创建新列用于存储新的轨迹ID
            new_id = 1  # 新的轨迹ID，从1开始

            drop_indices = []  # 需要删除的索引列表

            # 按 id 分组并遍历每个分组
            for _, group in limit_df.groupby('id'):
                for i, row in group.iterrows():
                    value = row[column_name]  # 获取当前行值

                    # 如果距离超过阈值，删除当前行，并创建新的轨迹 ID
                    if value > limit_value:
                        drop_indices.append(i)  # 记录需要删除的索引
                        new_id += 1  # 更新 new_id
                    else:
                        limit_df.at[i, 'new_id'] = new_id  # 继续使用当前 new_id
                new_id += 1  # 更新 new_id

            # 删除超过阈值的行
            limit_df.drop(index=drop_indices, inplace=True)

            # 重新设置索引，从 0 开始
            limit_df.reset_index(drop=True, inplace=True)

            # 将new_id列放到第一列
            cols = ['new_id'] + [col for col in limit_df.columns if col != 'new_id']
            limit_df = limit_df[cols]

            # 删除原 id 列，并将 new_id 更名为 id
            limit_df.drop(columns=['id'], inplace=True)
            limit_df.rename(columns={'new_id': 'id'}, inplace=True)

            num = limit_df['id'].nunique()
            print(f'{column_name}_{limit_value}分割为{num}段')

            return limit_df

        self.column_mean_median("distance")
        self.limit_dis_df = limit_column(self.extracted_df,column_name="distance",limit_value=limit_dis).copy()

        self.column_mean_median("interval")
        self.limit_dis_time_df = limit_column(self.limit_dis_df, column_name="interval", limit_value=limit_time).copy()

    """ 删除表格中少于num行的id数据，并对剩下的 id 重新编号 """
    def limit_by_num(self, limit_num, limit_csv_name='limit_data.csv'):
        # 统计每个 id 的出现次数
        id_counts = self.limit_dis_time_df['id'].value_counts()

        # 过滤掉少于num次的 id
        valid_ids = id_counts[id_counts >= limit_num].index
        self.limit_df = self.limit_dis_time_df[self.limit_dis_time_df['id'].isin(valid_ids)].copy()

        # 重新设置索引，从 0 开始
        self.limit_df.reset_index(drop=True, inplace=True)

        # 对 id 列进行重新编号，从 1 开始
        self.limit_df['id'] = self.limit_df['id'].astype('category').cat.codes + 1

        num = self.limit_df['id'].nunique()
        print(f'limit后有{num}段')

        self.limit_df.to_csv(self.save_limit_dir + limit_csv_name, index=False)
        print("limit完成，结果已保存到", limit_csv_name)

    """ 辅助函数，统计不同dis和time对轨迹数量的影响 """
    def diff_limit(self, dis_list, time_list, limit_num, result_csv_name):
        result_dict = {}
        # 每个limit_dis
        for dis in dis_list:
            row_data = {}  # 存储当前 dis 约束下不同 time 约束的轨迹数
            # 每个limit_time
            for time in time_list:
                self.limit_by_value(dis, time)  # 按距离和时间限制轨迹
                self.limit_by_num(limit_num)  # 过滤少于 limit_num 段的轨迹
                row_data[time] = self.limit_df['id'].nunique()  # 计算轨迹段数
            result_dict[dis] = row_data

        # 转换为 DataFrame
        result_df = pd.DataFrame.from_dict(result_dict, orient='index')

        # 设置列名和索引
        result_df.columns = [f"time_{t}" for t in time_list]  # 第一行是时间限制值
        result_df.index.name = "dis_limit"  # 第一列是距离限制值

        result_df.to_csv(self.save_limit_dir + result_csv_name)
        print("统计不同dis和time完成，结果已保存到", result_csv_name)


class TrajClass:
    def __init__(self, limit_data_csv, save_traj_dir):
        self.limit_data_csv = limit_data_csv
        self.limit_data_df = pd.read_csv(limit_data_csv, sep=',', header=0)

        self.save_traj_dir = save_traj_dir
        os.makedirs(self.save_traj_dir, exist_ok=True)

        self.limit_traj_df = None
        self.linear_traj_df = None
        self.norm_traj_df = None

    """ 初始traj每行为起点和终点坐标,转换为每行只有一个坐标点的形式，并计算均值和方差 """
    def traj_point(self, limit_traj_csv_name='limit_traj.csv',mean_std_csv_name='mean_std.csv'):
        # 提取需要的列
        traj_df = self.limit_data_df[['id', 'Slon', 'Slat', 'Elon', 'Elat']]
        # 初始化结果列表
        result_data = []
        # 遍历每一行数据
        for index, row in traj_df.iterrows():
            # 处理每个id的坐标变化
            if index == 0: # 第一行直接加入结果
                result_data.append([row['id'], row['Slon'], row['Slat']])
            else: # 如果当前行的起始坐标与上一行的结束坐标不一样，则加入结果
                if (row['id'] != traj_df.at[index - 1, 'id']) or (row['Slon'] != traj_df.at[index - 1, 'Elon']) or (
                        row['Slat'] != traj_df.at[index - 1, 'Elat']):
                    result_data.append([row['id'], row['Slon'], row['Slat']])
            # 处理结束坐标
            result_data.append([row['id'], row['Elon'], row['Elat']])
        # 转换为DataFrame
        self.limit_traj_df = pd.DataFrame(result_data, columns=['id', 'lon', 'lat'])
        self.limit_traj_df['id'] = self.limit_traj_df['id'].astype(int)

        # 计算均值标准差
        norm_traj_df = self.limit_traj_df.copy()
        # 选取第 2 到第 3 列
        cols = norm_traj_df.columns[1:3]
        scaler = StandardScaler()
        norm_traj_df[cols] = scaler.fit_transform(norm_traj_df[cols])

        # 记录均值标准差
        pd.DataFrame(data=scaler.mean_).T.to_csv(self.save_traj_dir + mean_std_csv_name, mode="a", header=False,
                                                 index=False)
        pd.DataFrame(data=scaler.scale_).T.to_csv(self.save_traj_dir + mean_std_csv_name, mode="a", header=False,
                                                  index=False)

        print("traj_mean:", scaler.mean_)
        print("traj_std :", scaler.scale_)

        # 保存结果到新的CSV文件
        self.limit_traj_df.to_csv(self.save_traj_dir + limit_traj_csv_name, index=False)

        print("traj提取完成，结果已保存到", limit_traj_csv_name)

    """ 对每段轨迹进行线性插值 """
    def linear_traj(self, linear_length, limit_traj_csv, linear_traj_csv_name='linear_traj.csv'):
        def resample_trajectory(x, length=200):
            len_x = len(x)
            time_steps = np.arange(length) * (len_x - 1) / (length - 1)
            x = x.T
            resampled_trajectory = np.zeros((2, length))
            for i in range(2):
                resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
            return resampled_trajectory.T
        traj_df = pd.read_csv(limit_traj_csv)
        # 按照 'id' 列进行分组
        grouped = traj_df.groupby('id')
        # 初始化一个空列表，用于存储经过处理的组
        processed_groups = []
        # 遍历每个组，进行处理
        for group_id, group_df in grouped:
            # 获取第一列和第二三列数据
            traj_columns = group_df.iloc[:, 1:3].values
            # 对每组进行线性插值
            traj_groups = resample_trajectory(traj_columns, linear_length).astype(str)
            # 在 resampled_traj 的第一列插入 ID
            new_id_column = np.full((1, 1), group_id)
            processed_groups.append(pd.DataFrame(np.insert(traj_groups, 0, new_id_column, axis=1)))
        # 将经过处理的数据并成一个新的 DataFrame
        self.linear_traj_df = pd.concat(processed_groups)
        # 修改所有列名，使用DataFrame的columns属性来指定新的列名列表
        self.linear_traj_df.columns = ['id', 'lon', 'lat']
        # 重新设置索引，从 0 开始
        # self.linear_traj_df.reset_index(drop=True, inplace=True)
        self.linear_traj_df.to_csv(self.save_traj_dir + linear_traj_csv_name, index=False)
        print("linear_traj已保存到",linear_traj_csv_name)

    """ traj归一化 """
    def normalize_traj(self, traj_npy_name='traj.npy'):
        self.norm_traj_df = self.linear_traj_df.copy()
        # 选取第 2 到第 3 列
        cols = self.norm_traj_df.columns[1:3]
        # 归一化
        scaler = StandardScaler()
        self.norm_traj_df[cols] = scaler.fit_transform(self.norm_traj_df[cols])

        # 转换为三维traj
        traj_groups = self.norm_traj_df.groupby('id')
        traj_grouped_data = []

        for _, group_df in traj_groups:
            lon_lat_array = group_df[['lat', 'lon']].values # lat纬度，lon经度
            traj_grouped_data.append(lon_lat_array)

        grouped_data_array = np.array(traj_grouped_data)
        np.save(self.save_traj_dir + traj_npy_name, grouped_data_array)
        print("traj.npy结果已保存到", traj_npy_name)


class HeadClass:
    def __init__(self, limit_data_csv, save_head_dir):
        self.limit_data_csv = limit_data_csv
        self.limit_data_df = pd.read_csv(limit_data_csv, sep=',', header=0)

        self.save_head_dir = save_head_dir
        os.makedirs(self.save_head_dir, exist_ok=True)

        self.part_head_df = None
        self.sid_eid_df = None
        self.init_head_df = None

        self.head_df = None

    """ 计算head六个数据 """
    def part_head(self, part_head_csv_name='part_head.csv'):
        """ 计算时间段编号 """
        def time_slot(stime):
            time_str = str(stime)[-6:]  # 取后六列 hhmmss
            hour, minute = int(time_str[:2]), int(time_str[2:4])
            return (hour * 60 + minute) // 5  # 每5分钟一个时间段，共288个时间段

        # 计算统计值
        result = self.limit_data_df.groupby("id").agg(
            departure=pd.NamedAgg(column="Stime", aggfunc=lambda x: time_slot(x.iloc[0])),  # 开始时间编号
            total_dis=pd.NamedAgg(column="distance", aggfunc="sum"), # 相邻两点距离累加
            total_time=pd.NamedAgg(column="interval", aggfunc="sum"), # 相邻两点时间累加
            total_len=pd.NamedAgg(column="id", aggfunc="count") # id行数
        ).reset_index()
        result["total_len"] = result["total_len"] + 1  # 行数+1

        # 计算平均值
        result["avg_dis"] = result["total_dis"] / (result["total_len"]-1)
        result["avg_speed"] = result["total_dis"] / result["total_time"]
        # 直接替换 inf 和 NaN
        result.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        # 重新排列列顺序
        new_order = ['id', 'departure', 'total_dis', 'total_time', 'total_len', 'avg_dis', 'avg_speed']
        self.part_head_df = result[new_order]

        self.part_head_df.to_csv(self.save_head_dir + part_head_csv_name, index=False)
        print("部分head计算完成，结果已保存到", part_head_csv_name)

        return self.part_head_df

    """ 计算sid、eid """
    def sid_eid(self, limit_traj_csv, lat_num, lon_num, sid_eid_csv_name='sid_eid.csv'):
        """ 找轨迹的最大最小经纬度 """
        def max_min_lat_lon(traj_csv):
            # 读取CSV文件
            traj_df = pd.read_csv(traj_csv, sep=',', header=0)
            # 计算 lon 和 lat 的最大最小值
            lon_min = min(traj_df["Slon"].min(), traj_df["Elon"].min())  # 最小经度
            lon_max = max(traj_df["Slon"].max(), traj_df["Elon"].max())  # 最大经度
            lat_min = min(traj_df["Slat"].min(), traj_df["Elat"].min())  # 最小纬度
            lat_max = max(traj_df["Slat"].max(), traj_df["Elat"].max())  # 最大纬度

            boundary = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}
            return boundary

        """ 划分网格 """

        def divide_grids(boundary, lat_num, lon_num):
            lat_min, lat_max = boundary['lat_min'], boundary['lat_max']
            lon_min, lon_max = boundary['lon_min'], boundary['lon_max']
            # Divide the lattude and lonitude into grids_num intervals.
            lat_interval = (lat_max - lat_min) / lat_num
            lon_interval = (lon_max - lon_min) / lon_num
            # Create arrays of lattude and lonitude values.
            lat_grids = np.arange(lat_min, lat_max, lat_interval)
            lon_grids = np.arange(lon_min, lon_max, lon_interval)
            return lat_grids, lon_grids

        """ 计算点 (lat, lon) 在网格中的索引 (lat_index, lon_index) """
        def get_grid_index(lat, lon, lat_grids, lon_grids):
            lat_index = np.searchsorted(lat_grids, lat, side='right') - 1
            lon_index = np.searchsorted(lon_grids, lon, side='right') - 1
            return lat_index, lon_index

        # 取出每段轨迹的起点和终点
        start_end_df = self.limit_data_df.groupby('id').agg(
                {'Slon': 'first', 'Slat': 'first', 'Elon': 'last', 'Elat': 'last'}).reset_index()

        # 划分网格
        lat_grids, lon_grids = divide_grids(max_min_lat_lon(limit_traj_csv), lat_num, lon_num)

        # 计算起点和终点所在网格编号
        start_end_df["sid"] = (start_end_df.apply(lambda row: get_grid_index(row["Slat"], row["Slon"], lat_grids, lon_grids),
                              axis=1)
                     .apply(lambda x: x[0] * len(lon_grids) + x[1]))

        start_end_df["eid"] = (start_end_df.apply(lambda row: get_grid_index(row["Elat"], row["Elon"], lat_grids, lon_grids),
                              axis=1)
                     .apply(lambda x: x[0] * len(lon_grids) + x[1]))

        self.sid_eid_df = start_end_df[["id", "sid", "eid"]]

        self.sid_eid_df.to_csv(self.save_head_dir + sid_eid_csv_name, index=False)
        print("sid_eid计算完成，结果已保存到", sid_eid_csv_name)

    """ 合并head """
    def init_head(self, part_head_csv, sid_eid_scv):
        part_head_df = pd.read_csv(part_head_csv, sep=',', header=0)
        sid_eid_df = pd.read_csv(sid_eid_scv, sep=',', header=0)
        self.init_head_df = pd.merge(part_head_df, sid_eid_df, on="id", how="inner")

    """ head归一化 """
    def normalized_head(self, head_npy_name='head.npy',mean_std_csv_name='mean_std.csv'):
        norm_head_df = self.init_head_df.copy()
        # 选取第 3 到第 7 列
        cols = norm_head_df.columns[2:7]
        # 归一化
        scaler = StandardScaler()
        norm_head_df = self.init_head_df.copy()
        norm_head_df[cols] = scaler.fit_transform(norm_head_df[cols])

        # 记录均值标准差
        pd.DataFrame(data=scaler.mean_).T.to_csv(self.save_head_dir + mean_std_csv_name, mode="a", header=False, index=False)
        pd.DataFrame(data=scaler.scale_).T.to_csv(self.save_head_dir + mean_std_csv_name, mode="a", header=False, index=False)

        print("head_mean:", scaler.mean_)
        print("head_std :", scaler.scale_)

        # 存为npy
        self.head_df = norm_head_df.iloc[:, 1:].astype(float) # 去掉id列
        np.save(self.save_head_dir + head_npy_name, self.head_df.to_numpy())
        print("head.npy结果已保存到", head_npy_name)

args = {
    'limit_dis': 100, # m
    'limit_time': 10000, # min
    'limit_num': 2,
    'linear_length': 200,
    'lat_num': 16, # 纬度，南北方向，高
    'lon_num': 16, # 经度，东西方向，宽
    'save_dir': 'dataset/save/',
    'save_img_path': 'dataset/save/img',
    'mean_std_csv_name': 'mean_std.csv',
}

if __name__ == '__main__':
    from types import SimpleNamespace
    from DataProcessing.utils import plot_2d_df, plot_2d_csv, resample_trajectory, get_grid_index, divide_grids

    # 将args转换为SimpleNamespace，用于将字典转换为对象，使用.访问属性
    config = SimpleNamespace(**args)

    operation = 'linear'
    # operation = 'sid_eid'
    # 处理Traj

    if operation == 'linear':
        traj = TrajClass(limit_data_csv='dataset/save/limit_data.csv',
                         save_traj_dir=config.save_dir_path)
        # 线性插值
        traj.linear_traj(linear_length=config.linear_length,
                         limit_traj_csv='dataset/save/limit_traj.csv',
                         linear_traj_csv_name='linear_traj_' + str(config.linear_length) + 'num.csv')
        # 绘制（直接绘制df失败）
        plot_2d_csv(traj_2d_csv='dataset/save/linear_traj_' + str(config.linear_length) + 'num.csv',
                    save_img_dir=config.save_img_path,
                    img_name='linear_traj_' + str(config.linear_length) + 'num')
        # 归一化
        traj.normalize_traj(traj_npy_name='traj_' + str(config.linear_length) + 'num')
        # 绘制
        plot_2d_df(traj_df=traj.norm_traj_df,
                   save_img_dir=config.save_img_path,
                   img_name='norm_traj_' + str(config.linear_length) + 'num')
    if operation == 'sid_eid':
        # 处理Head
        head = HeadClass(limit_data_csv='dataset/save/limit_data.csv',
                         save_head_dir=config.save_dir_path)
        # 合并
        head.init_head(part_head_csv='dataset/save/part_head.csv',
                       sid_eid_scv='dataset/save/sid_eid_' +'lat'+str(config.lat_num)+ '_lon'+str(config.lon_num)+ '.csv')

        # 归一化
        head.normalized_head(head_npy_name='head_' +'lat'+str(config.lat_num)+ '_lon'+str(config.lon_num),
                             mean_std_csv_name=config.mean_std_csv_name)