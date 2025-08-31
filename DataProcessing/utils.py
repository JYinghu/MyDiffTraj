import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
""" 2D CSV: Plot trajectories from CSV file using scatter """
def plot_2d_csv(traj_2d_csv, save_img_dir,
                color='blue', alpha=0.5, img_name='Wuhan_traj'):
    # 读CSV
    data = pd.read_csv(traj_2d_csv)
    trajs = pd.DataFrame(data.values)
    trajs_group = trajs.groupby(0)  # 按第0列（id）分组

    # 绘制
    plt.figure(figsize=(8, 8))
    for group_id, group_df in trajs_group:
        plt.scatter(group_df[1], group_df[2], color=color, alpha=alpha, s=5, label=f'Trajectory {group_id}')
    plt.tight_layout()
    plt.title(img_name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(save_img_dir + '/' + img_name + '.png')
    plt.show()

""" 2D DataFrame: Plot trajectories from DataFrame using scatter """
def plot_2d_df(traj_df, save_img_dir,
               color='blue', alpha=0.5, img_name='Wuhan_traj'):
    trajs_group = traj_df.groupby("id")  # 按 id 分组

    plt.figure(figsize=(8, 8))
    for group_id, group_df in trajs_group:  # 遍历每个 id 轨迹
        plt.scatter(group_df["lon"], group_df["lat"], color=color, alpha=alpha, s=5)
    plt.title(img_name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(save_img_dir + '/' + img_name + '.png')
    plt.show()

""" 3D NPY: Plot trajectories from .npy file using scatter """
def plot_3d_npy(traj_3d_npy, save_img_dir,
                color='blue', alpha=0.5, img_name='Wuhan_traj'):
    trajs = np.load(traj_3d_npy, allow_pickle=True)
    trajs = trajs[:, :, :2]  # 取 lon 和 lat

    plt.figure(figsize=(8, 8))
    for i in range(len(trajs)):
        traj = trajs[i]
        plt.scatter(traj[:, 0], traj[:, 1], color=color, alpha=alpha, s=5)
    plt.tight_layout()
    plt.title(img_name)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig(save_img_dir + '/' + img_name + '.png')
    plt.show()
""" 线性插值 """
def resample_trajectory(x, length=200):
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

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

def get_city_from_plt_file(plt_file_path):
    def reverse_geocode(geolocator,lat, lon):
        try:
            location = geolocator.reverse((lat, lon), language='en')
            if location and 'city' in location.raw['address']:
                return location.raw['address']['city']
            elif location and 'town' in location.raw['address']:
                return location.raw['address']['town']
            elif location and 'state' in location.raw['address']:
                return location.raw['address']['state']
            return "Unknown"
        except GeocoderTimedOut:
            time.sleep(1)
            return reverse_geocode(geolocator, lat, lon)

    geolocator = Nominatim(user_agent="geoapiExercises")
    with open(plt_file_path, 'r') as file:
        lines = file.readlines()[6:]  # 跳过前6行无效数据
        if not lines:
            return "Unknown"
        # 取第一个点
        first_line = lines[0].strip().split(',')
        lat = float(first_line[0])
        lon = float(first_line[1])
        return reverse_geocode(geolocator, lat, lon)

""" 根据特定经纬度范围判断城市 """
def get_city_from_plt_range(plt_file_path):
    def get_city_by_latlon(lat, lon):
        city_bounds = {
            "Beijing": {
                "lat_min": 39.4, "lat_max": 41.1,
                "lon_min": 115.7, "lon_max": 117.4
            },
            "Shanghai": {
                "lat_min": 30.6, "lat_max": 31.9,
                "lon_min": 120.8, "lon_max": 122.2
            },
            "Changchun": {
                "lat_min": 43.4, "lat_max": 44.3,
                "lon_min": 124.6, "lon_max": 126.0
            },
            "Chengdu": {
                "lat_min": 30.3, "lat_max": 31.4,
                "lon_min": 103.3, "lon_max": 104.8
            },
            "Qingdao": {
                "lat_min": 35.5, "lat_max": 37.1,
                "lon_min": 119.3, "lon_max": 121.0
            },
            "Shenzhen": {
                "lat_min": 22.3, "lat_max": 22.9,
                "lon_min": 113.7, "lon_max": 114.7
            }
        }

        for city, bounds in city_bounds.items():
            if bounds["lat_min"] <= lat <= bounds["lat_max"] and bounds["lon_min"] <= lon <= bounds["lon_max"]:
                return city
        return "Unknown"
    with open(plt_file_path, 'r') as file:
        lines = file.readlines()[6:]
        if not lines:
            return "Unknown"
        first_line = lines[0].strip().split(',')
        lat = float(first_line[0])
        lon = float(first_line[1])
        return get_city_by_latlon(lat, lon)


if __name__ == "__main__":
    plt_path = R"D:\MyProjects\PythonAbout\DiffusionModel\MyDiffTraj\DataProcessing\Geolife\base\010\Trajectory\20070804033032.plt"
    print(get_city_from_plt_range(plt_path))
