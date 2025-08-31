from types import SimpleNamespace
from DataProcessing.wkhj.wkhj_class import *
from DataProcessing.utils import plot_2d_df, plot_2d_csv


# 辅助函数，计算不同limit对轨迹段数影响
def diff_limit_test(config):
    limit_cal = LimitClass(init_data_csv='dataset/wkhj_data.csv',
                           save_limit_dir=config.save_dir)

    # 提取需要的列
    extract_cols = ['id', 'Slon','Slat','Elon','Elat','Stime','Etime','interval','distance']
    limit_cal.extract_data(columns=extract_cols)

    # dis平均数21.7 中位数10.5，time平均数47.8 中位数6
    # 不同dis和time限制的轨迹段数
    limit_cal.diff_limit(dis_list=[100000,150,125,100,75,50],
                         time_list=[100000,150,125,100,75,50],
                         limit_num=config.limit_num,
                         result_csv_name='calculate_result.csv')

# 辅助函数，转换为轨迹点
def traj_point(traj_csv):
    traj_df = pd.read_csv(traj_csv,sep=',',header=0).copy()
    traj_df['id'] = traj_df['id'].astype('category').cat.codes + 1
    # 提取需要的列
    traj_df = traj_df[['id', 'Slon', 'Slat', 'Elon', 'Elat']]
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
    limit_traj_df = pd.DataFrame(result_data, columns=['id', 'lon', 'lat'])
    limit_traj_df['id'] = limit_traj_df['id'].astype(int)

    return limit_traj_df

# 运行limit
def run_limit(config):
    # 处理Limit
    limit = LimitClass(init_data_csv='dataset/wkhj_data.csv',
                       save_limit_dir=config.save_dir)

    # 提取需要的列
    extract_cols = ['id', 'Slon','Slat','Elon','Elat','Stime','Etime','interval','distance']
    limit.extract_data(columns=extract_cols)

    # 按dis和time划分
    limit.limit_by_value(limit_dis=config.limit_dis, limit_time=config.limit_time)

    # 按num划分
    limit.limit_by_num(limit_num=config.limit_num, limit_csv_name='limit_data_'+str(config.limit_dis)+ '.csv')

# 运行traj
def run_traj(config):
    # 处理Traj
    traj = TrajClass(limit_data_csv='dataset/save/limit_data_'+str(config.limit_dis)+ '.csv',
                     save_traj_dir=config.save_dir)

    # 提取limit_traj
    traj.traj_point(limit_traj_csv_name='limit_traj.csv',mean_std_csv_name=config.mean_std_csv_name)
    # 绘制
    plot_2d_df(traj_df=traj.limit_traj_df,
               save_img_dir=config.save_img_path,
               img_name='limit_traj')

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

# 运行head
def run_head(config):
    # 处理Head
    head = HeadClass(limit_data_csv='dataset/save/limit_data_'+str(config.limit_dis)+ '.csv',
                     save_head_dir=config.save_dir)

    # 计算六个特征
    head.part_head(part_head_csv_name='part_head.csv')

    # 计算sid、eid
    head.sid_eid(limit_traj_csv='dataset/save/limit_data_'+str(config.limit_dis)+ '.csv',
                 lat_num=config.lat_num, lon_num=config.lon_num,
                 sid_eid_csv_name='sid_eid_' +'lat'+str(config.lat_num)+ '_lon'+str(config.lon_num)+ '.csv')

    # 合并
    head.init_head(part_head_csv='dataset/save/part_head.csv',
                   sid_eid_scv='dataset/save/sid_eid_' +'lat'+str(config.lat_num)+ '_lon'+str(config.lon_num)+ '.csv')

    # 归一化
    head.normalized_head(head_npy_name='head_' +'lat'+str(config.lat_num)+ '_lon'+str(config.lon_num),
                         mean_std_csv_name=config.mean_std_csv_name)

# 辅助函数，不同limit_dis对轨迹图片影响
def diff_dis(config):
    run_limit(config)

    # 处理Traj
    traj = TrajClass(limit_data_csv='dataset/save/limit_data_'+str(config.limit_dis)+ '.csv',
                     save_traj_dir=config.save_dir)
    # 提取limit_traj
    traj.traj_point(limit_traj_csv_name='limit_traj_'+str(config.limit_dis)+ '.csv', mean_std_csv_name=config.mean_std_csv_name)
    # 绘制
    plot_2d_df(traj_df=traj.limit_traj_df, save_img_dir=config.save_img_path,
               img_name='limit_traj_'+str(config.limit_dis))

args = {
    'limit_dis': 300, # m
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
    # 将args转换为SimpleNamespace，用于将字典转换为对象，使用.访问属性
    config = SimpleNamespace(**args)

    # 绘制原轨迹
    # plot_2d_df(traj_point('dataset/wkhj_data.csv'),
    #             save_img_dir=config.save_img_path,
    #             img_name='wkhj_traj')

    run_limit(config)

    run_traj(config)

    run_head(config)

    # # 测试不同limit_dis轨迹图片
    # diff_dis(config)

    # 不同limit的轨迹段数
    # diff_limit_test(config)
