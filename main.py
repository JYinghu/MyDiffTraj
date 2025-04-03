import torch
import torch.nn as nn
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from types import SimpleNamespace
from utils.config import args
from utils.EMA import EMAHelper
from utils.Traj_UNet import *
from utils.logger import Logger, log_info
from pathlib import Path
import shutil


# This code part from https://github.com/sunlin-ai/diffusion_tutorial


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # 加载同一动态链接库继续运行
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 日志级别，屏蔽info和warning的日志信息
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # 按PCI总线的ID顺序识别GPU设备
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 只使用第一个GPU

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t) # dim=-1，在最后一维选取索引t对应的元素
    return c.reshape(-1, 1, 1) # 形状调整，-1表示该维度大小不变，变成三维


def main(config, logger, exp_dir):

    # Modified to return the noise itself as well
    def q_xt_x0(x0, t): # 计算前向扩散过程
        mean = gather(alpha_bar, t)**0.5 * x0 # 时间步t的噪声均值
        var = 1 - gather(alpha_bar, t) # 时间步t的噪声方差
        eps = torch.randn_like(x0).to(x0.device) # 生成与x0形状相同的标准高斯噪声
        return mean + (var**0.5) * eps, eps  # also returns noise，计算xt

    # Create the model
    unet = Guide_UNet(config).cuda() # 模型
    # print(unet)
    traj = np.load('./dataset/normalized_traj_20num.npy',
                   allow_pickle=True)
    traj = traj[:, :, :2] # 只取前两维，即坐标
    head = np.load('./dataset/normalized_head_10m.npy',
                   allow_pickle=True)
    traj = np.swapaxes(traj, 1, 2) # 交换后两个维度(batch,channel,length)
    traj = torch.from_numpy(traj).float()
    head = torch.from_numpy(head).float()
    dataset = TensorDataset(traj, head) # 输入数据集
    dataloader = DataLoader(dataset,
                            batch_size=config.training.batch_size, # 批次大小
                            shuffle=True, # 打乱数据
                            num_workers=8) # 8个cpu线程加载数据

    # Training params
    # Set up some parameters
    n_steps = config.diffusion.num_diffusion_timesteps # 扩散步数
    beta = torch.linspace(config.diffusion.beta_start,
                          config.diffusion.beta_end, n_steps).cuda() # 均匀分布n_steps个值
    alpha = 1. - beta # 原始数据比例
    alpha_bar = torch.cumprod(alpha, dim=0) # 累乘，每个时间步全局数据比例
    lr = 2e-4  # 学习率，Explore this - might want it lower when training on the full dataset

    losses = []  # Store losses for later plotting，记录损失值，绘制损失曲线
    # optimizer
    optim = torch.optim.AdamW(unet.parameters(), lr=lr)  # Optimizer，优化器AdamW，权重衰减

    # EMA 指数移动平均，平滑模型权重，提高泛化能力
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate) # 平滑系数mu
        ema_helper.register(unet) # 注册unet参数，训练时计算EMA平滑后的权重
    else:
        ema_helper = None

    # new filefold for save model pt 保存model的目录
    model_save = exp_dir / 'models' / (timestamp + '/')
    if not os.path.exists(model_save):
        os.makedirs(model_save)

    # config.training.n_epochs = 1
    for epoch in range(1, config.training.n_epochs + 1): # 训练轮数
        logger.info("<----Epoch-{}---->".format(epoch))
        for _, (trainx, head) in enumerate(dataloader): # 读取数据
            x0 = trainx.cuda()
            head = head.cuda()
            t = torch.randint(low=0, high=n_steps,
                              size=(len(x0) // 2 + 1, )).cuda() # 在low-high范围内，随机采样size个t
            t = torch.cat([t, n_steps - t - 1], dim=0)[:len(x0)] # 对称时间步t，取前len(x0)个
            # Get the noised images (xt) and the noise (our target)
            xt, noise = q_xt_x0(x0, t) # 计算时间步t的带噪轨迹、噪声
            # Run xt through the network to get its predictions
            pred_noise = unet(xt.float(), t, head) # 预测噪声
            # Compare the predictions with the targets
            loss = F.mse_loss(noise.float(), pred_noise) # 均方误差损失
            # Store the loss for later viewing
            losses.append(loss.item()) # 存储损失
            optim.zero_grad() # 清空梯度
            loss.backward() # 反向传播
            optim.step() # 更新模型权重
            if config.model.ema:
                ema_helper.update(unet) # 更新EMA
        if (epoch) % 10 == 0: # 每10轮保存一次模型
            m_path = model_save / f"unet_{epoch}.pt"
            torch.save(unet.state_dict(), m_path)
            m_path = exp_dir / 'results' / f"loss_{epoch}.npy"
            np.save(m_path, np.array(losses))


if __name__ == "__main__":
    # Load configuration 参数对象
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)
    config = SimpleNamespace(**temp)

    # 存储目录
    root_dir = Path(__name__).resolve().parents[0]
    result_name = '{}_steps={}_len={}_{}_bs={}'.format(
        config.data.dataset, config.diffusion.num_diffusion_timesteps,
        config.data.traj_length, config.diffusion.beta_end,
        config.training.batch_size)
    exp_dir = root_dir / "DiffTraj" / result_name
    # 实验目录
    for d in ["results", "models", "logs","Files"]:
        os.makedirs(exp_dir / d, exist_ok=True)
    print("All files saved path ---->>", exp_dir)
    # 备份代码
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    files_save = exp_dir / 'Files' / (timestamp + '/')
    if not os.path.exists(files_save):
        os.makedirs(files_save)
    shutil.copy('./utils/config.py', files_save)
    shutil.copy('./utils/Traj_UNet.py', files_save)

    # 初始化日志
    logger = Logger(
        __name__,
        log_path=exp_dir / "logs" / (timestamp + '.log'),
        colorize=True,
    )
    log_info(config, logger)

    # 启动主程序
    main(config, logger, exp_dir)
