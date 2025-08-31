import math
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import torch.nn.functional as F


def get_timestep_embedding(timesteps, embedding_dim):
    """
        将时间步转换为正弦余弦嵌入，将离散的时间步信息映射到一个连续高维的向量空间
    """
    assert len(timesteps.shape) == 1
    assert torch.all(timesteps >= 0), f"Timesteps contain negative values: {timesteps}"
    assert torch.all(timesteps < 500), f"Timesteps exceed max value (500): {timesteps.max()}"
    timesteps = timesteps.long()  # 转换为整数
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1) # 频率缩放因子
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb) # 频率向量
    emb = emb.to(device=timesteps.device) # 调整设备
    emb = timesteps.float()[:, None] * emb[None, :] # 计算嵌入矩阵
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1) # 正余弦拼接
    if embedding_dim % 2 == 1:  # zero pad 奇数维度补零
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Attention(nn.Module):
    """
    注意力机制类

    计算输入特征在不同属性上（或时间步、通道等维度）的重要性权重，
    用于加权输入数据，突出模型认为更重要的部分
    """
    def __init__(self, embedding_dim):
        super(Attention, self).__init__()
        # 全连接层，将数据映射到1维，计算每个属性的注意力得分，越高越重要
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, num_attributes, embedding_dim)
        # 对x的最后一维进行线性变换，计算每个属性的注意力分数
        weights = self.fc(x)  # shape: (batch_size, num_attributes, 1)
        # apply softmax along the attributes dimension
        # 使weights变成权重和为1的概率分布
        weights = F.softmax(weights, dim=1)
        return weights


class WideAndDeep(nn.Module):
    """
    Wide部分（线性部分）学习连续特征的线性关系，
    Deep部分（深度神经网络）学习类别特征的高阶特征交互，使用嵌入层+全连接层

    output：将Wide和Deep部分的输出相加
    """
    def __init__(self, embedding_dim=128, hidden_dim=256):
        """
        Args:
            embedding_dim：最终输出的嵌入维度
            hidden_dim：深度网络的隐藏层维度
        """
        super(WideAndDeep, self).__init__()

        # Wide part (linear model for continuous attributes)
        # 输入5个连续特征，映射到embedding_dim维度
        self.wide_fc = nn.Linear(5, embedding_dim)

        # Deep part (neural network for categorical attributes)
        self.depature_embedding = nn.Embedding(288, hidden_dim) # 288个出发时间，映射到hidden_dim维度
        self.sid_embedding = nn.Embedding(257, hidden_dim) # 257个起点id，映射到hidden_dim维度
        self.eid_embedding = nn.Embedding(257, hidden_dim) # 257个终点id，映射到hidden_dim维度
        self.deep_fc1 = nn.Linear(hidden_dim*3, embedding_dim) # 输入hidden_dim*3维度，降维到embedding_dim维度
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim) # 线性层，维度保持在embedding_dim

    def forward(self, attr):
        # Continuous attributes
        # 第1到5列连续特征
        continuous_attrs = attr[:, 1:6]

        # Categorical attributes
        depature, sid, eid = attr[:, 0].long(), attr[:, 6].long(), attr[:, 7].long()

        # Wide part
        wide_out = self.wide_fc(continuous_attrs) # 映射到embedding_dim维度

        # Deep part
        # 都映射到hidden_dim维度
        depature_embed = self.depature_embedding(depature)
        sid_embed = self.sid_embedding(sid)
        eid_embed = self.eid_embedding(eid)
        # 连接三个特征，形状为(batch_size, hidden_dim*3)
        categorical_embed = torch.cat(
            (depature_embed, sid_embed, eid_embed), dim=1)
        # 将为到embedding_dim，ReLU激活
        deep_out = F.relu(self.deep_fc1(categorical_embed))
        # 保持维度不变
        deep_out = self.deep_fc2(deep_out)
        # Combine wide and deep embeddings
        # 相加，形状为(batch_size, embedding_dim)
        combined_embed = wide_out + deep_out

        return combined_embed


def nonlinearity(x):
    """
    实现swish激活函数，平滑且非单调
    在x=0处连续可微，在x<0时递减，允许负值通过
    """
    # swish
    # torch.nn.SiLU()
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    """
    对输入数据归一化，提高模型稳定性，加快收敛
    组归一化代替批归一化，适合小batch训练，不依赖batch_size
    """
    return torch.nn.GroupNorm(num_groups=32, # 将in_channels个通道分成num_groups组，每组独立归一化
                              num_channels=in_channels, # 归一化的输出通道数
                              eps=1e-6, # 防止除零错误的极小值
                              affine=True) # 允许缩放因子和偏移银子作为可学习参数


class Upsample(nn.Module):
    """
    实现1D上采样
    """
    def __init__(self, in_channels, with_conv=True):
        """
        Args:
            in_channels：输入数据通道数，特征维度
            with_conv：是否采样后添加1D卷积层，默认True
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # 不改变通道数，用于平滑上采样后的数据，避免插值导致像素块效应
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        # 上采样
        x = torch.nn.functional.interpolate(x,
                                            scale_factor=2.0, # 扩大2倍
                                            mode="nearest") # 最邻近插值，赋值最近的像素值
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    实现1D下采样
    """
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            # torch不支持非对称padding，手动填充
            self.conv = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2, # 下采样比例为2倍
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            # 在输入两端分别填充一个0，确保卷积后大小正确
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            # 3*1卷积核，步长2，进行下采样，比平均池化有更强的特征提取能力
            x = self.conv(x)
        else:
            # 平均池化，2倍下采样
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    """
    1D残差块，深度网络中的基本单元
    结合 归一化、残差连接、卷积 进行特征变换，加入时间步嵌入
    """
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 conv_shortcut=False,
                 dropout=0.1,
                 temb_channels=512):
        """
        Args：
            conv_shortcut：是否使用3*1卷积进行shortcut连接，默认False
            dropout：dropout概率，默认0.1，防止过拟合
            temb_channels：时间步嵌入的通道数，默认512
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels) # 组归一化
        self.conv1 = torch.nn.Conv1d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1) # 输入尺度不变，提取局部特征
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels) # 将时间步投影到out_channels维度

        self.norm2 = Normalize(out_channels) # 组归一化
        self.dropout = torch.nn.Dropout(dropout) # dropout
        self.conv2 = torch.nn.Conv1d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1) # 输入尺度不变，提取局部特征

        if self.in_channels != self.out_channels: # 通道变换，保证不同分辨率特征可以正确残差相加
            if self.use_conv_shortcut: # 使用3*1卷积进行通道变换
                self.conv_shortcut = torch.nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else: # 使用1*1逐点卷积进行变换
                self.nin_shortcut = torch.nn.Conv1d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x

        # 第一层计算，归一化、swish激活、卷积
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # 添加时间步嵌入，允许temb影响特征映射
        # swish激活temb、投影到维度、广播相加
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        # 第二层计算，归一化、swish激活、dropout、卷积提取特征
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 变换x维度
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h # 残差连接，缓解梯度消失


class AttnBlock(nn.Module):
    """
    1D自注意力机制块
    用于特征提取
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels) # 组归一化
        # 1*1卷积，相当于linear projection，不改变时间步维度
        # query，查询的特征
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # key，被查询的特征
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # value，要传播的信息
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # 投影层，将注意力计算后的特征重新映射到原始空间
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_) # 归一化，提升模型稳定性
        # 计算query、key、value
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        # 计算query和key之间的注意力分数，QK^T
        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,hw,c
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # 归一化注意力分数
        w_ = w_ * (int(c)**(-0.5)) # 乘以缩放因子，防止梯度消失或爆炸
        w_ = torch.nn.functional.softmax(w_, dim=2) # 得到注意力权重
        # attend to values
        # 对value加权求和
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        # 投影输出
        h_ = self.proj_out(h_)

        return x + h_ # 残差连接


class Model(nn.Module):
    """
    UNet变体
    结合 Resnet残差块、自注意力机制、下采样、上采样
    """
    def __init__(self, config):
        """
            config：配置模型参数
        """
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(
            config.model.ch_mult)
        print(f"self.ch: {ch}")
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.traj_length
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian': # 贝叶斯模型，用于估计噪声方差
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))

        self.ch = ch
        self.temb_ch = self.ch * 4 # 将时间步映射到高维
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([ # 两个全连接层进行线性变换
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling 下采样
        # 初始卷积，把in_channels投影到ch维度
        self.conv_in = torch.nn.Conv1d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1, ) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        # 构造Downsampling层，构造多个ResnetBlock
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level] # 当前层的输入通道
            block_out = ch * ch_mult[i_level] # 当前层的输出通道
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(in_channels=block_in,
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: # 指定的attn_resolutions位置，添加注意力层
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1: # 不是最后一层，添加Downsample进行降维
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2 # 时间步减半
            self.down.append(down)

        # middle 中间层，Resnet+AttnBlock+Resnet
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in) # 让每个时间步都能关注整个时间序列
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling 上采样
        self.up = nn.ModuleList()
        # 逆序遍历num_resolutions
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(in_channels=block_in + skip_in, # 残差连接
                                out_channels=block_out,
                                temb_channels=self.temb_ch,
                                dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions: # attn_resolutions处使用注意力层
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0: # 不是最后一层
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2 # 时间步长变为2倍
            self.up.insert(0, up)  # prepend to get consistent order

        # end 输出层
        self.norm_out = Normalize(block_in) # 组归一化
        # 将block_in变换到out_ch
        self.conv_out = torch.nn.Conv1d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, extra_embed=None):
        """
        Args：
            x：输入数据，形状(batch_size, in_channels, resolution)
            t：时间步
            extra_embed：额外的嵌入信息（可选）
        """
        assert x.shape[2] == self.resolution

        # timestep embedding 时间步嵌入
        # print(f"t.shape: {t.shape}")
        # print(f"t min: {t.min().item()}, t max: {t.max().item()}")
        temb = get_timestep_embedding(t, self.ch) # 时间步嵌入，映射到ch维
        temb = self.temb.dense[0](temb) # 第一层MLP线性变换
        temb = nonlinearity(temb) # swish激活
        temb = self.temb.dense[1](temb) # 第二层MLP线性变换
        if extra_embed is not None:
            temb = temb + extra_embed

        # downsampling 下采样
        hs = [self.conv_in(x)] # 初始卷积处理x，提取初步特征
        # print(hs[-1].shape)
        for i_level in range(self.num_resolutions): # 多尺度特征提取
            for i_block in range(self.num_res_blocks): # 多个ResnetBlock
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # print(i_level, i_block, h.shape)
                if len(self.down[i_level].attn) > 0: # 该层有注意力机制
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h) # 记录h，用于残差连接
            if i_level != self.num_resolutions - 1: # 不是最后一层
                hs.append(self.down[i_level].downsample(hs[-1])) # 进行下采样

        # middle 中间层
        # print(hs[-1].shape)
        # print(len(hs))
        h = hs[-1]  # [10, 256, 4, 4]
        h = self.mid.block_1(h, temb) # ResnetBlock提取特征
        h = self.mid.attn_1(h) # AttnBlock提取全局信息
        h = self.mid.block_2(h, temb) # ResnetBlock
        # print(h.shape)

        # upsampling 上采样
        for i_level in reversed(range(self.num_resolutions)): # 逆序遍历
            for i_block in range(self.num_res_blocks + 1):
                ht = hs.pop() # 取出对应的残差连接
                if ht.size(-1) != h.size(-1): # 填充
                    h = torch.nn.functional.pad(h,
                                                (0, ht.size(-1) - h.size(-1)))
                # 残差连接/跳跃连接，让高层特征融合底层信息
                h = self.up[i_level].block[i_block](torch.cat([h, ht], dim=1),
                                                    temb)
                # print(i_level, i_block, h.shape)
                if len(self.up[i_level].attn) > 0: #该层有注意力机制
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0: # 不是最后一层
                h = self.up[i_level].upsample(h) # 进行上采样

        # end 输出
        h = self.norm_out(h) # 归一化
        h = nonlinearity(h) # swish激活
        h = self.conv_out(h) # 卷积映射到out_ch
        return h


class Guide_UNet(nn.Module):
    """
    带引导机制的UNet变体，引入额外的引导信息（attr相关的嵌入）
    结合条件和无条件的UNet输出
    """
    def __init__(self, config):
        super(Guide_UNet, self).__init__()
        self.config = config
        self.ch = config.model.ch * 4
        self.attr_dim = config.model.attr_dim # head属性维度
        self.guidance_scale = config.model.guidance_scale # 引导强度
        self.unet = Model(config)
        # self.guide_emb = Guide_Embedding(self.attr_dim, self.ch)
        # self.place_emb = Place_Embedding(self.attr_dim, self.ch)
        self.guide_emb = WideAndDeep(self.ch) # 处理有条件信息
        self.place_emb = WideAndDeep(self.ch) # 处理无条件信息

    def forward(self, x, t, attr):
        """
        Args：
             x：输入数据（噪声）
             t：时间步
             atrr：条件引导信息（head）
        """
        guide_emb = self.guide_emb(attr) # 有条件的WideAndDeep层
        place_vector = torch.zeros(attr.shape, device=attr.device) # 全零张量
        place_emb = self.place_emb(place_vector) # 无条件的WideAndDeep层
        cond_noise = self.unet(x, t, guide_emb) # 有条件预测noise
        uncond_noise = self.unet(x, t, place_emb) # 无条件预测noise
        # CFG计算预测噪声
        pred_noise = cond_noise + self.guidance_scale * (cond_noise -
                                                         uncond_noise)
        return pred_noise


if __name__ == '__main__':
    """
    测试Guide_UNet
    """
    from utils.config import args

    # 将args转换为SimpleNamespace，用于将字典转换为对象，使用.访问属性
    temp = {}
    for k, v in args.items():
        temp[k] = SimpleNamespace(**v)

    config = SimpleNamespace(**temp)

    t = torch.randn(10) # 10个时间步
    # 全零head特征
    depature = torch.zeros(10)
    avg_dis = torch.zeros(10)
    avg_speed = torch.zeros(10)
    total_dis = torch.zeros(10)
    total_time = torch.zeros(10)
    total_len = torch.zeros(10)
    sid = torch.zeros(10)
    eid = torch.zeros(10)
    attr = torch.stack(
        [depature, total_dis, total_time, total_len, avg_dis, avg_speed, sid, eid], dim=1)

    # 实例化Guide_UNet
    unet = Guide_UNet(config)

    x = torch.randn(10, 2, 200) # 随机张量

    total_params = sum(p.numel() for p in unet.parameters()) # unet的总参数
    print(f'{total_params:,} total parameters.')

    out = unet(x, t, attr) # 前向传播
    print(out.shape) # 预测噪声值
