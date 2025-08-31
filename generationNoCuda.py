import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.Traj_UNet import *
from utils.config import args

def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_xt(xt, noise, t, next_t, beta, eta=0):
    at = compute_alpha(beta, t.long())
    at_next = compute_alpha(beta, next_t.long())
    x0_t = (xt - noise * (1 - at).sqrt()) / at.sqrt()
    c1 = (eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
    c2 = ((1 - at_next) - c1 ** 2).sqrt()
    eps = torch.randn(xt.shape)
    xt_next = at_next.sqrt() * x0_t + c1 * eps + c2 * noise
    return xt_next

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)
config = SimpleNamespace(**temp)

unet = Guide_UNet(config)
# load the model
# prePath = '/home/hjy/DiffTraj'
prePath = 'D:/MyProjects/PythonAbout/DiffusionModel/MyDiffTraj'
unet.load_state_dict(
    torch.load(prePath+'/DiffTraj/traj200_head1616_dis100/Wuhan_steps=500_len=200_0.05_bs=32'
                       '/models/04-06-11-38-56/unet_200.pt',
               map_location=torch.device('cpu'),
               weights_only=True
               )
)

# %%
n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                      config.diffusion.beta_end, n_steps)
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta = 0.0
timesteps = 100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 500
head = np.load('dataset/wkhj/head_lat16_lon16.npy',
               allow_pickle=True)
head = torch.from_numpy(head).float()
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=0)


# the mean and std of head information, using for rescaling
hmean = [0,105.96585267360813,274.78201058201057,7.617989417989418,18.560685945370466,3.5673982428933195]
hstd = [1,218.00241542172074,328.1221270478287,11.565095002315651,15.309060366633098,9.608857778062685]

mean = np.array([114.40814661498356,30.45608020694078])
std = np.array([0.0015514781697125869,0.0014796285727747566])
# the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 7.617989417989418  # Wuhan
len_std = 11.565095002315651  # Wuhan

Gen_traj = []
Gen_head = []
for i in range(1):
    head = next(iter(dataloader))
    lengths = head[:, 3]
    lengths = lengths * len_std + len_mean
    lengths = lengths.int()
    tes = head[:,:6].numpy()
    Gen_head.extend((tes*hstd+hmean))
    head = head
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length)
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i)
        next_t = (torch.ones(n) * j)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
            # pred_noise = torch.randn(batchsize, 2, x.size(2))
            # print(pred_noise.shape)
            x = p_xt(x, pred_noise, t, next_t, beta, eta)
            if i % 10 == 0:
                ims.append(x.cpu().squeeze(0))
    trajs = ims[-1].cpu().numpy()
    trajs = trajs[:,:2,:]
    # resample the trajectory length
    for j in range(batchsize):
        new_traj = resample_trajectory(trajs[j].T, lengths[j])
        new_traj = new_traj * std + mean
        Gen_traj.append(new_traj)
    break

plt.figure(figsize=(8, 8))
for i in range(len(Gen_traj)):
    traj = Gen_traj[i]
    plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.1)
plt.tight_layout()
plt.title('gen_wkhj_traj')
plt.grid(True)
plt.savefig('gen_wkhj_traj.png')
plt.show()
