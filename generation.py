import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

from utils.Traj_UNet import *
from utils.config import args
from utils.utils import *

temp = {}
for k, v in args.items():
    temp[k] = SimpleNamespace(**v)

config = SimpleNamespace(**temp)

unet = Guide_UNet(config).cuda()
# load the model
prePath = '/home/hjy/DiffTraj'
# prePath = 'D:/MyProjects/PythonAbout/DiffusionModel/DiffTraj'
unet.load_state_dict(
    torch.load(prePath+'/DiffTraj/Wuhan_steps=500_len=200_0.05_bs=32/models/04-06-19-13-50/unet_180.pt'))

# %%
n_steps = config.diffusion.num_diffusion_timesteps
beta = torch.linspace(config.diffusion.beta_start,
                      config.diffusion.beta_end, n_steps).cuda()
alpha = 1. - beta
alpha_bar = torch.cumprod(alpha, dim=0)
lr = 2e-4  # Explore this - might want it lower when training on the full dataset

eta = 0.0
timesteps = 100
skip = n_steps // timesteps
seq = range(0, n_steps, skip)

# load head information for guide trajectory generation
batchsize = 500
head = np.load('dataset/head_lat16_lon16.npy',
               allow_pickle=True)
head = torch.from_numpy(head).float()
dataloader = DataLoader(head, batch_size=batchsize, shuffle=True, num_workers=0)


# the mean and std of head information, using for rescaling
hmean = [0,116.87192154692256,8299.930162552679,21.31005418422637,8.66673271499384,0.22034542712971672]
hstd = [1,319.88159197073,11643.630572869386,118.66998066873354,11.375444704159696,1.1671275986022498]

mean = np.array([114.40815562100842,30.456079474774274])
std = np.array([0.001550441880006359,0.0014920193948092162])
# the original mean and std of trajectory length, using for rescaling the trajectory length
len_mean = 7.6563997262149215  # Wuhan
len_std = 11.528600647824929  # Wuhan

Gen_traj = []
Gen_head = []
for i in tqdm(range(1)):
    head = next(iter(dataloader))
    lengths = head[:, 3]
    lengths = lengths * len_std + len_mean
    lengths = lengths.int()
    tes = head[:,:6].numpy()
    Gen_head.extend((tes*hstd+hmean))
    head = head.cuda()
    # Start with random noise
    x = torch.randn(batchsize, 2, config.data.traj_length).cuda()
    ims = []
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        with torch.no_grad():
            pred_noise = unet(x, t, head)
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


# Save Gen_traj to CSV
def save_trajectories_to_csv(trajectories, output_file='generated_trajectories.csv'):
    """
    Save generated trajectories to a CSV file in the format: id,lon,lat.

    Parameters:
    - trajectories: List of trajectories, each a NumPy array of shape (n_points, 2).
    - output_file: Path to save the CSV file.
    """
    data = []
    for traj_id, traj in enumerate(trajectories, 1):  # Start ID from 1
        for lon, lat in traj:
            data.append([traj_id, lon, lat])

    df = pd.DataFrame(data, columns=['id', 'lon', 'lat'])
    df.to_csv(output_file, index=False)
    print(f"Trajectories saved to {output_file}")


# Save the generated trajectories
save_trajectories_to_csv(Gen_traj)
