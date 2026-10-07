import h5py, numpy as np, torch
from dp_manip.data import read_dataset_info, RGBWindowDataset
from dp_manip.policy import DiffusionPolicy
DATA='/userhome/cs5/u3684238/7606C/maniskill-demogen/data/dataset'
VAL='val/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5'
info=read_dataset_info(f'{DATA}/{VAL}',50); ds=RGBWindowDataset(info,2,16,preload=False)
sw={}
with h5py.File(info.path,'r') as f:
    for ei,ep in enumerate(info.episodes):
        sw[ei]=int(np.argmax(np.diff(np.asarray(f[ep.group]['actions'])[:,3])<-0.5))+1
near=[]; ctrl=[]
for i,(ei,t) in enumerate(ds.index):
    s=sw[ei]; st=t-1
    if s-6 <= st <= s-1: near.append(i)      # 切换点落在预测块的中间偏后
    elif st+16 <= s-25: ctrl.append(i)
near=near[:8]; ctrl=ctrl[:8]
idx=near+ctrl
print("windows: near",len(near),"ctrl",len(ctrl),flush=True)

def block(pol,b):
    rgb=torch.stack([torch.as_tensor(ds[i]['rgb']) for i in b]).float()
    prop=torch.stack([torch.as_tensor(ds[i]['proprio']) for i in b]).float()
    gen=torch.Generator().manual_seed(7)
    with torch.no_grad():
        obs=pol.observation_features(rgb,prop)
        pol.noise_scheduler.set_timesteps(pol.num_inference_iters)
        pol.noise_scheduler.alphas_cumprod=pol.noise_scheduler.alphas_cumprod.to('cpu')
        s=torch.randn(len(b),16,4,generator=gen)
        for ts in pol.noise_scheduler.timesteps:
            s=pol.noise_scheduler.step(pol.noise_predictor(s,ts,obs),ts,s,generator=gen).prev_sample
        return pol.unnormalize_action(s).numpy()

for tag,run in (('unet','pickcube_rgb_unet_n100_s1'),('transformer','pickcube_rgb_transformer_n100_s1'),('mlp','pickcube_rgb_mlp_n100_s1')):
    pol=DiffusionPolicy.from_checkpoint(torch.load(f'/userhome/cs5/u3684238/dp-runs-pickcube/{run}/checkpoints/final.pt',map_location='cpu',weights_only=False),'cpu').eval()
    P=block(pol,idx)
    print(f"--- {tag}",flush=True)
    for name,rows in (('near',range(len(near))),('control',range(len(near),len(idx)))):
        j=np.array([P[j] for j in rows])          # (n,16,4) 预测的 gripper
        g=j[...,3]
        maxjump=np.abs(np.diff(g,axis=1)).max(1)
        mid=((np.abs(g)<0.9)).mean(1)*100          # % 未饱和(介于 ±0.9 之间)的步
        rng=g.max(1)-g.min(1)                      # 该块里夹爪覆盖的幅度
        print(f"    {name:8s} |g| 平均={np.abs(g).mean():.3f}  幅度(max-min)均值={rng.mean():.3f}  "
              f"最大单步跳变均值={maxjump.mean():.3f}  未饱和步占比={mid.mean():.1f}%",flush=True)
