import h5py, numpy as np, torch
from dp_manip.data import read_dataset_info, RGBWindowDataset
from dp_manip.policy import DiffusionPolicy

DATA='/userhome/cs5/u3684238/7606C/maniskill-demogen/data/dataset'
VAL='val/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5'
info=read_dataset_info(f'{DATA}/{VAL}', 50)
ds=RGBWindowDataset(info,2,16,preload=False)
# 每条示范的夹爪切换步
sw={}
with h5py.File(info.path,'r') as f:
    for ei,ep in enumerate(info.episodes):
        a=np.asarray(f[ep.group]['actions'])
        sw[ei]=int(np.argmax(np.diff(a[:,3])<-0.5))+1
near=[]; ctrl=[]
for i,(ei,t) in enumerate(ds.index):
    s=sw[ei]; start=t-1
    if s-14 <= start <= s: near.append(i)
    elif start+16 <= s-25: ctrl.append(i)
near=near[:160]; ctrl=ctrl[:160]
print(f"windows: near={len(near)} control={len(ctrl)}", flush=True)

MODELS=[('unet','pickcube_rgb_unet_n100_s1'),('transformer','pickcube_rgb_transformer_n100_s1'),('mlp','pickcube_rgb_mlp_n100_s1')]
def batches(idx, bs=32):
    for k in range(0,len(idx),bs): yield idx[k:k+bs]

def collect(policy, idx, tag):
    rec=[]   # (d, dim_group, sq_err)
    for b in batches(idx):
        rgb=torch.stack([torch.as_tensor(ds[i]['rgb']) for i in b]).float()
        prop=torch.stack([torch.as_tensor(ds[i]['proprio']) for i in b]).float()
        act=torch.stack([torch.as_tensor(ds[i]['actions']) for i in b]).float()
        gen=torch.Generator().manual_seed(1234)
        with torch.no_grad():
            obs=policy.observation_features(rgb,prop)
            an=policy.normalize_action(act)
            ts=torch.randint(0,100,(len(b),),generator=gen)
            noise=torch.randn(an.shape, generator=gen)

            noisy=policy.noise_scheduler.add_noise(an, noise, ts)
            pred=policy.noise_predictor(noisy, ts, obs)
        se=((pred-noise)**2).numpy()          # (B,16,4)
        for j,i in enumerate(b):
            ei,t=ds.index[i]; s=sw[ei]
            for step in range(16):
                gi=min(max(t-1+step,0), info.episodes[ei].length-1)
                d=gi-s
                rec.append((d, se[j,step,3], se[j,step,:3].mean()))
    return rec

def summarize(rec, tag):
    out={}
    for name,sel in (('near', lambda d: -14<=d<=0), ('control', lambda d: d<-25)):
        g=[r[1] for r in rec if sel(r[0])]; p=[r[2] for r in rec if sel(r[0])]
        out[name]=(np.mean(g), np.mean(p), len(g))
    # 按距离细分（near 内）
    steps={}
    for d in (-8,-4,-2,-1,0,1,2,4,8):
        g=[r[1] for r in rec if r[0]==d]
        if g: steps[d]=np.mean(g)
    print(f"{tag:12s} gripper: near={out['near'][0]:.4f} control={out['control'][0]:.4f} "
          f"ratio={out['near'][0]/out['control'][0]:5.2f} | pos: near={out['near'][1]:.4f} control={out['control'][1]:.4f} "
          f"ratio={out['near'][1]/out['control'][1]:5.2f} | n={out['near'][2]}", flush=True)
    print(f"{'':12s} 逐距离 gripper 误差: "+"  ".join(f"d={d:+d}:{v:.4f}" for d,v in steps.items()), flush=True)

allrec={}
for tag,run in MODELS:
    ck=torch.load(f'/userhome/cs5/u3684238/dp-runs-pickcube/{run}/checkpoints/final.pt',map_location='cpu',weights_only=False)
    pol=DiffusionPolicy.from_checkpoint(ck,'cpu'); pol.eval()
    rec=collect(pol, near, tag)+collect(pol, ctrl, tag)
    allrec[tag]=rec
    summarize(rec, tag)
np.save('/tmp/a1b_rec.npy', np.array([(t,r[0],r[1],r[2]) for t,rs in allrec.items() for r in rs]))
