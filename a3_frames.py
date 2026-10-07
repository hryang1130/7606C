import h5py, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
P='/userhome/cs5/u3684238/7606C/maniskill-demogen/data/dataset/train/PickCube-v1/motionplanning/trajectory.state.pd_ee_delta_pos.physx_cpu.h5'
f=h5py.File(P,'r'); keys=sorted(f.keys(), key=lambda k:int(k.split('_')[1]))
sel=[keys[i] for i in np.linspace(0,len(keys)-1,6).astype(int)]
fig,axes=plt.subplots(len(sel),4,figsize=(4*3.0,6*3.0))
for r,k in enumerate(sel):
    rgb=np.asarray(f[k]['obs_rgb/rgb']); act=np.asarray(f[k]['actions'])
    sw=int(np.argmax(np.diff(act[:,3])<-0.5))+1
    idx=[0, max(sw-3,0), sw, len(act)-1]
    for c,i in enumerate(idx):
        ax=axes[r,c]; ax.imshow(rgb[i]); ax.set_xticks([]); ax.set_yticks([])
        if r==0: ax.set_title(["t=0 (start)","t=switch-3","t=switch (gripper closes)","t=end"][c], fontsize=11)
        ax.set_ylabel(f"{k}\nt={i}", fontsize=8)
    d=np.abs(rgb[sw].astype(int)-rgb[0].astype(int)).sum(-1)
    ax=axes[r,3]; 
    # 4x4 layer4 网格叠加在差异图上
    ax.imshow(rgb[sw]); ax.imshow(d, cmap='hot', alpha=0.45)
    ax.set_xticks([]); ax.set_yticks([])
    for g in range(1,4):
        ax.axvline(g*32-0.5, color='cyan', lw=0.8); ax.axhline(g*32-0.5, color='cyan', lw=0.8)
    if r==0: ax.set_title("|end - start| + layer4 4x4 grid (cyan)", fontsize=9)
fig.suptitle("PickCube 示范帧（128x128 base camera）：起点 / 闭合前 / 闭合瞬间 / 结束", fontsize=13)
fig.tight_layout()
out='/userhome/cs5/u3684238/7606C/report/figures/a3_frames.png'
fig.savefig(out, dpi=110, bbox_inches='tight'); print("wrote", out)
