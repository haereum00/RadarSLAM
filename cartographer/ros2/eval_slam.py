import sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
s, outdir = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else 'output_rig2'
d = np.load(f'data/{s}.npz'); gt_t = d['time']; gt_xy = d['pose_gt'][:, :2]
def load_tum(p):
    a = np.loadtxt(p); return a[:, 0], a[:, 1:3], a[:, 4:8]
def align(src, dst):
    ms, md = src.mean(0), dst.mean(0); U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    Rm = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt; return Rm, md - Rm @ ms
res = {}; traj = {}
for name, path in [('odom', f'{OUT}/odometry/{s}.txt'), ('slam', f'{OUT}/carto/output/{s}.txt')]:
    t, xy, _ = load_tum(path); keep = (t >= gt_t[0]) & (t <= gt_t[-1]); t, xy = t[keep], xy[keep]
    g = np.stack([np.interp(t, gt_t, gt_xy[:, k]) for k in range(2)], 1)
    Rm, tr = align(xy, g); xa = xy @ Rm.T + tr; e = np.linalg.norm(xa - g, axis=1)
    path_len = np.linalg.norm(np.diff(g, axis=0), axis=1).sum()
    res[name] = dict(ate_rmse=float(np.sqrt(np.mean(e**2))), ate_max=float(e.max()), final=float(e[-1]), path=float(path_len))
    traj[name] = xa
y = __import__('yaml').safe_load(open(f'{OUT}/carto/output/{s}_map.yaml')); img = np.array(Image.open(f'{OUT}/carto/output/{s}_map.pgm'))
occ = np.argwhere(img < 100); res_m = y['resolution']; ox, oy = y['origin'][:2]
mx = ox + (occ[:, 1] + 0.5) * res_m; my = oy + (img.shape[0] - occ[:, 0] - 0.5) * res_m
t, xy, _ = load_tum(f'{OUT}/carto/output/{s}.txt'); keep = (t >= gt_t[0]) & (t <= gt_t[-1])
g = np.stack([np.interp(t[keep], gt_t, gt_xy[:, k]) for k in range(2)], 1); Rm, tr = align(xy[keep], g)
mp = np.stack([mx, my], 1) @ Rm.T + tr
np.savez(f'{outdir}/{s}_slam.npz', gt=gt_xy, odom=traj['odom'], slam=traj['slam'], map=mp)
for theme, bg, fg in [('light', '#ffffff', '#16212a'), ('dark', '#161d22', '#e3e9ed')]:
    fig, axs = plt.subplots(1, 2, figsize=(11, 6), facecolor=bg)
    for ax, key, title in [(axs[0], 'odom', 'Radar odometry only'), (axs[1], 'slam', 'Radar SLAM (Cartographer)')]:
        ax.set_facecolor(bg)
        if key == 'slam': ax.scatter(mp[:, 0], mp[:, 1], s=0.3, c='#8a99a3', linewidths=0, label='SLAM map (radar)')
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], c='#0a8fa3', lw=2.2, label='FAST-LIO GT')
        ax.plot(traj[key][:, 0], traj[key][:, 1], c='#d9631e', lw=1.4, label=title.split(' (')[0])
        ax.set_aspect('equal'); ax.tick_params(colors=fg, labelsize=8); [sp_.set_color(fg) for sp_ in ax.spines.values()]
        ax.set_title(f"{title}\nATE {res[key]['ate_rmse']:.2f} m (max {res[key]['ate_max']:.2f} m)", color=fg, fontsize=11)
        leg = ax.legend(fontsize=8, loc='best', facecolor=bg, edgecolor=fg, markerscale=12)
        [tx.set_color(fg) for tx in leg.get_texts()]
    plt.tight_layout(); plt.savefig(f'{outdir}/{s}_slam_{theme}.png', dpi=80, facecolor=bg); plt.close()
json.dump(res, open(f'{outdir}/{s}_slam.json', 'w')); print(s, json.dumps(res))
