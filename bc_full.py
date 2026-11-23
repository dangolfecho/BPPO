"""BC on the FULL Minari HDF5 dataset, with scaled actions and normalized states.

- reads the HDF5 directly (no d3rlpy): state = obs[:-1], action = actions, same episode/step
- actions are scaled to [-1, 1] with the dataset's own action bounds (utils.scale_action) and
  unscaled (utils.unscale_action) before env.step
- the whole dataset goes into a disk-backed float32 cache (np.load mmap), so peak RAM stays small
- checkpoint selection is by held-out NLL on episodes never trained on, not by rollout score
Usage: python bc_full.py [--steps 30000] [--eval_every 10000] [--rollout_eps 3] [--max_episodes N]
"""
import argparse, json, os, sys, time
import h5py
import numpy as np
import torch
from tqdm import tqdm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # this script lives next to net.py / utils.py
from net import GaussPolicyMLP
from utils import orthogonal_initWeights, log_prob_func, scale_action, unscale_action

p = argparse.ArgumentParser()
p.add_argument('--dataset', default='QuadX-Hover-v4/dataset-14-fixednew-10M-v3')
p.add_argument('--steps', type=int, default=30000)
p.add_argument('--eval_every', type=int, default=10000)
p.add_argument('--rollout_eps', type=int, default=3)
p.add_argument('--max_episodes', type=int, default=None, help='None = the full dataset')
p.add_argument('--holdout', type=int, default=100, help='episodes kept out of training for validation')
p.add_argument('--cache_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bc_cache'))
p.add_argument('--out', default='bc_scaled.pt')
p.add_argument('--no_norm', action='store_true')
p.add_argument('--threads', type=int, default=6)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
p.add_argument('--gpu_data', action='store_true', help='keep the whole cached dataset on the device (~3 GB VRAM for the full dataset)')
p.add_argument('--batch_size', type=int, default=512)
args = p.parse_args()
torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
device = torch.device(args.device)

root = os.environ.get('MINARI_DATASETS_PATH', os.path.expanduser('~/.minari/datasets'))
h5path = os.path.join(root, args.dataset, 'data', 'main_data.hdf5')
meta = json.load(open(os.path.join(root, args.dataset, 'data', 'metadata.json')))
aspace = json.loads(meta['action_space']); spec = json.loads(meta['env_spec'])
low = np.array(aspace['low'], np.float32); high = np.array(aspace['high'], np.float32)   # env action bounds
cdir = os.path.join(args.cache_dir, args.dataset.replace('/', '_') + '_' + (str(args.max_episodes) if args.max_episodes else 'full'))


def build_cache():
    os.makedirs(cdir, exist_ok=True)
    with h5py.File(h5path, 'r') as f:
        keys = sorted(f.keys(), key=lambda k: int(k.split('_')[1]))
        if args.max_episodes and args.max_episodes < len(keys):
            keys = [keys[i] for i in np.sort(np.random.default_rng(args.seed).choice(len(keys), args.max_episodes, replace=False))]
        hold = set(np.random.default_rng(args.seed + 1).choice(len(keys), args.holdout, replace=False).tolist())
        tr = [k for i, k in enumerate(keys) if i not in hold]; ho = [k for i, k in enumerate(keys) if i in hold]
        lens = np.array([f[k]['actions'].shape[0] for k in tr]); N = int(lens.sum())
        od, ad = f[tr[0]]['observations'].shape[1], f[tr[0]]['actions'].shape[1]
        obs = np.lib.format.open_memmap(cdir + '/obs.npy', 'w+', np.float32, (N, od))
        act = np.lib.format.open_memmap(cdir + '/act.npy', 'w+', np.float32, (N, ad))
        off = 0; s1 = np.zeros(od); s2 = np.zeros(od)
        for n, k in enumerate(tqdm(tr, desc='building cache', mininterval=30)):
            g = f[k]; T = int(lens[n]); o = g['observations'][:T].astype(np.float32)        # obs[:-1]: T states for T actions
            obs[off:off + T] = o; act[off:off + T] = scale_action(g['actions'][:], low, high).astype(np.float32)
            o64 = o.astype(np.float64); s1 += o64.sum(0); s2 += (o64 ** 2).sum(0); off += T
            if n % 2000 == 1999: obs.flush(); act.flush()
        obs.flush(); act.flush(); mean = s1 / N; std = np.sqrt(np.maximum(s2 / N - mean ** 2, 0)) + 1e-6
        Sh = np.concatenate([f[k]['observations'][:-1] for k in ho]).astype(np.float32)
        Ah = scale_action(np.concatenate([f[k]['actions'][:] for k in ho]), low, high).astype(np.float32)
        np.savez(cdir + '/extra.npz', mean=mean.astype(np.float32), std=std.astype(np.float32), Sh=Sh, Ah=Ah, N=N, E=len(tr))
    open(cdir + '/READY', 'w').write('ok')


if not os.path.exists(cdir + '/READY'):
    t0 = time.time(); build_cache(); print(f'cache built in {time.time() - t0:.0f}s at {cdir}', flush=True)
obs = np.load(cdir + '/obs.npy', mmap_mode='r')
act = np.load(cdir + '/act.npy', mmap_mode='r')
ex = np.load(cdir + '/extra.npz')
N = len(obs)
mean, std = ex['mean'], ex['std']
if args.no_norm:
    mean, std = np.zeros_like(mean), np.ones_like(std)
print(f'training on {N} transitions from {int(ex["E"])} episodes ({len(ex["Sh"])} held-out transitions); '
      + ('FULL dataset' if not args.max_episodes else f'SUBSET (--max_episodes {args.max_episodes})'), flush=True)
mean_t = torch.tensor(mean).to(device)
std_t = torch.tensor(std).to(device)
Xv = (torch.tensor(ex['Sh']).to(device) - mean_t) / std_t
Yv = torch.tensor(ex['Ah']).to(device)

pi = GaussPolicyMLP(obs.shape[1], 1024, 2, act.shape[1]).to(device)
orthogonal_initWeights(pi)
opt = torch.optim.Adam(pi.parameters(), lr=1e-4)


def rollout(n):
    import copy, gymnasium as gym, PyFlyt.gym_envs               # only needed for evaluation
    pi_cpu = copy.deepcopy(pi).cpu()                             # one-sample inference is faster on CPU than a GPU round trip per step
    env = gym.make(spec['id'], **spec['kwargs'])
    env.reset(seed=8)
    res = []
    for _ in range(n):
        s, _ = env.reset()
        R = 0.
        L = 0
        done = False
        while not done:
            with torch.no_grad():
                a = pi_cpu(torch.FloatTensor((s.reshape(1, -1) - mean) / std)).mean.clamp(-1, 1).numpy().flatten()
            s, r, te, tr, _ = env.step(unscale_action(a, low, high))
            R += r
            L += 1
            done = te or tr
        res.append((L, int(te), round(R)))
    env.close()
    return res


def save(path, step, nll):
    torch.save({'policy': {k: v.cpu() for k, v in pi.state_dict().items()}, 'mean': mean, 'std': std, 'low': low, 'high': high, 'step': step, 'val_nll': nll, 'dataset': args.dataset}, path)


if args.gpu_data:      # whole cache resident on the device, copied in chunks (~3 GB for the full fixednew dataset)
    obs_g = torch.empty((N, obs.shape[1]), dtype=torch.float32, device=device); act_g = torch.empty((N, act.shape[1]), dtype=torch.float32, device=device)
    for lo in range(0, N, 2_000_000):
        hi = min(lo + 2_000_000, N)
        obs_g[lo:hi] = torch.from_numpy(np.ascontiguousarray(obs[lo:hi])).to(device); act_g[lo:hi] = torch.from_numpy(np.ascontiguousarray(act[lo:hi])).to(device)
    print(f'dataset resident on {device}: {(obs_g.numel() + act_g.numel()) * 4 / 1e9:.1f} GB', flush=True)
best = float('inf'); rng = np.random.default_rng(args.seed); t0 = time.time()
for k in range(args.steps + 1):
    if k % args.eval_every == 0:
        with torch.no_grad():
            d = pi(Xv)
            nll = -log_prob_func(d, Yv).mean().item()
            r2 = (1 - ((d.mean - Yv) ** 2).mean(0) / Yv.var(0)).cpu().numpy().round(2)
        line = f'step {k}: held-out NLL {nll:.3f} R2 {r2} [{time.time() - t0:.0f}s]'
        if args.rollout_eps:
            line += ' | rollouts (len, crashed, return): ' + str(rollout(args.rollout_eps))
        print(line, flush=True)
        if nll < best: 
            best = nll
            save(args.out, k, nll)
    if args.gpu_data:
        idx = torch.randint(0, N, (args.batch_size,), device=device)
        x = (obs_g[idx] - mean_t) / std_t
        y = act_g[idx]
    else:
        idx = np.sort(rng.integers(0, N, args.batch_size))       # sorted for disk locality; order inside a batch is irrelevant
        x = torch.from_numpy(((obs[idx] - mean) / std).astype(np.float32)).to(device)
        y = torch.from_numpy(np.ascontiguousarray(act[idx])).to(device)
    loss = -log_prob_func(pi(x), y).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad(): nll = -log_prob_func(pi(Xv), Yv).mean().item()   # val NLL of the final policy (not the last eval's)
save(args.out.replace('.pt', '_last.pt'), args.steps, nll)
print(f'done. best held-out NLL {best:.3f} saved to {args.out}; last policy in {args.out.replace(".pt", "_last.pt")}')
