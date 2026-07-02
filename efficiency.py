"""Measured efficiency (referee: the cost paper must report real cost, not just big-O).
Per-round wall-clock and peak GPU memory for static / naive FT / memory / controller,
under the same streaming protocol. Backs the 'cost-optimal' claim with hardware numbers."""
import time, json
import numpy as np, torch
import learned_continual as L

def setup(ds, seed=0):
    df = L.load(ds); nu = int(df.user_idx.max()+1); npo = int(df.poi_idx.max()+1)
    S0 = L.make_samples(df[df.block=='T0'], npo); St = L.make_samples(df[df.block!='T0'], npo)
    warm = df[df.block=='T0'].user_idx.unique(); margs = (nu, npo, 128, 'gru')
    torch.manual_seed(seed); np.random.seed(seed)
    b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
    bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
    rng = np.random.default_rng(seed); pj = rng.choice(len(St['tgt']), 512, replace=False)
    fS = L.subset(S0, rng.choice(len(S0['tgt']), min(2000, len(S0['tgt'])), replace=False))
    rounds = np.array_split(np.arange(len(St['tgt'])), 15)
    return (bs, St, rounds, pj, fS, margs), warm, S0

def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter(); fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated()/1e6 if torch.cuda.is_available() else 0.0
    return dt, mem

out = {}
for ds in ['tky', 'brightkite']:
    com, warm, S0 = setup(ds)
    # warmup (CUDA init / kernels)
    L.run_policy('w', *com, (lambda r,a,e: False), 30, S0, 1024, np.random.default_rng(123), warm)
    res = {}
    dt, mem = timed(lambda: L.run_policy('s', *com, (lambda r,a,e: False), 30, S0, 1024, np.random.default_rng(123), warm)); res['static'] = [round(dt/15, 4), round(mem, 1)]
    dt, mem = timed(lambda: L.run_policy('a', *com, (lambda r,a,e: True), 30, S0, 1024, np.random.default_rng(123), warm, ft_lr=1e-3)); res['naive_FT'] = [round(dt/15, 4), round(mem, 1)]
    dt, mem = timed(lambda: L.run_giram('g', *com, warm, 0.5, 0.9)); res['memory'] = [round(dt/15, 4), round(mem, 1)]
    dt, mem = timed(lambda: L.run_controller('c', *com, ft_steps=30, lr0=1e-3, eps=0.005, warm=warm, grow=1.0)); res['controller'] = [round(dt/15, 4), round(mem, 1)]
    out[ds] = res
    print(ds, {k: f"{v[0]*1000:.1f}ms/round, {v[1]:.0f}MB" for k, v in res.items()})
json.dump(out, open(L.ROOT/'efficiency.json', 'w'), indent=2)
print('-> efficiency.json')
