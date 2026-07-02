"""Rigorous clean-injection test (referee round 5). Geographic holdout: hold out a
contiguous LONGITUDE BAND of POIs from the base (a 'new neighborhood'), avoiding the
co-occurrence leakage of random-POI sampling. 5 seeds, mean+-std, paired t-test for
memory vs static -- SAME rigor as every other claim. Decomposes accuracy by target type
(NEW = held-out POI vs OLD = known POI) to test whether memory's edge captures the
injected headroom or is just regime-independent old-POI revisit."""
import json
import numpy as np, torch
import learned_continual as L
try:
    from scipy import stats
    def pval(d): return float(stats.ttest_1samp(d, 0).pvalue)
except Exception:
    def pval(d):
        d = np.asarray(d); n = len(d); s = d.std(ddof=1)
        if s == 0: return 0.0
        t = d.mean() / (s / np.sqrt(n)); return float(2 * (1 - 0.5 * (1 + np.math.erf(abs(t) / np.sqrt(2)))))

DS = ['nyc', 'tky', 'gowalla_ca', 'brightkite']; SEEDS = [0, 1, 2, 3, 4]; FRAC = 0.20

def geo_split(df, frac, seed):
    plon = df.groupby('poi_idx').lon.median()
    rng = np.random.default_rng(seed); lo = rng.uniform(0, 1 - frac)
    a, b = plon.quantile(lo), plon.quantile(lo + frac)
    inj = set(plon[(plon >= a) & (plon < b)].index)
    return df[(df.block == 'T0') & (~df.poi_idx.isin(inj))], df[df.block != 'T0'], inj

def setup(df, seed):
    base, stream, inj = geo_split(df, FRAC, seed)
    npo = int(df.poi_idx.max() + 1); nu = int(df.user_idx.max() + 1)
    S0 = L.make_samples(base, npo); St = L.make_samples(stream, npo)
    margs = (nu, npo, 128, 'gru')
    torch.manual_seed(seed); np.random.seed(seed)
    b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
    bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
    rounds = np.array_split(np.arange(len(St['tgt'])), 15)
    npr = float(np.mean(np.isin(St['tgt'].numpy(), list(inj))))
    return bs, St, rounds, margs, inj, npr

@torch.no_grad()
def topk_static(model, Sr, k=10):
    return np.concatenate([model(s, u, h, d).topk(k, 1).indices.cpu().numpy()
                           for s, u, h, d, t in L.batches(Sr, np.arange(len(Sr['tgt'])), 1024)])

@torch.no_grad()
def topk_mem(model, M, Sr, alpha, k=10):
    out = []
    for s, u, h, d, t in L.batches(Sr, np.arange(len(Sr['tgt'])), 1024):
        pm = model(s, u, h, d).softmax(1); pmem = M[u] / (M[u].sum(1, keepdim=True) + 1e-9)
        out.append((alpha * pm + (1 - alpha) * pmem).topk(k, 1).indices.cpu().numpy())
    return np.concatenate(out)

def split_hits(top, t, inj):
    isnew = np.isin(t, list(inj)); hit = np.array([t[i] in top[i] for i in range(len(t))])
    return hit[isnew].sum(), isnew.sum(), hit[~isnew].sum(), (~isnew).sum()

def run_static(bs, St, rounds, margs, inj):
    m = L.NextPOI(*margs).to(L.DEVICE); m.load_state_dict(bs); m.eval()
    HN = NN = HO = NO = 0
    for idx in rounds:
        Sr = L.subset(St, idx); hn, nn, ho, no = split_hits(topk_static(m, Sr), Sr['tgt'].numpy(), inj)
        HN += hn; NN += nn; HO += ho; NO += no
    return (HN + HO) / (NN + NO), HN / max(NN, 1), HO / max(NO, 1)

def run_mem(bs, St, rounds, margs, inj, alpha=0.5, decay=0.9):
    nu, npo = margs[0], margs[1]; m = L.NextPOI(*margs).to(L.DEVICE); m.load_state_dict(bs); m.eval()
    M = torch.zeros(nu, npo, device=L.DEVICE); HN = NN = HO = NO = 0
    for idx in rounds:
        Sr = L.subset(St, idx); hn, nn, ho, no = split_hits(topk_mem(m, M, Sr, alpha), Sr['tgt'].numpy(), inj)
        HN += hn; NN += nn; HO += ho; NO += no
        M.mul_(decay); u = Sr['user'].to(L.DEVICE); tt = Sr['tgt'].to(L.DEVICE)
        M.index_put_((u, tt), torch.ones(len(tt), device=L.DEVICE), accumulate=True)
    return (HN + HO) / (NN + NO), HN / max(NN, 1), HO / max(NO, 1)

out = {}
for ds in DS:
    df = L.load(ds); R = {k: [] for k in ['st', 'mem', 'sN', 'mN', 'sO', 'mO', 'npr']}
    for seed in SEEDS:
        bs, St, rounds, margs, inj, npr = setup(df, seed)
        so, sN, sO = run_static(bs, St, rounds, margs, inj)
        mo, mN, mO = run_mem(bs, St, rounds, margs, inj)
        for k, v in [('st', so), ('mem', mo), ('sN', sN), ('mN', mN), ('sO', sO), ('mO', mO), ('npr', npr)]:
            R[k].append(v)
    d = np.array(R['mem']) - np.array(R['st'])
    out[ds] = {'newPOI_rate': round(float(np.mean(R['npr'])), 3),
               'static': [round(float(np.mean(R['st'])), 4), round(float(np.std(R['st'])), 4)],
               'memory': [round(float(np.mean(R['mem'])), 4), round(float(np.std(R['mem'])), 4)],
               'mem_minus_static': [round(float(np.mean(d)), 4), round(float(np.std(d)), 4), round(pval(d), 4)],
               'NEWtarget_static': round(float(np.mean(R['sN'])), 4), 'NEWtarget_memory': round(float(np.mean(R['mN'])), 4),
               'OLDtarget_static': round(float(np.mean(R['sO'])), 4), 'OLDtarget_memory': round(float(np.mean(R['mO'])), 4)}
    print(ds, json.dumps(out[ds]))
json.dump(out, open(L.ROOT / 'inject_rigorous.json', 'w'), indent=2)
print('-> inject_rigorous.json')
