"""Symmetric completion: run static, fixed-LR FT (3e-5), and the controller in the SAME
geographic-injection regime, 5 seeds, with the new/old target split and paired tests vs
static -- so the claim 'fine-tuning captures none of the headroom, memory captures the
small revisit slice' is stated at uniform rigor."""
import json
import numpy as np, torch
import learned_continual as L
from inject_rigorous import geo_split, FRAC, topk_static, split_hits, pval

DS = ['nyc', 'tky', 'gowalla_ca', 'brightkite']; SEEDS = [0, 1, 2, 3, 4]

def setup(df, seed):
    base, stream, inj = geo_split(df, FRAC, seed)
    npo = int(df.poi_idx.max() + 1); nu = int(df.user_idx.max() + 1)
    S0 = L.make_samples(base, npo); St = L.make_samples(stream, npo)
    margs = (nu, npo, 128, 'gru')
    torch.manual_seed(seed); np.random.seed(seed)
    b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
    bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
    rng = np.random.default_rng(seed)
    fS = L.subset(S0, rng.choice(len(S0['tgt']), min(2000, len(S0['tgt'])), replace=False))
    rounds = np.array_split(np.arange(len(St['tgt'])), 15)
    return bs, St, rounds, fS, margs, inj

def run_ft_split(bs, St, rounds, fS, margs, inj, mode):
    model = L.NextPOI(*margs).to(L.DEVICE); model.load_state_dict(bs)
    HN = NN = HO = NO = 0; lr = 1e-3 if mode == 'ctrl' else 3e-5
    f0, _ = L.evaluate(model, fS)
    for idx in rounds:
        Sr = L.subset(St, idx); model.eval()
        hn, nn, ho, no = split_hits(topk_static(model, Sr), Sr['tgt'].numpy(), inj)
        HN += hn; NN += nn; HO += ho; NO += no
        if mode == 'ctrl':
            snap = {k: v.detach().clone() for k, v in model.state_dict().items()}; pre, _ = L.evaluate(model, fS)
            L.fit(model, Sr, epochs=10, max_steps=30, lr=lr); post, _ = L.evaluate(model, fS)
            if pre - post > 0.005:
                model.load_state_dict(snap); lr = max(lr * 0.5, 1e-6)
        else:
            L.fit(model, Sr, epochs=10, max_steps=30, lr=3e-5)
    fN, _ = L.evaluate(model, fS)
    return (HN + HO) / (NN + NO), HN / max(NN, 1), HO / max(NO, 1), float(f0 - fN)

def run_static(bs, St, rounds, margs, inj):
    m = L.NextPOI(*margs).to(L.DEVICE); m.load_state_dict(bs); m.eval()
    HN = NN = HO = NO = 0
    for idx in rounds:
        Sr = L.subset(St, idx); hn, nn, ho, no = split_hits(topk_static(m, Sr), Sr['tgt'].numpy(), inj)
        HN += hn; NN += nn; HO += ho; NO += no
    return (HN + HO) / (NN + NO), HN / max(NN, 1), HO / max(NO, 1)

out = {}
for ds in DS:
    df = L.load(ds); R = {k: [] for k in ['st', 'fx', 'ct', 'fxN', 'ctN', 'stN', 'fxF', 'ctF']}
    for seed in SEEDS:
        bs, St, rounds, fS, margs, inj = setup(df, seed)
        so, sN, sO = run_static(bs, St, rounds, margs, inj)
        fo, fN, fO, ff = run_ft_split(bs, St, rounds, fS, margs, inj, 'fixed')
        co, cN, cO, cf = run_ft_split(bs, St, rounds, fS, margs, inj, 'ctrl')
        R['st'].append(so); R['fx'].append(fo); R['ct'].append(co)
        R['stN'].append(sN); R['fxN'].append(fN); R['ctN'].append(cN); R['fxF'].append(ff); R['ctF'].append(cf)
    df_fx = np.array(R['fx']) - np.array(R['st']); df_ct = np.array(R['ct']) - np.array(R['st'])
    out[ds] = {'static': round(float(np.mean(R['st'])), 4),
               'fixedFT': [round(float(np.mean(R['fx'])), 4), round(float(np.mean(df_fx)), 4), round(pval(df_fx), 4), round(float(np.mean(R['fxF'])), 3)],
               'controller': [round(float(np.mean(R['ct'])), 4), round(float(np.mean(df_ct)), 4), round(pval(df_ct), 4), round(float(np.mean(R['ctF'])), 3)],
               'NEWtarget': {'static': round(float(np.mean(R['stN'])), 4), 'fixedFT': round(float(np.mean(R['fxN'])), 4), 'controller': round(float(np.mean(R['ctN'])), 4)}}
    print(ds, json.dumps(out[ds]))
json.dump(out, open(L.ROOT / 'inject_rigorous_ft.json', 'w'), indent=2)
print('-> inject_rigorous_ft.json')
