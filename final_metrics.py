#!/usr/bin/env python3
"""Final-state ranking quality at multiple cutoffs (Acc@1/5/10/20 + MRR), GRU.
Leakage-free: the per-user memory is built from blocks T1-T4 and the fine-tuned
baseline is trained on T1-T4; both are evaluated on the held-out final block T5,
whose targets are absent from the memory. Complements the streaming main table by
reporting the MRR defined in Eq. (1) and the full cutoff curve.
"""
import json
from pathlib import Path

import numpy as np
import torch

import learned_continual as L

DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
KS = [1, 5, 10, 20]
TR = ["T1", "T2", "T3", "T4"]


@torch.no_grad()
def metrics(score_fn, S):
    acc = {k: 0 for k in KS}; rr = 0.0; N = 0
    for seq, u, h, d, t in L.batches(S, np.arange(len(S["tgt"])), 1024):
        sc = score_fn(seq, u, h, d)
        tgt = sc.gather(1, t.unsqueeze(1))
        rank = (sc > tgt).sum(1) + 1
        for k in KS:
            acc[k] += int((rank <= k).sum())
        rr += float((1.0 / rank).sum()); N += len(rank)
    return {**{f"a{k}": round(acc[k] / N, 4) for k in KS}, "mrr": round(rr / N, 4)}


def main():
    out = {}
    for ds in DS:
        df = L.load(ds); nu = int(df.user_idx.max() + 1); npo = int(df.poi_idx.max() + 1)
        S0 = L.make_samples(df[df.block == "T0"], npo)
        Str = L.make_samples(df[df.block.isin(TR)], npo)
        Ste = L.make_samples(df[df.block == "T5"], npo)
        torch.manual_seed(0); np.random.seed(0)
        base = L.NextPOI(nu, npo, 128, "gru").to(L.DEVICE); L.fit(base, S0, epochs=8); base.eval()
        M = torch.zeros(nu, npo, device=L.DEVICE)                 # memory from T1-T4 only
        u = Str["user"].to(L.DEVICE); t = Str["tgt"].to(L.DEVICE)
        M.index_put_((u, t), torch.ones(len(t), device=L.DEVICE), accumulate=True)
        Mn = M / (M.sum(1, keepdim=True) + 1e-9); a = 0.5
        vae = L.InterestVAE(npo).to(L.DEVICE); L.fit_interest_vae(vae, S0, npo); vae.eval()
        ft = L.NextPOI(nu, npo, 128, "gru").to(L.DEVICE)          # FT baseline on T1-T4
        ft.load_state_dict(base.state_dict()); L.fit(ft, Str, epochs=3); ft.eval()
        res = {
            "static": metrics(lambda s, u, h, d: base(s, u, h, d), Ste),
            "always (FT)": metrics(lambda s, u, h, d: ft(s, u, h, d), Ste),
            "GIRAM (memory)": metrics(
                lambda s, u, h, d: a * base(s, u, h, d).softmax(1) + (1 - a) * Mn[u], Ste),
            "full GIRAM (VAE)": metrics(
                lambda s, u, h, d: a * base(s, u, h, d).softmax(1)
                + (1 - a) * (0.5 * Mn[u] + 0.5 * vae.retrieve(L._bag(s, npo))), Ste),
        }
        out[ds] = res
        print(ds, {m: d for m, d in res.items()})
    json.dump(out, open(L.ROOT / "final_metrics.json", "w"), indent=2)
    print("-> final_metrics.json")


if __name__ == "__main__":
    main()
