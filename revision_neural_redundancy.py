#!/usr/bin/env python3
"""
Round-1 revision item (4): measure decision redundancy on the NEURAL backbone,
not only the Markov reference. We train a static model on T0 and a cumulatively
refit model before each block, and measure how often they agree on the top-1
recommendation for the same test transition. High agreement => updating the neural
model is mostly decision-redundant, the neural analogue of the ~70% Markov result.

Run with the CUDA env. -> prints + neural_redundancy.json
"""
import json
import numpy as np
import torch

import learned_continual as L


@torch.no_grad()
def top1(model, S):
    model.eval()
    out = []
    for seq, u, h, d, t in L.batches(S, np.arange(len(S["tgt"])), 1024):
        out.append(model(seq, u, h, d).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main():
    res = {}
    for name in ["nyc", "tky", "gowalla_ca", "brightkite"]:
        df = L.load(name)
        n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
        margs = (n_users, n_pois, 128, "gru")
        S0 = L.make_samples(df[df.block == "T0"], n_pois)
        torch.manual_seed(0); np.random.seed(0)
        static = L.NextPOI(*margs).to(L.DEVICE); L.fit(static, S0, epochs=8)
        agree, hit_static, hit_refit = [], [], []
        for k in range(1, 6):
            Sk = L.make_samples(df[df.block == f"T{k}"], n_pois)
            if Sk is None:
                continue
            cum = df[df.block.isin([f"T{j}" for j in range(k)])]
            torch.manual_seed(0); np.random.seed(0)
            refit = L.NextPOI(*margs).to(L.DEVICE); L.fit(refit, L.make_samples(cum, n_pois), epochs=8)
            a_s, a_r = top1(static, Sk), top1(refit, Sk)
            agree.append(float((a_s == a_r).mean()))
            t = Sk["tgt"].numpy()
            hit_static.append(float((a_s == t).mean())); hit_refit.append(float((a_r == t).mean()))
        res[name] = {
            "top1_agreement_static_vs_refit": round(float(np.mean(agree)), 4),
            "update_changes_top1_frac": round(1 - float(np.mean(agree)), 4),
            "top1_acc_static": round(float(np.mean(hit_static)), 4),
            "top1_acc_refit": round(float(np.mean(hit_refit)), 4),
        }
        print(f"[{name}] neural top-1 agreement static vs refit = {res[name]['top1_agreement_static_vs_refit']*100:.1f}% "
              f"(update changes top-1 only {res[name]['update_changes_top1_frac']*100:.1f}% of the time)")
    json.dump(res, open(L.ROOT / "neural_redundancy.json", "w"), indent=2)
    print("-> neural_redundancy.json")


if __name__ == "__main__":
    main()
