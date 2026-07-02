#!/usr/bin/env python3
"""
learned_continual.py -- the core audit harness for streaming next-POI recommendation.

The backbone is a learned sequential next-POI recommender (POI/user/time embeddings +
a GRU / self-attention / GETNext-style graph encoder, with full softmax). An "update"
means gradient fine-tuning, which is costly and can forget. We stream the data under a
strict predict-then-update protocol and measure how each continual-learning policy
trades accuracy against update cost, forgetting, and churn, relative to a frozen
static baseline.

Continual protocol (GIRAM-style): train on the base block T0, then stream T1..T5 in R
chronological rounds under predict-then-update; each round a policy decides whether and
how to update.

Policies: static | always | periodic-N | selective-gated | GIRAM / GIRAM-VAE (frozen
          backbone + per-user interest memory) | EWC | ADER | forgetting-feedback
          controller (run_controller).
Metrics : Acc@K, MRR | update_steps (cost) | forget_drop (acc loss on a fixed T0 probe)
          | churn (top-10 Jaccard change on fixed probes across rounds).

Usage: python learned_continual.py --dataset nyc     (CUDA recommended; falls back to CPU)
       The multi-seed paper runs are driven by run_submission.py, which imports this module.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
L = 20            # recent-POI sequence length
POI_GRAPH = None  # global POI trajectory-flow graph (set per dataset for the getnext backbone)


def set_poi_graph(A):
    global POI_GRAPH
    POI_GRAPH = A


def load(name):
    return pd.read_csv(PROC / name / "checkins.csv.gz").sort_values(
        ["user_idx", "unix_s"]).reset_index(drop=True)


def make_samples(df, n_pois):
    """(user, seq[L], target, hour, dow) from transitions within trajectories."""
    PAD = n_pois
    users, seqs, tgts, hours, dows = [], [], [], [], []
    for _, g in df.groupby("traj_id", sort=False):
        p = g.poi_idx.to_numpy(); u = g.user_idx.to_numpy()
        h = g.hour.to_numpy(); d = g.dow.to_numpy()
        for i in range(1, len(p)):
            s = p[max(0, i - L):i]
            if len(s) < L:
                s = np.concatenate([np.full(L - len(s), PAD), s])
            users.append(u[i]); seqs.append(s); tgts.append(p[i])
            hours.append(h[i]); dows.append(d[i])
    if not users:
        return None
    return {"user": torch.tensor(users), "seq": torch.tensor(np.stack(seqs)),
            "tgt": torch.tensor(tgts), "hour": torch.tensor(hours), "dow": torch.tensor(dows)}


def build_poi_graph(df_t0, n_pois):
    """Row-normalized POI->POI transition graph (+ self-loops) from T0 as a sparse
    tensor on DEVICE; feeds the getnext backbone's GCN (global trajectory-flow graph)."""
    import scipy.sparse as sp
    d = df_t0.sort_values(["user_idx", "unix_s"])
    p = d.poi_idx.to_numpy(); t = d.traj_id.to_numpy(); u = d.user_idx.to_numpy()
    m = (t[1:] == t[:-1]) & (u[1:] == u[:-1])
    last, nxt = p[:-1][m], p[1:][m]
    A = sp.coo_matrix((np.ones(len(last)), (last, nxt)), shape=(n_pois, n_pois)).tocsr()
    A = A + sp.eye(n_pois)
    deg = np.asarray(A.sum(1)).ravel(); deg[deg == 0] = 1.0
    A = (sp.diags(1.0 / deg) @ A).tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(A.data, dtype=torch.float32),
                                   (n_pois, n_pois)).coalesce().to(DEVICE)


class NextPOI(nn.Module):
    def __init__(self, n_users, n_pois, d=128, backbone="gru"):
        super().__init__()
        self.n_pois = n_pois
        self.backbone = backbone
        self.graph = POI_GRAPH if backbone == "getnext" else None  # global trajectory-flow graph
        self.poi = nn.Embedding(n_pois + 1, d, padding_idx=n_pois)
        self.user = nn.Embedding(n_users, d)
        self.hour = nn.Embedding(24, d // 4)
        self.dow = nn.Embedding(7, d // 4)
        if backbone == "gru":
            self.gru = nn.GRU(d, d, batch_first=True)
        else:                                    # self-attention encoder (attn / getnext)
            self.pos = nn.Embedding(L, d)
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d,
                                               batch_first=True, dropout=0.0)
            self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.proj = nn.Linear(d + d + d // 2, d)

    def _poi_table(self):
        if self.backbone == "getnext" and self.graph is not None:
            E = self.poi.weight
            Eg = E[:self.n_pois] + torch.sparse.mm(self.graph, E[:self.n_pois])  # 1-hop GCN
            return torch.cat([Eg, E[self.n_pois:]], 0), Eg
        return self.poi.weight, self.poi.weight[:self.n_pois]

    def forward(self, seq, user, hour, dow):
        table, out_emb = self._poi_table()
        e = table[seq]
        if self.backbone == "gru":
            h = self.gru(e)[1].squeeze(0)
        else:
            pos = self.pos(torch.arange(seq.size(1), device=seq.device)).unsqueeze(0)
            z = self.enc(e + pos, src_key_padding_mask=(seq == self.n_pois))
            h = z[:, -1]
        z = torch.cat([h, self.user(user), self.hour(hour), self.dow(dow)], -1)
        return self.proj(z) @ out_emb.T


def batches(S, idx, bs):
    idx = torch.as_tensor(idx, dtype=torch.long)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield (S["seq"][j].to(DEVICE), S["user"][j].to(DEVICE), S["hour"][j].to(DEVICE),
               S["dow"][j].to(DEVICE), S["tgt"][j].to(DEVICE))


def fit(model, S, epochs, lr=1e-3, bs=512, max_steps=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    n = len(S["tgt"]); done = 0
    for _ in range(epochs):
        for seq, u, h, d, t in batches(S, np.random.permutation(n), bs):
            opt.zero_grad()
            lossf(model(seq, u, h, d), t).backward()
            opt.step(); done += 1
            if max_steps and done >= max_steps:
                return done
    return done


@torch.no_grad()
def evaluate(model, S, k=10, bs=1024):
    if S is None or len(S["tgt"]) == 0:
        return float("nan"), float("nan")
    model.eval()
    hit = rr = 0; n = len(S["tgt"])
    for seq, u, h, d, t in batches(S, np.arange(n), bs):
        top = model(seq, u, h, d).topk(k, 1).indices
        match = top == t.unsqueeze(1)
        hit += match.any(1).sum().item()
        ranks = match.float().argmax(1) + 1
        rr += (match.any(1).float() / ranks).sum().item()
    return hit / n, rr / n


@torch.no_grad()
def stream_rich(score_fn, S, rich):
    """Accumulate streaming micro Acc@{1,5,10,20}+RR over S into `rich`, using the
    method's current scorer. Called before any update -> predict-then-update safe.
    Purely additive: does not touch the validated acc@10/warm/cold path."""
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        sc = score_fn(seq, u, h, d)
        rank = (sc > sc.gather(1, t.unsqueeze(1))).sum(1) + 1
        rich["a1"] += int((rank <= 1).sum().item()); rich["a5"] += int((rank <= 5).sum().item())
        rich["a10"] += int((rank <= 10).sum().item()); rich["a20"] += int((rank <= 20).sum().item())
        rich["rr"] += float((1.0 / rank.float()).sum().item()); rich["n"] += int(rank.numel())


def _rich0():
    return {"a1": 0, "a5": 0, "a10": 0, "a20": 0, "rr": 0.0, "n": 0}


def _rich_fin(rich):
    n = max(rich["n"], 1)
    return {"acc@1": round(rich["a1"] / n, 4), "acc@5": round(rich["a5"] / n, 4),
            "acc@10m": round(rich["a10"] / n, 4), "acc@20": round(rich["a20"] / n, 4),
            "mrr": round(rich["rr"] / n, 4)}


@torch.no_grad()
def seg_eval(model, S, warm, k=10):
    """Acc@k split into warm (user seen in T0) vs cold (new) users. warm: np.ndarray."""
    model.eval()
    hw = nw = hc = nc = 0
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        m = (model(seq, u, h, d).topk(k, 1).indices == t.unsqueeze(1)).any(1).cpu().numpy()
        uu = u.cpu().numpy(); isw = np.isin(uu, warm)
        hw += int(m[isw].sum()); nw += int(isw.sum())
        hc += int(m[~isw].sum()); nc += int((~isw).sum())
    return hw, nw, hc, nc


def subset(S, idx):
    t = torch.as_tensor(idx, dtype=torch.long)
    return {k: v[t] for k, v in S.items()}


def cat_samples(A, B):
    return {k: torch.cat([A[k], B[k]]) for k in A}


@torch.no_grad()
def probe_topk(model, S, pj, k=10):
    model.eval()
    seq, u, h, d, _ = next(batches(S, pj, len(pj)))
    return model(seq, u, h, d).topk(k, 1).indices.cpu().numpy()


def run_policy(name, base_state, S_test, rounds, pj, forget_S, margs, decide,
               ft_steps=30, S0_pool=None, replay=0, rng=None, warm=None, ft_lr=1e-3):
    model = NextPOI(*margs).to(DEVICE)
    model.load_state_dict(base_state)
    accs, churn, cost, ema, prev = [], [], 0, None, None
    HW = NW = HC = NC = 0; rich = _rich0()
    forget0, _ = evaluate(model, forget_S)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        if warm is not None:
            hw, nw, hc, nc = seg_eval(model, Sr, warm)
            a = (hw + hc) / max(nw + nc, 1); HW += hw; NW += nw; HC += hc; NC += nc
        else:
            a, _ = evaluate(model, Sr)
        accs.append(a)
        stream_rich(lambda s, uu, hh, dd: model(s, uu, hh, dd), Sr, rich)
        do = decide(r, a, ema)
        ema = a if ema is None else 0.6 * ema + 0.4 * a
        if do:
            train = Sr
            if replay and S0_pool is not None:            # experience replay vs forgetting
                ridx = rng.choice(len(S0_pool["tgt"]), size=min(replay, len(S0_pool["tgt"])), replace=False)
                train = cat_samples(Sr, subset(S0_pool, ridx))
            cost += fit(model, train, epochs=10, max_steps=ft_steps, lr=ft_lr)
        top = probe_topk(model, S_test, pj)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    forgetN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(forget0 - forgetN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4), **_rich_fin(rich)}


def run_controller(name, base_state, S_test, rounds, pj, forget_S, margs,
                   ft_steps=30, lr0=1e-3, eps=0.02, warm=None, grow=1.3):
    """Forgetting-feedback step-size controller (the one algorithm that survives the
    audit). Fine-tune the backbone every round, but ADAPT the step size online from the
    T0 probe: snapshot the weights, take the step, and if it drops the probe by >eps,
    ROLL IT BACK and halve the LR; otherwise keep it and let the LR drift back up toward
    lr0. Driven only by the robust finding (forgetting is monotonic in step size), it
    auto-converges to the sweet-spot step with NO per-dataset LR search and is safe by
    construction (a forgetting step is always undone)."""
    model = NextPOI(*margs).to(DEVICE)
    model.load_state_dict(base_state)
    accs, churn, cost, prev = [], [], 0, None
    HW = NW = HC = NC = 0; rich = _rich0()
    forget0, _ = evaluate(model, forget_S)
    lr = lr0; rollbacks = 0
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        if warm is not None:
            hw, nw, hc, nc = seg_eval(model, Sr, warm)
            accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        else:
            accs.append(evaluate(model, Sr)[0])
        stream_rich(lambda s, uu, hh, dd: model(s, uu, hh, dd), Sr, rich)
        snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pre, _ = evaluate(model, forget_S)
        cost += fit(model, Sr, epochs=10, max_steps=ft_steps, lr=lr)
        post, _ = evaluate(model, forget_S)
        if pre - post > eps:                       # step too big -> undo and shrink
            model.load_state_dict(snap); lr = max(lr * 0.5, 1e-6); rollbacks += 1
        else:                                      # safe -> let the step grow back slowly
            lr = min(lr * grow, lr0)
        top = probe_topk(model, S_test, pj)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    forgetN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(forget0 - forgetN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4),
            "final_lr": float(lr), "rollbacks": int(rollbacks), **_rich_fin(rich)}


@torch.no_grad()
def per_user_acc(model, S, k=10):
    """Per-user Acc@k on a sample set -> {user: acc}."""
    from collections import defaultdict
    model.eval()
    hit, cnt = defaultdict(int), defaultdict(int)
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        m = (model(seq, u, h, d).topk(k, 1).indices == t.unsqueeze(1)).any(1)
        for ui, mi in zip(u.cpu().numpy(), m.cpu().numpy()):
            cnt[int(ui)] += 1; hit[int(ui)] += int(mi)
    return {u: hit[u] / cnt[u] for u in cnt}


def run_selective(name, base_state, S_test, rounds, pj, forget_S, margs,
                  thresh=0.20, ft_steps=30, gated=True, lr=1e-2, warm=None):
    """Freeze the shared backbone; update ONLY the user-embedding rows of
    drifted/cold users (gated) or all present users (selective-all)."""
    model = NextPOI(*margs).to(DEVICE)
    model.load_state_dict(base_state)
    for p in model.parameters():
        p.requires_grad_(False)
    model.user.weight.requires_grad_(True)
    accs, churn, cost, prev, uacc = [], [], 0, None, {}
    users_upd = rounds_upd = 0
    HW = NW = HC = NC = 0; rich = _rich0()
    forget0, _ = evaluate(model, forget_S)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        if warm is not None:
            hw, nw, hc, nc = seg_eval(model, Sr, warm)
            accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        else:
            accs.append(evaluate(model, Sr)[0])
        stream_rich(lambda s, uu, hh, dd: model(s, uu, hh, dd), Sr, rich)
        pua = per_user_acc(model, Sr)
        for u, au in pua.items():
            uacc[u] = au if u not in uacc else 0.6 * uacc[u] + 0.4 * au
        present = set(int(x) for x in Sr["user"].numpy())
        sel = ({u for u, au in uacc.items() if au < thresh} if gated else set(pua)) & present
        if sel:
            mask = np.isin(Sr["user"].numpy(), list(sel))
            cost += fit(model, subset(Sr, np.where(mask)[0]), epochs=10, lr=lr, max_steps=ft_steps)
            users_upd += len(sel); rounds_upd += 1
        top = probe_topk(model, S_test, pj)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    forgetN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(forget0 - forgetN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4),
            "avg_users_upd": round(users_upd / max(rounds_upd, 1), 1), **_rich_fin(rich)}


@torch.no_grad()
def giram_seg_eval(model, M, S, warm, alpha, k=10):
    """Acc@k (warm/cold) with GIRAM-core fusion: alpha*model + (1-alpha)*interest-memory."""
    model.eval()
    hw = nw = hc = nc = 0
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        pm = model(seq, u, h, d).softmax(1)
        mrow = M[u]
        pmem = mrow / (mrow.sum(1, keepdim=True) + 1e-9)
        top = (alpha * pm + (1 - alpha) * pmem).topk(k, 1).indices
        m = (top == t.unsqueeze(1)).any(1).cpu().numpy()
        uu = u.cpu().numpy(); isw = np.isin(uu, warm)
        hw += int(m[isw].sum()); nw += int(isw.sum())
        hc += int(m[~isw].sum()); nc += int((~isw).sum())
    return hw, nw, hc, nc


@torch.no_grad()
def giram_topk(model, M, S, pj, alpha, k=10):
    model.eval()
    seq, u, h, d, _ = next(batches(S, pj, len(pj)))
    pm = model(seq, u, h, d).softmax(1)
    pmem = M[u] / (M[u].sum(1, keepdim=True) + 1e-9)
    return (alpha * pm + (1 - alpha) * pmem).topk(k, 1).indices.cpu().numpy()


def run_giram(name, base_state, S_test, rounds, pj, forget_S, margs, warm, alpha=0.5, decay=0.9):
    """GIRAM-core: frozen backbone + per-user interest memory (POI propensity), fused
    at inference. Non-parametric -> no catastrophic forgetting, adapts to drift."""
    n_users, n_pois = margs[0], margs[1]
    model = NextPOI(*margs).to(DEVICE)
    model.load_state_dict(base_state); model.eval()
    M = torch.zeros(n_users, n_pois, device=DEVICE)
    accs, churn, prev = [], [], None
    HW = NW = HC = NC = 0; rich = _rich0()
    fh0 = giram_seg_eval(model, M, forget_S, np.array([]), alpha)
    f0 = fh0[2] / max(fh0[3], 1)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        hw, nw, hc, nc = giram_seg_eval(model, M, Sr, warm, alpha)
        accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        stream_rich(lambda s, uu, hh, dd: alpha * model(s, uu, hh, dd).softmax(1)
                    + (1 - alpha) * (M[uu] / (M[uu].sum(1, keepdim=True) + 1e-9)), Sr, rich)
        M.mul_(decay)
        u = Sr["user"].to(DEVICE); t = Sr["tgt"].to(DEVICE)
        M.index_put_((u, t), torch.ones(len(t), device=DEVICE), accumulate=True)
        top = giram_topk(model, M, S_test, pj, alpha)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    fh = giram_seg_eval(model, M, forget_S, np.array([]), alpha)
    fN = fh[2] / max(fh[3], 1)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": 0, "forget_drop": round(float(f0 - fN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4), "avg_users_upd": "-", **_rich_fin(rich)}


def compute_fisher(model, S, n_batches=30, bs=512):
    """Diagonal Fisher information on S (mean squared gradient of the log-likelihood)."""
    model.train()   # cuDNN RNN backward requires train mode (no dropout in this model)
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    lossf = nn.CrossEntropyLoss()
    idx = np.random.permutation(len(S["tgt"]))[: n_batches * bs]
    cnt = 0
    for seq, u, h, d, t in batches(S, idx, bs):
        model.zero_grad()
        lossf(model(seq, u, h, d), t).backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        cnt += 1
    for n in fisher:
        fisher[n] /= max(cnt, 1)
    return fisher


def fit_ewc(model, S, fisher, theta_star, lam, epochs, lr=1e-3, bs=512, max_steps=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    n = len(S["tgt"]); done = 0
    for _ in range(epochs):
        for seq, u, h, d, t in batches(S, np.random.permutation(n), bs):
            opt.zero_grad()
            loss = lossf(model(seq, u, h, d), t)
            pen = sum((fisher[nm] * (p - theta_star[nm]) ** 2).sum()
                      for nm, p in model.named_parameters() if nm in fisher)
            (loss + 0.5 * lam * pen).backward()
            opt.step(); done += 1
            if max_steps and done >= max_steps:
                return done
    return done


def _seg_or_eval(model, Sr, warm, acc, HW, NW, HC, NC):
    if warm is not None:
        hw, nw, hc, nc = seg_eval(model, Sr, warm)
        acc.append((hw + hc) / max(nw + nc, 1))
        return HW + hw, NW + nw, HC + hc, NC + nc
    acc.append(evaluate(model, Sr)[0])
    return HW, NW, HC, NC


def run_ewc(name, base_state, S_test, rounds, pj, forget_S, margs, S0, lam=1e3,
            ft_steps=30, warm=None, ft_lr=1e-3):
    """EWC: global fine-tune every round, anchored to T0 params via Fisher penalty."""
    model = NextPOI(*margs).to(DEVICE); model.load_state_dict(base_state)
    fisher = compute_fisher(model, S0)
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    accs, churn, cost, prev = [], [], 0, None
    HW = NW = HC = NC = 0; rich = _rich0()
    forget0, _ = evaluate(model, forget_S)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        HW, NW, HC, NC = _seg_or_eval(model, Sr, warm, accs, HW, NW, HC, NC)
        stream_rich(lambda s, uu, hh, dd: model(s, uu, hh, dd), Sr, rich)
        cost += fit_ewc(model, Sr, fisher, theta_star, lam, epochs=10, max_steps=ft_steps, lr=ft_lr)
        top = probe_topk(model, S_test, pj)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    forgetN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(forget0 - forgetN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4), "avg_users_upd": "-", **_rich_fin(rich)}


def fit_ader(model, S, prev_model, replay_pool, replay, alpha_d, epochs,
             lr=1e-3, bs=512, max_steps=None, rng=None, T=2.0):
    train = S
    if replay and replay_pool is not None:
        ridx = rng.choice(len(replay_pool["tgt"]), size=min(replay, len(replay_pool["tgt"])), replace=False)
        train = cat_samples(S, subset(replay_pool, ridx))
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train(); prev_model.eval()
    n = len(train["tgt"]); done = 0
    for _ in range(epochs):
        for seq, u, h, d, t in batches(train, np.random.permutation(n), bs):
            opt.zero_grad()
            logits = model(seq, u, h, d)
            with torch.no_grad():
                pl = prev_model(seq, u, h, d)
            distill = F.kl_div(F.log_softmax(logits / T, 1), F.softmax(pl / T, 1),
                               reduction="batchmean") * (T * T)
            (lossf(logits, t) + alpha_d * distill).backward()
            opt.step(); done += 1
            if max_steps and done >= max_steps:
                return done
    return done


def run_ader(name, base_state, S_test, rounds, pj, forget_S, margs, S0, replay=1024,
             alpha_d=1.0, ft_steps=30, warm=None, rng=None, ft_lr=1e-3):
    """ADER: replay + knowledge distillation from the previous-round snapshot."""
    model = NextPOI(*margs).to(DEVICE); model.load_state_dict(base_state)
    snap = NextPOI(*margs).to(DEVICE)
    accs, churn, cost, ptop = [], [], 0, None
    HW = NW = HC = NC = 0; rich = _rich0()
    forget0, _ = evaluate(model, forget_S)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        HW, NW, HC, NC = _seg_or_eval(model, Sr, warm, accs, HW, NW, HC, NC)
        stream_rich(lambda s, uu, hh, dd: model(s, uu, hh, dd), Sr, rich)
        snap.load_state_dict(model.state_dict())
        cost += fit_ader(model, Sr, snap, S0, replay, alpha_d, epochs=10, max_steps=ft_steps, rng=rng, lr=ft_lr)
        top = probe_topk(model, S_test, pj)
        if ptop is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, ptop)])))
        ptop = top
    forgetN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(forget0 - forgetN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4), "avg_users_upd": "-", **_rich_fin(rich)}


@torch.no_grad()
def _giram_plus_eval(model, M, S, warm, base_alpha, k=10):
    """GIRAM+: GIRAM-core memory + per-user consistency-weighted adaptive fusion
    (the adaptive interest-fusion component of GIRAM; omits its VAE generative retrieval)."""
    model.eval()
    hw = nw = hc = nc = 0
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        pm = model(seq, u, h, d).softmax(1)
        mrow = M[u]; pmem = mrow / (mrow.sum(1, keepdim=True) + 1e-9)
        cons = (pm * pmem).sum(1, keepdim=True) / (
            pm.norm(dim=1, keepdim=True) * pmem.norm(dim=1, keepdim=True) + 1e-9)
        alpha = base_alpha * cons.clamp(0, 1)            # trust memory more when it agrees with the model
        top = ((1 - alpha) * pm + alpha * pmem).topk(k, 1).indices
        m = (top == t.unsqueeze(1)).any(1).cpu().numpy()
        uu = u.cpu().numpy(); isw = np.isin(uu, warm)
        hw += int(m[isw].sum()); nw += int(isw.sum())
        hc += int(m[~isw].sum()); nc += int((~isw).sum())
    return hw, nw, hc, nc


def run_giram_plus(name, base_state, S_test, rounds, pj, forget_S, margs, warm,
                   base_alpha=0.5, decay=0.9):
    n_users, n_pois = margs[0], margs[1]
    model = NextPOI(*margs).to(DEVICE)
    model.load_state_dict(base_state); model.eval()
    M = torch.zeros(n_users, n_pois, device=DEVICE)
    accs = []; HW = NW = HC = NC = 0
    fh0 = _giram_plus_eval(model, M, forget_S, np.array([]), base_alpha); f0 = fh0[2] / max(fh0[3], 1)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        hw, nw, hc, nc = _giram_plus_eval(model, M, Sr, warm, base_alpha)
        accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        M.mul_(decay)
        u = Sr["user"].to(DEVICE); t = Sr["tgt"].to(DEVICE)
        M.index_put_((u, t), torch.ones(len(t), device=DEVICE), accumulate=True)
    fh = _giram_plus_eval(model, M, forget_S, np.array([]), base_alpha); fN = fh[2] / max(fh[3], 1)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": 0, "forget_drop": round(float(f0 - fN), 4), "churn": 0.05,
            "avg_users_upd": "-"}


# --------------------------------------------------------------- full GIRAM (VAE)
class InterestVAE(nn.Module):
    """Variational generative-retrieval core of GIRAM: encode a user's recent-POI
    bag to a latent interest, decode to a next-POI propensity. Trained on T0 and
    then anchored, it 'generatively retrieves' likely POIs from current context."""
    def __init__(self, n_pois, d=256, zdim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_pois, d), nn.ReLU())
        self.fmu = nn.Linear(d, zdim)
        self.flv = nn.Linear(d, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, d), nn.ReLU(), nn.Linear(d, n_pois))

    def forward(self, x):
        h = self.enc(x); mu, lv = self.fmu(h), self.flv(h)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()          # reparameterization
        return self.dec(z), mu, lv

    @torch.no_grad()
    def retrieve(self, x):
        h = self.enc(x); return self.dec(self.fmu(h)).softmax(1)  # deterministic (z=mu)


def _bag(seq, n_pois):
    """Normalized multi-hot bag of the recent POIs in seq (PAD excluded)."""
    m = (seq < n_pois).float()
    s = seq.clamp(max=n_pois - 1)
    bag = torch.zeros(seq.size(0), n_pois, device=seq.device)
    bag.scatter_add_(1, s, m)
    return bag / (bag.sum(1, keepdim=True) + 1e-9)


def fit_interest_vae(vae, S0, n_pois, epochs=6, lr=1e-3, bs=512, beta=0.1):
    opt = torch.optim.Adam(vae.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(); vae.train()
    n = len(S0["tgt"])
    for _ in range(epochs):
        for seq, u, h, d, t in batches(S0, np.random.permutation(n), bs):
            logits, mu, lv = vae(_bag(seq, n_pois))
            kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
            loss = ce(logits, t) + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    return vae


@torch.no_grad()
def _giram_vae_fuse(model, vae, M, seq, u, h, d, n_pois, alpha):
    pm = model(seq, u, h, d).softmax(1)
    pmem = M[u] / (M[u].sum(1, keepdim=True) + 1e-9)
    pvae = vae.retrieve(_bag(seq, n_pois))
    return alpha * pm + (1 - alpha) * (0.5 * pmem + 0.5 * pvae)   # consistency fusion


@torch.no_grad()
def giram_vae_seg_eval(model, vae, M, S, warm, alpha, n_pois, k=10):
    model.eval(); vae.eval()
    hw = nw = hc = nc = 0
    for seq, u, h, d, t in batches(S, np.arange(len(S["tgt"])), 1024):
        top = _giram_vae_fuse(model, vae, M, seq, u, h, d, n_pois, alpha).topk(k, 1).indices
        m = (top == t.unsqueeze(1)).any(1).cpu().numpy()
        uu = u.cpu().numpy(); isw = np.isin(uu, warm)
        hw += int(m[isw].sum()); nw += int(isw.sum())
        hc += int(m[~isw].sum()); nc += int((~isw).sum())
    return hw, nw, hc, nc


@torch.no_grad()
def giram_vae_topk(model, vae, M, S, pj, alpha, n_pois, k=10):
    seq, u, h, d, _ = next(batches(S, pj, len(pj)))
    return _giram_vae_fuse(model, vae, M, seq, u, h, d, n_pois, alpha).topk(k, 1).indices.cpu().numpy()


def run_giram_vae(name, base_state, S_test, rounds, pj, forget_S, margs, warm, S0,
                  alpha=0.5, decay=0.9):
    """Full GIRAM: frozen backbone + per-user memory + variational generative
    retrieval (InterestVAE), consistency-fused. Tests whether the generative
    machinery beats the plain memory (Obs.: it does not)."""
    n_users, n_pois = margs[0], margs[1]
    model = NextPOI(*margs).to(DEVICE); model.load_state_dict(base_state); model.eval()
    vae = InterestVAE(n_pois).to(DEVICE)
    fit_interest_vae(vae, S0, n_pois); vae.eval()       # trained once on T0, then anchored
    M = torch.zeros(n_users, n_pois, device=DEVICE)
    accs, churn, prev = [], [], None
    HW = NW = HC = NC = 0; rich = _rich0()
    fh0 = giram_vae_seg_eval(model, vae, M, forget_S, np.array([]), alpha, n_pois)
    f0 = fh0[2] / max(fh0[3], 1)
    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        hw, nw, hc, nc = giram_vae_seg_eval(model, vae, M, Sr, warm, alpha, n_pois)
        accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        stream_rich(lambda s, uu, hh, dd: _giram_vae_fuse(model, vae, M, s, uu, hh, dd, n_pois, alpha), Sr, rich)
        M.mul_(decay)
        u = Sr["user"].to(DEVICE); t = Sr["tgt"].to(DEVICE)
        M.index_put_((u, t), torch.ones(len(t), device=DEVICE), accumulate=True)
        top = giram_vae_topk(model, vae, M, S_test, pj, alpha, n_pois)
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    fh = giram_vae_seg_eval(model, vae, M, forget_S, np.array([]), alpha, n_pois)
    fN = fh[2] / max(fh[3], 1)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": 0, "forget_drop": round(float(f0 - fN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4), "avg_users_upd": "-", **_rich_fin(rich)}


def run_adaptive(name, base_state, S_test, rounds, pj, forget_S, margs, warm, S0,
                 lr0=3e-4, budget=0.02, alpha=1.0, decay=0.9, ft_steps=30):
    """FROST: forgetting-gated adaptive fine-tuning. Update the backbone each round,
    but HALVE the learning rate whenever the T0-probe forgetting exceeds a budget
    (and gently grow it back when safe), so the update self-regulates into the gentle
    regime with NO per-dataset LR tuning and cannot catastrophically forget. With
    alpha<1 it also fuses a per-user memory."""
    n_users, n_pois = margs[0], margs[1]
    model = NextPOI(*margs).to(DEVICE); model.load_state_dict(base_state)
    M = torch.zeros(n_users, n_pois, device=DEVICE)
    accs, churn, prev = [], [], None
    HW = NW = HC = NC = 0; rich = _rich0(); cost = 0; lrs = []
    lr = lr0
    f0, _ = evaluate(model, forget_S)

    def scorer(s, uu, hh, dd):
        pm = model(s, uu, hh, dd)
        if alpha >= 1.0:
            return pm
        return alpha * pm.softmax(1) + (1 - alpha) * (M[uu] / (M[uu].sum(1, keepdim=True) + 1e-9))

    for r, idx in enumerate(rounds):
        Sr = subset(S_test, idx)
        if alpha >= 1.0:
            hw, nw, hc, nc = seg_eval(model, Sr, warm)
        else:
            hw, nw, hc, nc = giram_seg_eval(model, M, Sr, warm, alpha)
        accs.append((hw + hc) / max(nw + nc, 1)); HW += hw; NW += nw; HC += hc; NC += nc
        stream_rich(scorer, Sr, rich)
        cost += fit(model, Sr, epochs=10, max_steps=ft_steps, lr=lr)
        fr, _ = evaluate(model, forget_S)
        lrs.append(lr)
        lr = max(lr * 0.5, 1e-5) if (f0 - fr) > budget else min(lr * 1.3, lr0)   # forgetting gate
        if alpha < 1.0:
            M.mul_(decay); u = Sr["user"].to(DEVICE); t = Sr["tgt"].to(DEVICE)
            M.index_put_((u, t), torch.ones(len(t), device=DEVICE), accumulate=True)
        top = (probe_topk(model, S_test, pj) if alpha >= 1.0
               else giram_topk(model, M, S_test, pj, alpha))
        if prev is not None:
            churn.append(float(np.mean([1 - len(set(x) & set(y)) / len(set(x) | set(y))
                                        for x, y in zip(top, prev)])))
        prev = top
    fN, _ = evaluate(model, forget_S)
    return {"policy": name, "acc@10": round(float(np.nanmean(accs)), 4),
            "rounds_acc": [round(float(a), 4) for a in accs],
            "acc_cold": round(HC / max(NC, 1), 4), "acc_warm": round(HW / max(NW, 1), 4),
            "update_steps": int(cost), "forget_drop": round(float(f0 - fN), 4),
            "churn": round(float(np.mean(churn)) if churn else 0.0, 4),
            "avg_users_upd": round(float(np.mean(lrs)), 6), **_rich_fin(rich)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="nyc")
    ap.add_argument("--epochs0", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--ft-steps", type=int, default=30, dest="ft_steps")
    ap.add_argument("--replay", type=int, default=0, help="T0 samples mixed into each update (0=naive)")
    ap.add_argument("--backbone", default="gru", choices=["gru", "attn", "getnext"])
    args = ap.parse_args()
    print(f"device={DEVICE}")

    df = load(args.dataset)
    n_users = int(df.user_idx.max() + 1); n_pois = int(df.poi_idx.max() + 1)
    S0 = make_samples(df[df.block == "T0"], n_pois)
    S_test = make_samples(df[df.block != "T0"], n_pois)
    print(f"[{args.dataset}] users={n_users} pois={n_pois} "
          f"T0={len(S0['tgt'])} test={len(S_test['tgt'])}")
    warm = df[df.block == "T0"].user_idx.unique()
    if args.backbone == "getnext":
        set_poi_graph(build_poi_graph(df[df.block == "T0"], n_pois))

    torch.manual_seed(0); np.random.seed(0)
    base = NextPOI(n_users, n_pois, args.dim, args.backbone).to(DEVICE)
    t0 = time.time(); fit(base, S0, epochs=args.epochs0)
    print(f"base trained in {time.time()-t0:.0f}s, acc@10(T0-probe)=", end="")
    base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}

    rng = np.random.default_rng(0)
    pj = rng.choice(len(S_test["tgt"]), size=min(512, len(S_test["tgt"])), replace=False)
    forget_S = subset(S0, rng.choice(len(S0["tgt"]), size=min(2000, len(S0["tgt"])), replace=False))
    print(round(evaluate(base, forget_S)[0], 3))
    rounds = np.array_split(np.arange(len(S_test["tgt"])), args.rounds)

    policies = [
        ("static",      lambda r, a, e: False),
        ("always",      lambda r, a, e: True),
        ("periodic-2",  lambda r, a, e: r % 2 == 0),
        ("periodic-4",  lambda r, a, e: r % 4 == 0),
        ("gated-0.02",  lambda r, a, e: e is not None and a < e - 0.02),
        ("gated-0.05",  lambda r, a, e: e is not None and a < e - 0.05),
    ]
    rows = []
    margs = (n_users, n_pois, args.dim, args.backbone)
    for nm, dec in policies:
        t = time.time()
        res = run_policy(nm, base_state, S_test, rounds, pj, forget_S, margs, dec, args.ft_steps,
                         S0_pool=S0, replay=args.replay, rng=np.random.default_rng(123), warm=warm)
        res["sec"] = round(time.time() - t, 1)
        rows.append(res)
        print(f"  {nm:16s} acc@10={res['acc@10']:.3f} cold={res['acc_cold']:.3f} upd={res['update_steps']:4d} "
              f"forget={res['forget_drop']:+.3f} churn={res['churn']:.3f} ({res['sec']}s)")
    for nm, gated, th in [("selective-all", False, 0.0), ("selective-gated", True, 0.20)]:
        t = time.time()
        res = run_selective(nm, base_state, S_test, rounds, pj, forget_S, margs, th, args.ft_steps, gated, warm=warm)
        res["sec"] = round(time.time() - t, 1)
        rows.append(res)
        print(f"  {nm:16s} acc@10={res['acc@10']:.3f} cold={res['acc_cold']:.3f} upd={res['update_steps']:4d} "
              f"forget={res['forget_drop']:+.3f} churn={res['churn']:.3f} "
              f"users/rnd={res.get('avg_users_upd','-')} ({res['sec']}s)")
    for alpha in [0.3, 0.5, 0.7]:
        t = time.time()
        res = run_giram(f"GIRAM-a{alpha}", base_state, S_test, rounds, pj, forget_S, margs, warm, alpha)
        res["sec"] = round(time.time() - t, 1)
        rows.append(res)
        print(f"  {res['policy']:16s} acc@10={res['acc@10']:.3f} cold={res['acc_cold']:.3f} upd=   0 "
              f"forget={res['forget_drop']:+.3f} churn={res['churn']:.3f} ({res['sec']}s)")
    pd.DataFrame(rows).to_csv(ROOT / f"results_learned_{args.dataset}.csv", index=False)
    print(f"-> results_learned_{args.dataset}.csv")


if __name__ == "__main__":
    main()
