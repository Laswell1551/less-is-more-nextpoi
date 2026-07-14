"""Compare the CURRENT (user-major) round split against a TRUE chronological stream.
Same base model, same seed, same everything else -- only the ordering of S_test changes."""
import numpy as np, pandas as pd, torch, sys, time
sys.path.insert(0, '.')
import learned_continual as lc

DS = sys.argv[1] if len(sys.argv) > 1 else "nyc"
BB = sys.argv[2] if len(sys.argv) > 2 else "gru"

df = lc.load(DS)
n_users = int(df.user_idx.max()+1); n_pois = int(df.poi_idx.max()+1)
S0 = lc.make_samples(df[df.block=="T0"], n_pois)

# ---- rebuild S_test, but ALSO capture each sample's timestamp -------------
te = df[df.block!="T0"]
times = []
for _, g in te.groupby("traj_id", sort=False):
    t = g.unix_s.to_numpy()
    times.extend(t[1:])          # make_samples emits one sample per i in 1..len-1
times = np.asarray(times)
S_test = lc.make_samples(te, n_pois)
assert len(times) == len(S_test["tgt"])

order = np.argsort(times, kind="stable")          # TRUE chronological order
S_chrono = {k: v[torch.as_tensor(order)] for k, v in S_test.items()}

warm = df[df.block=="T0"].user_idx.unique()
if BB == "getnext":
    lc.set_poi_graph(lc.build_poi_graph(df[df.block=="T0"], n_pois))

torch.manual_seed(0); np.random.seed(0)
base = lc.NextPOI(n_users, n_pois, 128, BB).to(lc.DEVICE)
lc.fit(base, S0, epochs=8)
base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
margs = (n_users, n_pois, 128, BB)

rng = np.random.default_rng(0)
forget_S = lc.subset(S0, rng.choice(len(S0["tgt"]), size=2000, replace=False))

def run_all(S, tag):
    R = np.array_split(np.arange(len(S["tgt"])), 15)
    pj = np.random.default_rng(0).choice(len(S["tgt"]), size=512, replace=False)
    out = {}
    out["static"]   = lc.run_policy("static", base_state, S, R, pj, forget_S, margs,
                                    lambda r,a,e: False, 30, S0_pool=S0, replay=1024,
                                    rng=np.random.default_rng(123), warm=warm)
    out["always+replay"] = lc.run_policy("always", base_state, S, R, pj, forget_S, margs,
                                    lambda r,a,e: True, 30, S0_pool=S0, replay=1024,
                                    rng=np.random.default_rng(123), warm=warm)
    out["GIRAM-memory"] = lc.run_giram("giram", base_state, S, R, pj, forget_S, margs, warm, 0.5)
    out["selective"] = lc.run_selective("sel", base_state, S, R, pj, forget_S, margs,
                                    0.20, 30, True, warm=warm)
    for k, v in out.items():
        print(f"  {tag:8s} {k:14s} acc@10={v['acc@10']:.4f}  forget={v['forget_drop']:+.3f}  churn={v['churn']:.3f}")
    return out

print(f"\n=== {DS} / {BB} / seed 0 ===")
print("\n-- CURRENT protocol (user-major chunks) --")
cur = run_all(S_test, "user")
print("\n-- FIXED protocol (true chronological stream) --")
fix = run_all(S_chrono, "chrono")

print("\n=== DELTA vs static, under each protocol ===")
print(f"{'method':16s} {'current':>10s} {'chrono':>10s}")
for m in ["always+replay", "GIRAM-memory", "selective"]:
    dc = cur[m]["acc@10"] - cur["static"]["acc@10"]
    df_ = fix[m]["acc@10"] - fix["static"]["acc@10"]
    print(f"{m:16s} {dc:+10.4f} {df_:+10.4f}")
print(f"{'static (abs)':16s} {cur['static']['acc@10']:10.4f} {fix['static']['acc@10']:10.4f}")
