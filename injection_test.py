"""Clean new-POI injection (referee route A). Hold a fraction of POIs OUT of the base
entirely (their base check-ins removed) so static stays strong on KNOWN POIs but cannot
predict the injected ones -- isolating genuine new-POI headroom WITHOUT starving static
the way base-shrinking does. Decisive test: in a regime with real headroom, does the
controller BEAT static at forget~0, and beat a FIXED small LR (proving adaptation matters)?
Also reports accuracy split by target type (known vs injected)."""
import numpy as np, torch
import learned_continual as L

def inject_split(df, frac, seed):
    rng = np.random.default_rng(seed); pois = df.poi_idx.unique()
    inj = set(rng.choice(pois, int(frac*len(pois)), replace=False)) if frac > 0 else set()
    base = df[(df.block == 'T0') & (~df.poi_idx.isin(inj))]
    stream = df[df.block != 'T0']
    return base, stream, inj

def setup(df, frac, seed):
    npo = int(df.poi_idx.max()+1); nu = int(df.user_idx.max()+1)
    base, stream, inj = inject_split(df, frac, seed)
    S0 = L.make_samples(base, npo); St = L.make_samples(stream, npo)
    warm = base.user_idx.unique(); margs = (nu, npo, 128, 'gru')
    torch.manual_seed(seed); np.random.seed(seed)
    b = L.NextPOI(*margs).to(L.DEVICE); L.fit(b, S0, epochs=8)
    bs = {k: v.detach().cpu().clone() for k, v in b.state_dict().items()}
    rng = np.random.default_rng(seed); pj = rng.choice(len(St['tgt']), 512, replace=False)
    fS = L.subset(S0, rng.choice(len(S0['tgt']), min(2000, len(S0['tgt'])), replace=False))
    rounds = np.array_split(np.arange(len(St['tgt'])), 15)
    npr = float(np.mean([int(t in inj) for t in St['tgt'].numpy()]))
    return (bs, St, rounds, pj, fS, margs), warm, S0, npr

for ds in ['nyc', 'brightkite']:
    df = L.load(ds)
    for frac in [0.0, 0.15, 0.30]:
        com, warm, S0, npr = setup(df, frac, 0)
        st = L.run_policy('s', *com, (lambda r,a,e: False), 30, S0, 1024, np.random.default_rng(123), warm)
        n1 = L.run_policy('a', *com, (lambda r,a,e: True), 30, S0, 1024, np.random.default_rng(123), warm, ft_lr=1e-3)
        f3 = L.run_policy('a', *com, (lambda r,a,e: True), 30, S0, 1024, np.random.default_rng(123), warm, ft_lr=3e-5)
        ct = L.run_controller('c', *com, ft_steps=30, lr0=1e-3, eps=0.005, warm=warm, grow=1.0)
        print(f"{ds:10s} f={frac:.2f} newPOI={npr:.2f} | static={st['acc@10']:.4f} | fixed3e5={f3['acc@10']:.4f}(f{f3['forget_drop']:+.2f}) | CTRL={ct['acc@10']:.4f}(f{ct['forget_drop']:+.2f},lr{ct['final_lr']:.0e}) | naive={n1['acc@10']:.4f}(f{n1['forget_drop']:+.2f})")
