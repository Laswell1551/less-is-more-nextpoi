"""Does the per-user MEMORY capture the injected new-POI headroom that fine-tuning cannot?
static vs memory (GIRAM-core) vs controller, across injection rates."""
import numpy as np, torch
import learned_continual as L
from injection_test import setup
for ds in ['nyc', 'brightkite']:
    df = L.load(ds)
    for frac in [0.0, 0.15, 0.30]:
        com, warm, S0, npr = setup(df, frac, 0)
        st = L.run_policy('s', *com, (lambda r,a,e: False), 30, S0, 1024, np.random.default_rng(123), warm)
        mem = L.run_giram('g', *com, warm, 0.5, 0.9)
        ct = L.run_controller('c', *com, ft_steps=30, lr0=1e-3, eps=0.005, warm=warm, grow=1.0)
        d_mem = mem['acc@10'] - st['acc@10']; d_ct = ct['acc@10'] - st['acc@10']
        print(f"{ds:10s} f={frac:.2f} newPOI={npr:.2f} | static={st['acc@10']:.4f} | memory={mem['acc@10']:.4f} ({d_mem:+.4f},f{mem['forget_drop']:+.2f}) | CTRL={ct['acc@10']:.4f} ({d_ct:+.4f})")
