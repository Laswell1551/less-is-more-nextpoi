#!/usr/bin/env python3
"""
getnext_select.py -- pin down GETNext's reported row using GETNext's OWN model-selection rule.

WHY THIS EXISTS
The paper's central table quotes five metrics for official GETNext. Those five must come from
ONE epoch -- the epoch a practitioner would actually deploy -- not from a per-metric maximum
taken across different epochs, which would flatter it. GETNext's train.py selects on

    argmax over epochs of  4 * Acc@1 + Acc@20

so we apply exactly that rule to its own per-epoch validation log and read off all five metrics
at the winning epoch. Writes getnext_selected.json, which the manuscript and the verifier both
read. Nothing here is hand-copied.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GN = ROOT.parent / "baselines" / "GETNext" / "runs" / "nyc"

KEYS = ["top1_acc", "top5_acc", "top10_acc", "top20_acc", "mrr"]
OUT_KEYS = ["acc@1", "acc@5", "acc@10", "acc@20", "mrr"]


def series(txt, key):
    m = re.search(rf"val_epochs_{key}_list=\[(.*?)\]", txt, re.S)
    if not m:
        raise KeyError(f"val_epochs_{key}_list not found")
    return [float(x) for x in m.group(1).split(",") if x.strip()]


def main():
    out = {}
    for tag, sub in (("as-released", "asreleased"), ("index-corrected", "fixed")):
        txt = (GN / sub / "metrics-val.txt").read_text()
        S = {k: series(txt, k) for k in KEYS}
        n = len(S["top1_acc"])
        assert all(len(v) == n for v in S.values()), "ragged metric lists"

        # GETNext's own rule, from its train.py
        crit = [4 * S["top1_acc"][i] + S["top20_acc"][i] for i in range(n)]
        best = max(range(n), key=lambda i: crit[i])

        row = {ok: round(S[k][best], 4) for ok, k in zip(OUT_KEYS, KEYS)}
        row["epoch"] = best
        row["n_epochs"] = n
        row["criterion"] = round(crit[best], 4)
        out[tag] = row

        # what a per-metric maximum WOULD have given -- to show we are not doing that
        cherry = {ok: round(max(S[k]), 4) for ok, k in zip(OUT_KEYS, KEYS)}
        out[tag + " (per-metric max, NOT used)"] = cherry

        print(f"{tag:16s} epoch {best:2d}/{n}  "
              + "  ".join(f"{ok}={row[ok]:.4f}" for ok in OUT_KEYS))
        print(f"{'':16s} cherry-picked would be: "
              + "  ".join(f"{ok}={cherry[ok]:.4f}" for ok in OUT_KEYS))

    (ROOT / "getnext_selected.json").write_text(json.dumps(out, indent=2))
    print("\n-> getnext_selected.json")


if __name__ == "__main__":
    main()
