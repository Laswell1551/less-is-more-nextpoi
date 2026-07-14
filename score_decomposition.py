#!/usr/bin/env python3
"""
score_decomposition.py -- where does the score actually COME from?

    Acc@10  =  s_ret * a_ret   +   s_dis * a_dis
               \___________/       \___________/
                return term         discovery term

The two terms are not comparable. Return accuracy is attainable at 0.70-0.91. Discovery accuracy,
FOR ANY METHOD WE RAN, tops out at 0.065. So the return term dominates the sum even where returns
are a minority of the instances.

TWO VERSIONS, and the second is the one that carries the argument.

  (1) PER-METHOD. For each method, what share of ITS score comes from returns? For the counter
      this is 96-100%. But the counter's discovery accuracy is near zero more or less BY
      CONSTRUCTION -- it cannot rank an unvisited POI above a visited one -- so on its own this
      number risks reading as a tautology about the counter rather than a fact about the
      benchmark. It is necessary but not sufficient.

  (2) BEST-ACHIEVABLE (the honest, method-independent version). Take the BEST return accuracy
      anyone achieved and the BEST discovery accuracy anyone achieved -- across every method we
      ran, including the popularity floor and the neural model that beats it on Gowalla-CA. Give
      a single hypothetical method BOTH, for free. What share of ITS score would come from
      returns?

      That number is a property of the BENCHMARK, not of any model. It says: no matter how good
      you get at discovery -- as good as the best thing anyone has -- the metric will not pay you
      for it.

Writes score_decomposition.json.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DS = ["nyc", "tky", "gowalla_ca", "brightkite"]
LAB = {"nyc": "Foursquare-NYC", "tky": "Foursquare-TKY",
       "gowalla_ca": "Gowalla-CA", "brightkite": "Brightkite-US"}


def main():
    d = json.loads((ROOT / "discovery_stream.json").read_text())
    out = {}

    print("(1) PER-METHOD: what share of each method's OWN score comes from returns?\n")
    print(f"  {'dataset':16s} {'ret share':>10s}" + "".join(f"{m:>14s}" for m in
                                                            ("popularity", "neural", "COUNT")))
    for k in DS:
        e = d[k]
        s_ret = e["return_share"]
        s_dis = 1.0 - s_ret
        row = {"return_share": round(s_ret, 4)}
        cells = []
        for m in ("popularity", "static (neural)", "COUNT"):
            a_ret = e["acc@10"][m]["return"]
            a_dis = e["acc@10"][m]["discovery"]
            rt, dt = s_ret * a_ret, s_dis * a_dis
            share = rt / (rt + dt) if (rt + dt) > 0 else float("nan")
            row[m] = round(share, 4)
            cells.append(f"{share:13.1%}")
        out[k] = row
        print(f"  {LAB[k]:16s} {s_ret:9.1%} " + "".join(cells))

    print("\n" + "=" * 92)
    print("(2) BEST-ACHIEVABLE: give ONE hypothetical method the best return accuracy AND the best")
    print("    discovery accuracy that ANY method achieved. What share of its score is returns?")
    print("    This is a property of the benchmark, not of any model.\n")
    print(f"  {'dataset':16s} {'ret sh':>7s} {'best a_ret':>11s} {'best a_dis':>11s} "
          f"{'(by)':>12s} {'ret term':>9s} {'dis term':>9s} {'DIS % OF SCORE':>15s}")
    for k in DS:
        e = d[k]
        s_ret = e["return_share"]
        s_dis = 1.0 - s_ret
        methods = list(e["acc@10"].keys())
        best_ret = max(e["acc@10"][m]["return"] for m in methods)
        best_dis = max(e["acc@10"][m]["discovery"] for m in methods)
        who_dis = max(methods, key=lambda m: e["acc@10"][m]["discovery"])
        rt, dt = s_ret * best_ret, s_dis * best_dis
        dis_share = dt / (rt + dt)
        c = e["acc@10"]["COUNT"]
        out[k].update({
            "best_a_ret": round(best_ret, 4),
            "best_a_dis": round(best_dis, 4),
            "best_a_dis_by": who_dis,
            "BEST_ACHIEVABLE_return_share_of_score": round(1 - dis_share, 4),
            "BEST_ACHIEVABLE_discovery_share_of_score": round(dis_share, 4),
            "COUNT_return_over_discovery_ratio": round(c["return"] / max(c["discovery"], 1e-9), 1),
        })
        short = {"popularity": "popularity", "static (neural)": "neural", "COUNT": "COUNT"}[who_dis]
        print(f"  {LAB[k]:16s} {s_ret:6.1%} {best_ret:11.4f} {best_dis:11.4f} "
              f"{short:>12s} {rt:9.4f} {dt:9.4f} {dis_share:14.1%}")

    lo = min(out[k]["BEST_ACHIEVABLE_return_share_of_score"] for k in DS)
    hi = max(out[k]["BEST_ACHIEVABLE_return_share_of_score"] for k in DS)
    plo = min(out[k]["COUNT"] for k in DS)
    phi = max(out[k]["COUNT"] for k in DS)
    rlo = min(out[k]["COUNT_return_over_discovery_ratio"] for k in DS)
    rhi = max(out[k]["COUNT_return_over_discovery_ratio"] for k in DS)

    print(f"\n  => PER-METHOD (COUNT):   returns supply {plo:.1%}-{phi:.1%} of its score.")
    print(f"  => BEST-ACHIEVABLE:      even at the best discovery accuracy ANY method reaches,")
    print(f"                           returns still supply {lo:.1%}-{hi:.1%} of the score.")
    print(f"  => For COUNT, return accuracy is {rlo:.0f}-{rhi:.0f}x its discovery accuracy.")

    out["_summary"] = {
        "COUNT_return_share_of_score_range": [round(plo, 4), round(phi, 4)],
        "best_achievable_return_share_range": [round(lo, 4), round(hi, 4)],
        "COUNT_return_over_discovery_ratio_range": [rlo, rhi],
    }
    (ROOT / "score_decomposition.json").write_text(json.dumps(out, indent=2))
    print("\n-> score_decomposition.json")


if __name__ == "__main__":
    main()
