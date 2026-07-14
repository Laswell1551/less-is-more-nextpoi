#!/usr/bin/env python3
"""
ranking.py -- ONE definition of "the rank of the target", used by every experiment.

WHY THIS FILE EXISTS

Every ranking site in this project originally computed

    rank = (scores > scores[target]).sum() + 1

which places the target FIRST among everything it ties with. That is the OPTIMISTIC convention,
and it is not what a deployment gets.

It matters here more than it usually would, because the two methods being compared tie at wildly
different rates:

  * The counter's score is  <integer visit count> + 1e-6 * <normalised popularity>, and
    popularity is itself an integer count. Two POIs with the same visit count AND the same
    popularity count therefore have EXACTLY equal scores. On NYC the target is tied with at
    least one other POI in ~40% of instances.

  * GETNext's scores are float32 activations from a neural network. Exact ties are ~0%.

So the optimistic convention silently hands the counter an advantage it denies its opponent. It
inflated our headline: COUNT-traj Acc@10 read .6505 optimistically and .6211 honestly, and the
margin over GETNext went from a claimed +17.5% to a real +12.0%.

WHAT WE USE INSTEAD

    expected rank = (scores > s*).sum() + (n_tied_with_target + 1) / 2

This is the expected rank under random tie-breaking, which is what argsort on an arbitrary index
order gives you in expectation -- i.e. what a real system actually achieves. It is exact for
tie-free scores, so it changes nothing for GETNext, and it costs the counter precisely the
advantage it should never have had.

Never reintroduce `(s > s_target).sum() + 1`.
"""
import numpy as np
import torch


def expected_rank_np(scores, target_idx):
    """Expected rank (1 = best) of `target_idx` under random tie-breaking. NumPy, 1-D."""
    st = scores[target_idx]
    gt = int((scores > st).sum())      # strictly better
    eq = int((scores == st).sum())     # tied with the target, INCLUDING the target itself
    return gt + (eq + 1) / 2.0


def expected_rank_torch(scores, targets):
    """Vectorised expected rank. scores: (B, N). targets: (B,). Returns float tensor (B,)."""
    st = scores.gather(1, targets.unsqueeze(1))          # (B, 1)
    gt = (scores > st).sum(1)                            # strictly better
    eq = (scores == st).sum(1)                           # tied, includes the target
    return gt.float() + (eq.float() + 1.0) / 2.0


def optimistic_rank_torch(scores, targets):
    """The old convention. Kept ONLY so experiments can report the gap. Do not use for results."""
    st = scores.gather(1, targets.unsqueeze(1))
    return (scores > st).sum(1) + 1


def tie_fraction_torch(scores, targets):
    """Share of instances whose target is tied with >=1 other item."""
    st = scores.gather(1, targets.unsqueeze(1))
    eq = (scores == st).sum(1)
    return (eq > 1).float().mean().item()


def metrics_from_ranks(ranks, ks=(1, 5, 10, 20)):
    r = np.asarray(ranks, dtype=np.float64)
    out = {f"acc@{k}": round(float((r <= k).mean()), 4) for k in ks}
    out["mrr"] = round(float((1.0 / r).mean()), 4)
    return out
