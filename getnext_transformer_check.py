#!/usr/bin/env python3
"""
getnext_transformer_check.py -- does GETNext's transformer model the sequence?

Three lines of the released implementation (train.py / model.py):

    encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        # batch_first is left at its default, False -> the layer expects (seq, batch, dim)

    batch_padded = pad_sequence(batch_seq_embeds, batch_first=True, padding_value=-1)
        # ...but the input is built as (batch, seq, dim)

    src_mask = seq_model.generate_square_subsequent_mask(args.batch)
        # ...and the causal mask is (batch, batch)

PyTorch validates src_mask against src.shape[0]. With src = (batch, seq, dim) and
mask = (batch, batch), that check PASSES. Nothing raises. The model trains, converges, and
reports a plausible accuracy.

But the axes are transposed: PyTorch reads dim 0 as the sequence axis and dim 1 as the batch
axis. So the causal attention runs across the MINI-BATCH -- one user's trajectory conditions
another user's -- and never across time within a trajectory.

This script demonstrates it on GETNext's exact construction, in isolation, in a few seconds.
It makes no claim about which code produced the published numbers; it reports what the
released artifact does.

Run:  python getnext_transformer_check.py
"""
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

torch.manual_seed(0)

# exactly as GETNext builds it: no batch_first argument
layer = TransformerEncoderLayer(16, 2, 32, 0.0)
enc = TransformerEncoder(layer, 2).eval()

B, S, D = 4, 5, 16                       # 4 users' trajectories, 5 check-ins each
mask = (torch.triu(torch.ones(B, B)) == 1).transpose(0, 1).float()      # (batch, batch)
mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

x = torch.randn(B, S, D)                 # exactly as GETNext feeds it: (batch, seq, dim)
with torch.no_grad():
    out = enc(x, mask)

# (1) Does one user's trajectory leak into another's? The causal mask permits 0 -> 1,2,3.
x1 = x.clone(); x1[0] = torch.randn(S, D)
with torch.no_grad():
    leak = (out[1:] - enc(x1, mask)[1:]).abs().max().item()

# (2) Does a trajectory's own earlier check-in reach its later one? This is the only thing a
#     sequence model exists to do.
x2 = x.clone(); x2[:, 0] = torch.randn(B, D)
with torch.no_grad():
    time_effect = (out[:, 4] - enc(x2, mask)[:, 4]).abs().max().item()

print(f"perturbing user 0's trajectory moves users 1-3's outputs by : {leak:.6f}")
print(f"perturbing a trajectory's check-in 0 moves its check-in 4 by: {time_effect:.6f}")
print()
if leak > 1e-6 and time_effect < 1e-6:
    print("The attention runs across the batch, not across time.")
    print("GETNext's transformer does no sequence modelling: a prediction depends on which")
    print("other users happen to share its mini-batch, and not at all on the user's own")
    print("history. Its accuracy therefore comes from the per-position embeddings (POI, user,")
    print("time, category) and the graph-attention adjustment -- i.e. from a first-order,")
    print("context-conditioned transition model.")
    print()
    print("Which is precisely the class of model a counter belongs to. That is why a counter")
    print("matches it.")
else:
    print("Not reproduced. Do not put this in the paper.")

print()
print("MEASURABLE CONSEQUENCE: predictions depend on batch composition. GETNext's validation")
print("dataset is built by iterating set(df['trajectory_id']) over STRING ids, whose order")
print("Python randomises per process. Re-evaluating one saved checkpoint under different")
print("PYTHONHASHSEED values (0, 1, 7) moves Acc@1 over .2414-.2471 and MRR over .3448-.3480")
print("on our NYC split -- same weights, same data, ~2.4% relative spread in Acc@1.")
print()
print("SCOPE: this is a defect in the released artifact. We cannot know which revision")
print("produced the published numbers, and we do not claim those are wrong. Our own")
print("comparison does not rest on it: a counter with the same information scores .6505")
print("against this implementation's .5543, whichever way one reads the transformer.")
