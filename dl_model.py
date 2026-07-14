#!/usr/bin/env python3
"""Download the LLM for llm_audit.py.

The first attempt stalled: duplicate partial blobs, zero bytes/min, safetensors shards
never landing. Diagnosis (measured, not guessed):
  * huggingface.co serves bytes fine -- 1.16 MB/s on a ranged GET of a real shard.
  * hf-mirror.com answers 308 Permanent Redirect, which huggingface_hub will not follow
    (LocalEntryNotFoundError), so the mirror is not usable here.
  * The hang was the accelerated-transfer backends, not the endpoint.
So: direct endpoint, Xet and hf_transfer both OFF, parallel shard workers, and a retry
loop because the link drops. ~15 GB at ~1 MB/s per stream, 4 streams => roughly an hour.
"""
import os
import sys
import time

os.environ.pop("HF_ENDPOINT", None)              # direct; the mirror 308s and hub won't follow
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"    # deprecated
os.environ["HF_HUB_DISABLE_XET"] = "1"           # this is what hung

from huggingface_hub import snapshot_download   # noqa: E402  (must follow the env vars)

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
print(f"endpoint=direct (huggingface.co)  model={MODEL}", flush=True)

for attempt in range(1, 21):
    try:
        t0 = time.time()
        p = snapshot_download(
            MODEL,
            allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json"],
            max_workers=4,
        )
        print(f"\nDONE {MODEL} -> {p}  ({time.time()-t0:.0f}s, attempt {attempt})", flush=True)
        break
    except Exception as e:                       # network drops are expected; resume
        print(f"[attempt {attempt}] {type(e).__name__}: {str(e)[:160]} -- retrying",
              flush=True)
        time.sleep(5)
else:
    print("FAILED after 20 attempts", flush=True)
    sys.exit(1)
