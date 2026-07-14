#!/usr/bin/env bash
# Fetch the LLM with plain resumable curl, in parallel.
#
# Why not huggingface_hub: it hangs at "Fetching 11 files: 0%" on this network with no
# error and no bytes, while a raw ranged GET of the same shard pulls at ~1.16 MB/s. The
# mirror (hf-mirror.com) answers 308 and the hub client won't follow it. So we skip the
# library. `curl -C -` resumes, and the loop retries, so a dropped link costs nothing.
#
# Downloads into a plain directory; transformers loads a local path just as happily as a
# hub id (llm_audit.py --model-path points at it).
set -u

REPO="${1:-Qwen/Qwen2.5-7B-Instruct}"
DEST="${2:-models/$(basename "$REPO")}"
BASE="https://huggingface.co/${REPO}/resolve/main"

FILES=(
  config.json generation_config.json model.safetensors.index.json
  tokenizer.json tokenizer_config.json vocab.json merges.txt
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)

mkdir -p "$DEST"
echo "-> $DEST  (from $BASE)"

fetch() {
  local f="$1"
  for attempt in $(seq 1 40); do
    # -C - resumes; -f fails on 4xx/5xx so we retry instead of writing an HTML error page
    if curl -sfL -C - --retry 5 --retry-delay 3 --connect-timeout 20 \
            -o "$DEST/$f" "$BASE/$f"; then
      echo "   ok  $f"
      return 0
    fi
    sleep 3
  done
  echo "   FAILED  $f"
  return 1
}

for f in "${FILES[@]}"; do fetch "$f" & done
wait
echo "--- done ---"
ls -la "$DEST" | awk '{printf "  %10s  %s\n", $5, $9}'
