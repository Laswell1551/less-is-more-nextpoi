#!/usr/bin/env python3
"""
download_data.py -- fetch raw check-in datasets for the Next-POI study.

Datasets
--------
  Foursquare TSMC2014 (NYC + TKY)   Yang et al., IEEE T-SMC 2015
      mirror: http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip
      columns (tab): user, venue, catId, catName, lat, lon, tzOffsetMin, utcTime
  Gowalla total check-ins           Cho et al., KDD 2011 (via SNAP)
      https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz
      columns (tab): user, time(ISO-Z), lat, lon, locationId

Raw files land in <root>/data/raw/. Idempotent: a file is skipped if it already
exists with the expected byte size. Auto-download first; if a source is ever
unreachable, drop the archive into data/raw/ by hand and re-run -- extraction
will proceed from the local copy.
"""
import argparse
import gzip
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SOURCES = {
    "foursquare": {
        "url": "http://www-public.imtbs-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip",
        "archive": "dataset_tsmc2014.zip",
        "size": 25546284,
        "members": ["dataset_TSMC2014_NYC.txt", "dataset_TSMC2014_TKY.txt"],
    },
    "gowalla": {
        "url": "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz",
        "archive": "loc-gowalla_totalCheckins.txt.gz",
        "size": 105470044,
        "members": ["loc-gowalla_totalCheckins.txt"],
    },
    "brightkite": {
        "url": "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz",
        "archive": "loc-brightkite_totalCheckins.txt.gz",
        "size": None,
        "members": ["loc-brightkite_totalCheckins.txt"],
    },
}
UA = "Mozilla/5.0 (academic research; next-poi-ipm) Python-urllib"


def download(url, dest, expected_size=None):
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  [skip] {dest.name} present ({dest.stat().st_size:,} B)")
        return dest
    print(f"  [get ] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)
    if expected_size and dest.stat().st_size != expected_size:
        print(f"  [warn] size mismatch: got {dest.stat().st_size:,}, expected {expected_size:,}",
              file=sys.stderr)
    return dest


def extract_zip(archive, members, out_dir):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        for m in members:
            hit = next((n for n in names if Path(n).name == Path(m).name), None)
            if hit is None:
                print(f"  [warn] {m} not found in {Path(archive).name}", file=sys.stderr)
                continue
            target = Path(out_dir) / Path(m).name
            with z.open(hit) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"  [ext ] {target.name} ({target.stat().st_size:,} B)")


def extract_gz(archive, out_name, out_dir):
    target = Path(out_dir) / out_name
    with gzip.open(archive, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"  [ext ] {target.name} ({target.stat().st_size:,} B)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent),
                    help="project root (default: this script's dir)")
    ap.add_argument("--datasets", nargs="+", default=["foursquare", "gowalla"],
                    choices=list(SOURCES))
    args = ap.parse_args()

    raw = Path(args.root) / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        s = SOURCES[name]
        print(f"[{name}]")
        arc = download(s["url"], raw / s["archive"], s.get("size"))
        if str(arc).endswith(".zip"):
            extract_zip(arc, s["members"], raw)
        elif str(arc).endswith(".gz"):
            extract_gz(arc, s["members"][0], raw)
    print(f"done -> {raw}")


if __name__ == "__main__":
    main()
