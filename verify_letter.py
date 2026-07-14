#!/usr/bin/env python3
"""
verify_letter.py -- the response letter is a document too, and nobody was checking it.

WHY THIS EXISTS

verify_manuscript.py and verify_prose.py check main_v2.tex. They do not check
response_to_editor.md, and for weeks nothing did. So the letter drifted, in exactly the way the
paper says documents drift, and it drifted into claims an editor can falsify in one click:

  * it claimed "11 numbered equations and 2 definitions" when the manuscript had 3 and 0 -- those
    counts had been carried over from the SUPERSEDED manuscript (main_lessismore.tex), which the
    v2 rewrite replaced. Point 11 of the letter is a reply to "match the level of research of
    recent IP&M articles", and its whole evidence was that sentence.
  * it claimed 25 pages / 12 tables / 6 figures. Same origin, all wrong.
  * it claimed "the bibliography went to 101 entries" when 101 is the size of the .bib file and
    the printed reference list has 62. An editor asked about the reference list, and counts it.
  * its Summary-of-changes table still carried a TITLE we had retracted two rounds earlier.
  * its prose quoted a +12.1% margin while its OWN table two hundred lines up said +12.4%.

None of these were lies. All of them were a document that moved while a copy of its numbers did
not -- which is the paper's thesis, committed against the paper's own cover letter.

WHAT THIS CHECKS

  (1) STRUCTURE. Every structural claim in the letter, recomputed from main_v2.tex / .log / .bbl:
      pages, tables, figures, numbered equations, definitions, printed references, IP&M
      references, abstract length, and the title.
  (2) NUMBERS. Every 3-4 decimal number in the letter must appear somewhere in the manuscript
      corpus (main_v2.tex + the generated tab_*.tex), OR be listed in SUPERSEDED below -- values
      the letter quotes precisely BECAUSE they are what we got wrong. Anything else is a number
      that exists only in the letter, which is how the +12.1% survived.

Exit 1 on any mismatch. Run it before sending.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent
TEX = DOC / "main_v2.tex"
LOG = DOC / "main_v2.log"
BBL = DOC / "main_v2.bbl"
BIB = DOC / "refs_v2.bib"
LET = DOC / "response_to_editor.md"

tex = TEX.read_text(encoding="utf-8")
log = LOG.read_text(encoding="utf-8", errors="ignore")
bbl = BBL.read_text(encoding="utf-8", errors="ignore")
let = LET.read_text(encoding="utf-8")

fails = []

# ---------------------------------------------------------------- (0) is the file even intact?
#
# This check exists because an earlier version of this script PASSED on a letter that had been
# silently destroyed. A PowerShell `Get-Content -Raw` (which, with no -Encoding, decodes using the
# system ANSI codepage -- GBK on this machine) followed by a UTF-8 write turned every em-dash into
# a CJK character and ATE the byte after it. 105 sites. The letter still contained all its ASCII
# digits, so every numeric check below sailed through and reported success on a ruined document.
#
# A checker that only looks at what it was told to look at will certify a corpse. Look at the file.
cjk = sum(1 for c in let if "一" <= c <= "鿿")
repl = let.count("�")
print("Encoding integrity:\n")
print(f"  [{'ok ' if not cjk else 'FAIL'}] CJK characters (mojibake)       {cjk}")
print(f"  [{'ok ' if not repl else 'FAIL'}] U+FFFD replacement characters   {repl}")
if cjk:
    fails.append("mojibake: the letter was decoded with the wrong codepage somewhere")
if repl:
    fails.append("U+FFFD in the letter")
# an em-dash followed immediately by a digit or '%' is the signature of the eaten byte
for m in re.finditer(r"—\s?[\d%]", let):
    fails.append(f"eaten byte after an em-dash: {let[m.start()-30:m.start()+12]!r}")
    print(f"  [FAIL] em-dash swallowed the character after it: "
          f"...{let[m.start()-30:m.start()+12]}...")
print()


def check(label, claimed, actual):
    ok = claimed == actual
    print(f"  [{'ok ' if ok else 'FAIL'}] {label:34s} letter says {str(claimed):>7s}   "
          f"actual {str(actual):>7s}")
    if not ok:
        fails.append(label)


# --------------------------------------------------------------- (1) structure, from the source
def n(pat, s=tex):
    return len(re.findall(pat, s))


pages = int(re.search(r"Output written on .*?\((\d+) pages", log).group(1))
tables = n(r"\\begin\{table")
figures = n(r"\\begin\{inlinefig\}|\\begin\{figure")
# a numbered equation is one that gets a number: equation, and each line of an align
equations = n(r"\\begin\{equation\}") + n(r"\\\\", re.search(
    r"(?s)\\begin\{align\}.*?\\end\{align\}", tex).group(0)) + n(r"\\begin\{align\}")
definitions = n(r"\\begin\{definition\}")
printed_refs = n(r"\\bibitem", bbl)

# IP&M entries that are actually CITED (i.e. reach the printed list). "Information Processing"
# also matches "Advances in Neural Information Processing Systems" -- exclude NeurIPS explicitly,
# a false positive that cost us a wrong count once already.
ipm = 0
for m in re.finditer(r"(?s)@\w+\{([^,]+),(.*?)(?=\n@|\Z)", BIB.read_text(encoding="utf-8")):
    key, body = m.group(1).strip(), m.group(2)
    if re.search(r"Information Processing", body) and "Neural Information" not in body:
        if re.search(r"\{" + re.escape(key) + r"\}", bbl):
            ipm += 1

abs_body = re.search(r"(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}", tex).group(1)
abs_words = len(re.sub(r"\\[a-zA-Z]+\{?|\}|\$|~|\\\\", "", abs_body).split())
title = re.search(r"\\title\[mode=title\]\{(.*?)\}", tex).group(1)


def claimed_int(pat):
    m = re.search(pat, let)
    return int(m.group(1)) if m else None


print("Structural claims in response_to_editor.md, recomputed from the manuscript:\n")
check("pages", claimed_int(r"\*\*(\d+) pages"), pages)
check("tables", claimed_int(r"(\d+) tables"), tables)
check("figures", claimed_int(r"(\d+) figures"), figures)
check("numbered equations", claimed_int(r"(\d+) numbered equations"), equations)
check("definitions", claimed_int(r"(\d+) definitions"), definitions)
check("printed references", claimed_int(r"\*\*(\d+) works\*\*"), printed_refs)
check("IP&M references cited", claimed_int(r"\*\*(\d+) of the 62"), ipm)
check("abstract words", claimed_int(r"250 words \(\*\*(\d+)\*\*\)"), abs_words)

t_ok = title in let
print(f"  [{'ok ' if t_ok else 'FAIL'}] title matches the manuscript      "
      f"{'yes' if t_ok else 'NO -- the letter names a different title'}")
if not t_ok:
    fails.append("title")
    print(f"         manuscript: {title}")

# --------------------------------------------------------------- (2) numbers
# Values the letter quotes ON PURPOSE because they are what we got WRONG. Each must be traceable
# to the control that measured it, not to a memory of what we used to print.
SUPERSEDED = {
    "0.651": "float32 + optimistic ties: the counter's old headline (dtype_control.json)",
    "0.6505": "float32 + optimistic ties (dtype_control.json)",
    "0.6211": "float32 + expected ties: superseded by float64 .6222 (dtype_control.json)",
    "0.5752": "GIRAM's best seed -- the cherry-pick we fixed (results_chrono_nyc_gru.csv)",
    "0.5531": "GETNext's own batch-averaged Acc@10 (the estimator we did NOT use)",
    "0.5544": "GETNext per-instance at one hash seed",
}

corpus = tex + "\n" + "\n".join(p.read_text(encoding="utf-8") for p in DOC.glob("tab_*.tex"))
print("\nNumbers in the letter that must also exist in the manuscript:\n")

missing = []
for tok in sorted(set(re.findall(r"\b\d\.\d{3,4}\b", let))):
    if tok in SUPERSEDED:
        continue
    dot = tok[1:]                       # 0.6222 -> .6222 (the manuscript's convention)
    stem = dot.rstrip("0")
    if (tok in corpus) or (dot in corpus) or re.search(re.escape(stem) + r"\d?\b", corpus):
        continue
    missing.append(tok)

if missing:
    for tok in missing:
        ctx = re.search(r"[^\n]{0,70}" + re.escape(tok) + r"[^\n]{0,40}", let)
        print(f"  [FAIL] {tok}  traces to NOTHING in the manuscript")
        print(f"         ...{' '.join(ctx.group(0).split())}")
        fails.append(f"number {tok}")
else:
    print("  [ok ] every number in the letter also appears in the manuscript, or is an")
    print("        explicitly-registered superseded value the letter is owning up to.")

print()
if fails:
    print(f"{len(fails)} MISMATCH(ES). The letter and the manuscript disagree. Fix before sending.")
    sys.exit(1)
print("The response letter is consistent with the manuscript it describes.")
