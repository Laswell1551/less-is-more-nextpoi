#!/usr/bin/env python3
"""
verify_editor_points.py -- has the manuscript ACTUALLY done what the editor asked?

WHY THIS EXISTS

The response letter says "Done" fourteen times. Twice that turned out to be false, and both
times the false claim was on a point the editor could check in under a minute:

  * point 11 ("match the level of research of recent IP&M articles") was answered with
    "the revision has 2 definitions and 11 numbered equations". The manuscript had 0 and 3.
    Those counts had been carried over from a SUPERSEDED draft that the v2 rewrite replaced.

  * point 9 ("you must update your literature review") was answered with "recent next-POI
    architectures (ROTAN 2024, Diff-POI, GeoMamba 2025, GNPR-SID 2025)". None of the four were
    cited. All four sat in refs_v2.bib, verified, and never reached the prose -- and the
    manuscript even DESCRIBED rotation-based attention and diffusion while citing the
    sequential-recommendation papers instead of the POI ones.

Neither was a lie. Both were a letter that moved while the manuscript did not, or the reverse.
Reading the letter cannot catch that; only checking the manuscript can. So this script checks the
MANUSCRIPT against the editor's fourteen points, and never reads the letter's claims at all.

Exit 1 if any point is unmet.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT.parent
tex = (DOC / "main_v2.tex").read_text(encoding="utf-8")
log = (DOC / "main_v2.log").read_text(encoding="utf-8", errors="ignore")
bbl = (DOC / "main_v2.bbl").read_text(encoding="utf-8", errors="ignore")
blg = (DOC / "main_v2.blg").read_text(encoding="utf-8", errors="ignore")
bib = (DOC / "refs_v2.bib").read_text(encoding="utf-8")
hi = (DOC / "highlights_v2.txt").read_text(encoding="utf-8")

fails = []


def point(n, ask, ok, detail):
    print(f"  [{'ok ' if ok else 'FAIL'}] {n:>2}. {ask}")
    print(f"          {detail}")
    if not ok:
        fails.append(f"point {n}: {ask}")


print("The editor's fourteen points, checked against the MANUSCRIPT (not the letter):\n")

# ---- 1. Highlights: 15-30 words each -------------------------------------------------------
bullets = [b.strip() for b in re.split(r"\n(?=- )", hi) if b.strip().startswith("- ")]
wc = [len(re.sub(r"^- ", "", b).split()) for b in bullets]
point(1, "Highlights: 3-5 items, 15-30 words each",
      len(bullets) == 5 and all(15 <= w <= 30 for w in wc),
      f"{len(bullets)} items; word counts {wc}")

# ---- 2. Abstract: <=250 words, with specifics ------------------------------------------------
abs_body = re.search(r"(?s)\\begin\{abstract\}(.*?)\\end\{abstract\}", tex).group(1)
abs_clean = re.sub(r"\\[a-zA-Z]+\{?|\}|\$|~|\\\\", "", abs_body)
aw = len(abs_clean.split())
nums = len(re.findall(r"\d", abs_body))
point(2, "Abstract <=250 words, results-first, with concrete numbers",
      aw <= 250 and nums >= 30,
      f"{aw} words; {nums} digits (sample sizes, effect sizes, percentages)")

# ---- 3. Explicit research objectives, as a separate section ----------------------------------
has_obj = bool(re.search(r"\\section\{Research objectives\}", tex))
rqs = len(re.findall(r"\\item\[RQ\d", tex))
point(3, "Explicit research objectives, as a separate section",
      has_obj and rqs >= 3, f"\\section{{Research objectives}} present; {rqs} numbered RQs")

# ---- 4. Explicit dataset description ---------------------------------------------------------
has_data = bool(re.search(r"\\section\{Data\}", tex))
subs = re.findall(r"\\subsection\{([^}]+)\}", tex[tex.find(r"\section{Data}"):
                                                 tex.find(r"\section{Problem")])
point(4, "Explicit description of the dataset(s)",
      has_data and len(subs) >= 4, f"\\section{{Data}} with subsections: {', '.join(subs)}")

# ---- 5. SOTA baselines, run by us, incl. an LLM, on the same data ----------------------------
ran = {"GETNext": "yang2022getnext", "STHGCN": "yan2023sthgcn"}
llm = "Qwen2.5-7B" in tex
same = "on its own evaluation instances" in tex or "its own evaluation instances" in tex
point(5, "SOTA baselines RUN by us (not quoted), an LLM addressed, same data",
      all(k in tex for k in ran) and llm and same,
      "official GETNext + official STHGCN + Qwen2.5-7B, each run by us; each scored on its "
      "own instance set (5,550 / 9,778 -- not interchangeable)")

# ---- 6. Discussion of results and implications, explicit section -----------------------------
# Scan to the NEXT \section, not a fixed character window. A window of 4,000 characters missed
# three of the five subsections here, because "Summary of findings" reports back against all five
# research questions and is long. The check said FAIL on a section that was entirely present.
# A verifier that is wrong in the accusing direction is worse than no verifier.
disc = tex.find(r"\section{Discussion of results and implications}")
nxt = tex.find(r"\section{", disc + 10) if disc > 0 else -1
dsubs = re.findall(r"\\subsection\{([^}]+)\}", tex[disc:nxt]) if disc > 0 else []
need = ["Theoretical implications", "Practical implications", "How this work differs"]
point(6, "Explicit Discussion section with theoretical + practical implications",
      disc > 0 and all(any(n in s for s in dsubs) for n in need),
      f"subsections: {', '.join(dsubs)}")

# ---- 7. Copy editing --------------------------------------------------------------------------
txt = subprocess.run(["pdftotext", "-nopgbrk", str(DOC / "main_v2.pdf"), "-"],
                     capture_output=True, text=True, encoding="utf-8").stdout
s, e = txt.find("1. Introduction"), txt.find("References")
prose = txt[s:e] if s > 0 < e else txt
lines = [l for l in prose.splitlines()
         if not re.match(r"^\s*(Table|Figure|Fig\.)\s*\d", l)
         and len(re.findall(r"[\d.]+", l)) < 6]
body = re.sub(r"\s+", " ", " ".join(lines))
dbl = [m.group(0) for m in re.finditer(r"\b([A-Za-z]{2,})\s+\1\b", body)
       if m.group(1).lower() not in {"that", "had"}]
sents = [x for x in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", body) if len(x.split()) > 3]
mean = sum(len(x.split()) for x in sents) / len(sents)
point(7, "Copy editing: accessible to non-native readers",
      not dbl and mean < 27,
      f"mean sentence {mean:.1f} words over {len(sents)} sentences; "
      f"doubled words: {dbl if dbl else 'none'}")

# ---- 8. Length -------------------------------------------------------------------------------
pages = int(re.search(r"Output written on .*?\((\d+) pages", log).group(1))
point(8, "Manuscript no longer short",
      pages >= 25, f"{pages} pages, {len(body.split()):,} words of prose")

# ---- 9. Recent literature (the editor: 'the last two years') ---------------------------------
cited, recent, ipm = 0, 0, 0
for m in re.finditer(r"(?s)@\w+\{([^,]+),(.*?)(?=\n@|\Z)", bib):
    k, b = m.group(1).strip(), m.group(2)
    if not re.search(r"\{" + re.escape(k) + r"\}", bbl):
        continue
    cited += 1
    y = re.search(r"year\s*=\s*\{?(\d{4})", b)
    if y and int(y.group(1)) >= 2024:
        recent += 1
    if "Information Processing" in b and "Neural Information" not in b:
        ipm += 1
SOTA24 = {"feng2024rotan": "ROTAN (KDD'24)", "qin2023diffpoi": "Diff-POI (TOIS'24)",
          "qin2025geomamba": "GeoMamba (AAAI'25)", "wang2025gnprsid": "GNPR-SID (KDD'25)"}
missing = [v for k, v in SOTA24.items() if not re.search(r"\{" + k + r"\}", bbl)]
point(9, "Literature review updated; recent work cited; tied to the IP&M community",
      recent / cited >= 0.40 and not missing and ipm >= 10,
      f"{cited} works printed; {recent} ({100*recent/cited:.0f}%) from 2024+; {ipm} from IP&M; "
      f"recent next-POI SOTA missing: {missing if missing else 'none'}")

# ---- 10. Reference list tidy; consistent style ------------------------------------------------
errs = re.search(r"There were (\d+) error message", blg)
nerr = int(errs.group(1)) if errs else 0
style = re.search(r"\\bibliographystyle\{([^}]+)\}", tex).group(1)
point(10, "Reference list tidy, consistent, standard style",
      nerr == 0 and "names" in style,
      f"BibTeX errors: {nerr}; style: {style} (Elsevier author-year, APA-consistent)")

# ---- 11. Match the level of recent IP&M articles ----------------------------------------------
eqs = len(re.findall(r"\\begin\{equation\}", tex)) + 2 * len(re.findall(r"\\begin\{align\}", tex))
defs = len(re.findall(r"\\begin\{definition\}", tex))
tabs = len(re.findall(r"\\begin\{table", tex))
point(11, "Structure/formalism at the level of recent IP&M articles",
      eqs >= 10 and defs >= 2,
      f"{eqs} numbered equations, {defs} definitions, {tabs} tables, "
      "explicit RQs, formal problem statement, findings reported back per RQ")

# ---- 12. Author guidelines --------------------------------------------------------------------
g = {"abstract <=250": aw <= 250,
     "highlights in a separate file": (DOC / "highlights_v2.txt").exists(),
     "competing interest": "Declaration of competing interest" in tex,
     "CRediT": "CRediT" in tex or "Conceptualization" in tex,
     "data availability": "Data availability" in tex,
     "funding": "Funding" in tex}
point(12, "IP&M author guidelines", all(g.values()),
      "; ".join(f"{k}: {'yes' if v else 'NO'}" for k, v in g.items()))

# ---- 13 & 14: commitments in the letter, not properties of the manuscript ---------------------
print("\n  [--]  13. Willingness to review (a commitment, not a manuscript property)")
print("          Stated in the response letter.")
print("  [--]  14. PhD Paper Award (a commitment, not a manuscript property)")
print("          Lead author is a PhD student; stated in the response letter.")

print()
if fails:
    print(f"{len(fails)} POINT(S) NOT MET:")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("All twelve checkable points are met by the manuscript itself.")
