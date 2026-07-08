#!/usr/bin/env python3
"""
inventory.py -- scan the fastq folders, match files to the sample sheets,
and write a manifest that pipeline.py consumes.

Expected layout:
    alz/
      rsq/  art1/  art2/  art3/
        RSQ01_L000_R1_001.fastq.gz   ->  SAMPLE _ LANE _ READ _ CHUNK . fastq.gz

Each sample was sequenced across several lanes/runs, so it has multiple R1/R2
files that later get concatenated. R1 and R2 are the two ends of the same
paired-end fragment, so every sample must have an EQUAL number of R1 and R2
files -- the script flags any mismatch.

Run:
    python inventory.py --root alz
"""

import argparse, csv, re, sys
from collections import defaultdict
from pathlib import Path

# SAMPLE _ LANE _ READ _ CHUNK . fastq.gz   e.g. RSQ01_L000_R1_001.fastq.gz
FASTQ_RE = re.compile(
    r"^(?P<sample>[A-Za-z]+\d+)_(?P<lane>L\d+)_(?P<read>R[12])_(?P<chunk>\d+)\.fastq\.gz$"
)


def load_samplesheet(path):
    """Return {sample_id: {column: value}} from a CSV with a 'Sample' column."""
    meta = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = row["Sample"].strip()
            meta[sid] = {k: (v or "").strip() for k, v in row.items()}
    return meta


def scan(root):
    """Walk alz/*/ -> {sample: {'R1': [...], 'R2': [...], 'folders': set}}."""
    found = defaultdict(lambda: {"R1": [], "R2": [], "folders": set()})
    unparsed = []
    for path in sorted(Path(root).rglob("*.fastq.gz")):
        m = FASTQ_RE.match(path.name)
        if not m:
            unparsed.append(path)
            continue
        s = found[m["sample"]]
        s[m["read"]].append(str(path))
        s["folders"].add(path.parent.name)
    return found, unparsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="alz", help="folder holding rsq/ art1/ art2/ art3/")
    ap.add_argument("--rsq-csv", default="RSQ_samples.csv")
    ap.add_argument("--art-csv", default="ART_samples.csv")
    ap.add_argument("--manifest", default="manifest.csv")
    args = ap.parse_args()

    meta = {**load_samplesheet(args.rsq_csv), **load_samplesheet(args.art_csv)}
    found, unparsed = scan(args.root)
    problems = []

    # --- per-sample report -------------------------------------------------
    print(f"{'sample':8} {'folder(s)':16} {'#R1':>4} {'#R2':>4}  paired?")
    print("-" * 46)
    for sid in sorted(set(meta) | set(found)):
        f = found.get(sid)
        if not f:
            print(f"{sid:8} {'-- MISSING --':16}")
            problems.append(f"{sid}: in sample sheet but no fastq files found")
            continue
        n1, n2 = len(f["R1"]), len(f["R2"])
        folders = ",".join(sorted(f["folders"]))
        ok = "yes" if n1 == n2 and n1 > 0 else "NO"
        print(f"{sid:8} {folders:16} {n1:>4} {n2:>4}  {ok}")
        if sid not in meta:
            problems.append(f"{sid}: fastq files found but not in any sample sheet")
        if n1 != n2:
            problems.append(f"{sid}: R1/R2 mismatch ({n1} vs {n2}) -- paired-end broken")
        if len(f["folders"]) > 1:
            problems.append(f"{sid}: spans multiple folders ({folders})")

    # --- per-folder group composition (confirms folder == group) -----------
    print("\nFolder composition (to confirm each folder == one group):")
    by_folder = defaultdict(list)
    for sid, f in found.items():
        for fld in f["folders"]:
            by_folder[fld].append(sid)
    for fld in sorted(by_folder):
        tags = []
        for sid in by_folder[fld]:
            m = meta.get(sid, {})
            tags.append(m.get("Group") or f"{m.get('Genotype','?')}/{m.get('Treatment','?')}")
        print(f"  {fld:6} n={len(by_folder[fld]):2}  groups: {', '.join(sorted(set(tags)))}")

    # --- unparsed files ----------------------------------------------------
    if unparsed:
        print(f"\n{len(unparsed)} file(s) did not match the expected name pattern:")
        for p in unparsed[:10]:
            print(f"  {p}")

    # --- write manifest ----------------------------------------------------
    with open(args.manifest, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sample", "experiment", "group", "sex", "r1_files", "r2_files"])
        for sid in sorted(found):
            if sid not in meta:
                continue
            m = meta[sid]
            exp = "RSQ" if sid.startswith("RSQ") else "ART"
            group = m.get("Group") or "_".join(
                x for x in (m.get("Genotype", ""), m.get("Treatment", "")) if x
            )
            w.writerow([
                sid, exp, group, m.get("Sex", ""),
                ";".join(sorted(found[sid]["R1"])),
                ";".join(sorted(found[sid]["R2"])),
            ])

    # --- verdict -----------------------------------------------------------
    print()
    if problems:
        print(f"[!] {len(problems)} issue(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"[ok] all samples matched and paired. manifest -> {args.manifest}")


if __name__ == "__main__":
    main()
