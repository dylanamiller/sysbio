#!/usr/bin/env python3
"""
pipeline.py -- streamed, resumable RNA-seq quantification driven by manifest.csv.

Designed for raw data on an external drive:
  * reads lane files straight from the drive (never copies the ~70GB inward)
  * NO concatenation step -- salmon takes each sample's lanes as multiple inputs
    and merges them on the fly, so no giant intermediate files are created
  * writes index + (optional) temp + results to your fast internal SSD
  * processes ONE sample at a time (kind to a spinning USB drive's read head)
  * resumable -- any sample that already has a quant.sf is skipped, so a
    disconnect/crash just means rerun and it continues where it stopped

Per sample:
  (optional) fastp per lane -> temp on SSD, deleted after use   [--trim]
  salmon quant  -- counts reads per transcript, merging all lanes in one call
Then once: multiqc aggregates every report.

Jargon:
  lane      one sequencing run of a sample; a sample has several, hence multiple files
  index     search structure salmon builds ONCE from the mouse transcriptome (GRCm39)
  trimming  removing adapters/low-quality bases; OPTIONAL here -- salmon's mapping
            tolerates leftover adapters well, so it is off by default

Tools (not pip packages):
    mamba create -n rnaseq -c bioconda -c conda-forge salmon fastp multiqc

Typical run -- index + outputs on the SSD, raw data on the drive:
    python pipeline.py --txome GRCm39_transcripts.fa.gz \
        --root /media/you/EXTDRIVE/alz \
        --outdir ~/rnaseq/results --index ~/rnaseq/salmon_index --threads 8
    # later / after a disconnect -- just rerun (index built, done samples skipped):
    python pipeline.py --root /media/you/EXTDRIVE/alz \
        --outdir ~/rnaseq/results --index ~/rnaseq/salmon_index --threads 8
"""

import argparse, csv, shutil, subprocess, sys
from pathlib import Path


def run(cmd, dry):
    print("  $", " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def read_manifest(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_index(txome_fasta, index_dir, threads, dry):
    """Build the salmon index once from a mouse transcriptome FASTA (GENCODE GRCm39)."""
    run(["salmon", "index", "-t", str(txome_fasta), "-i", str(index_dir),
         "-k", "31", "-p", str(threads)], dry)


def trim_lanes(sid, r1, r2, tmp_dir, threads, dry):
    """fastp each lane pair separately -> temp files on the SSD. Returns (t1s, t2s)."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t1s, t2s = [], []
    for i, (a, b) in enumerate(zip(r1, r2)):
        o1 = tmp_dir / f"{sid}_L{i}_R1.fastq.gz"
        o2 = tmp_dir / f"{sid}_L{i}_R2.fastq.gz"
        run(["fastp", "-i", a, "-I", b, "-o", str(o1), "-O", str(o2),
             "-j", str(tmp_dir / f"{sid}_L{i}.fastp.json"),
             "-h", str(tmp_dir / f"{sid}_L{i}.fastp.html"),
             "-w", str(min(threads, 16))], dry)
        t1s.append(str(o1)); t2s.append(str(o2))
    return t1s, t2s


def process(sample, index_dir, outdir, threads, trim, force, dry):
    sid = sample["sample"]
    r1 = sample["r1_files"].split(";")
    r2 = sample["r2_files"].split(";")

    quant = outdir / "quant"; quant.mkdir(parents=True, exist_ok=True)
    out = quant / sid

    # --- resumability: skip if already quantified -------------------------
    if (out / "quant.sf").exists() and not force:
        print(f"[{sid}] already done -> skipping")
        return

    print(f"[{sid}] {len(r1)} lane(s){' + trim' if trim else ''} -> quant")

    tmp_dir = outdir / "trim_tmp"
    if trim:
        in1, in2 = trim_lanes(sid, r1, r2, tmp_dir, threads, dry)
    else:
        in1, in2 = r1, r2  # feed raw lane files straight from the external drive

    # salmon merges multiple lane files in one call: -1 laneA laneB -2 laneA laneB
    run(["salmon", "quant", "-i", str(index_dir), "-l", "A",
         "-1", *in1, "-2", *in2,
         "-p", str(threads), "--validateMappings", "-o", str(out)], dry)

    # free the SSD temp immediately -- keeps footprint tiny
    if trim and not dry:
        for f in in1 + in2:
            Path(f).unlink(missing_ok=True)


def reroot(path_str, root):
    """Re-point a manifest path onto the drive's mount: drops the leading
    'alz/' component and rejoins under --root."""
    p = Path(path_str)
    return str(root / Path(*p.parts[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--root", type=Path,
                    help="external-drive alz/ path (use if manifest paths are relative)")
    ap.add_argument("--outdir", default="results", type=Path, help="on your internal SSD")
    ap.add_argument("--index", default="salmon_index", type=Path, help="on your internal SSD")
    ap.add_argument("--txome", help="transcriptome FASTA; if given, (re)build index first")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--trim", action="store_true", help="run fastp per lane (default: off)")
    ap.add_argument("--only", help="comma-separated sample IDs to run (default: all)")
    ap.add_argument("--force", action="store_true", help="re-run samples already done")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()

    samples = read_manifest(args.manifest)

    # if the manifest holds relative paths, re-root them onto the drive mount point
    if args.root:
        for s in samples:
            for col in ("r1_files", "r2_files"):
                s[col] = ";".join(reroot(p, args.root) for p in s[col].split(";"))

    if args.only:
        keep = set(args.only.split(","))
        samples = [s for s in samples if s["sample"] in keep]
    if not samples:
        sys.exit("no samples selected -- check --manifest / --only")

    if args.txome:
        build_index(args.txome, args.index, args.threads, args.dry_run)
    elif not args.index.exists() and not args.dry_run:
        sys.exit(f"index '{args.index}' not found -- pass --txome to build it first")

    for s in samples:  # sequential: one sample's reads off the drive at a time
        process(s, args.index, args.outdir, args.threads, args.trim, args.force, args.dry_run)

    tmp_dir = args.outdir / "trim_tmp"
    if args.trim and tmp_dir.exists() and not args.dry_run:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    run(["multiqc", str(args.outdir), "-o", str(args.outdir / "multiqc")], args.dry_run)
    print("\n[ok] done. per-sample counts are in <outdir>/quant/<sample>/quant.sf")
    print("     next step (differential expression) reads those with pydeseq2.")


if __name__ == "__main__":
    main()