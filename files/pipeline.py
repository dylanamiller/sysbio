#!/usr/bin/env python3
"""
pipeline.py -- steps 1-4 of the RNA-seq workflow, driven by manifest.csv.

For each sample:
  1. concatenate its per-lane R1 files (and R2 files) into one pair
  2. fastp  -- trims sequencing adapters + low-quality bases, writes a QC report
  3. salmon -- quantifies expression: counts how many reads map to each transcript
Then once at the end:
  4. multiqc -- aggregates every fastp + salmon report into one summary page

Jargon:
  paired-end  each fragment is read from both ends -> R1 and R2 files kept in step
  adapter     short synthetic DNA the sequencer adds; must be trimmed off the reads
  index       a search structure salmon builds ONCE from the mouse transcriptome
              (the set of all known mouse RNA sequences) so it can map reads fast

These are NOT python packages -- install the tools with conda/mamba:
    mamba create -n rnaseq -c bioconda -c conda-forge salmon fastp multiqc
    conda activate rnaseq

Typical run (build the index the first time, then process everything):
    # download a mouse transcriptome FASTA first, e.g. GENCODE GRCm39 'transcripts'
    python pipeline.py --txome gencode.vM36.transcripts.fa.gz --threads 8
    # subsequent runs reuse the built index:
    python pipeline.py --threads 8
"""

import argparse, csv, subprocess, sys
from pathlib import Path


def run(cmd, dry):
    print("  $", " ".join(str(c) for c in cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def read_manifest(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_index(txome_fasta, index_dir, threads, dry):
    """Build the salmon index once from a mouse transcriptome FASTA.
    A decoy-aware index (transcriptome + genome) is more accurate; the basic
    version here is fine to start. GENCODE mouse = genome build GRCm39."""
    run(["salmon", "index", "-t", txome_fasta, "-i", str(index_dir),
         "-k", "31", "-p", str(threads)], dry)


def process(sample, index_dir, outdir, threads, dry):
    sid = sample["sample"]
    r1 = sample["r1_files"].split(";")
    r2 = sample["r2_files"].split(";")

    cat = outdir / "concat";        cat.mkdir(parents=True, exist_ok=True)
    trim = outdir / "trimmed";      trim.mkdir(parents=True, exist_ok=True)
    quant = outdir / "quant";       quant.mkdir(parents=True, exist_ok=True)
    qc = outdir / "fastp_reports";  qc.mkdir(parents=True, exist_ok=True)

    c1, c2 = cat / f"{sid}_R1.fastq.gz", cat / f"{sid}_R2.fastq.gz"
    t1, t2 = trim / f"{sid}_R1.fastq.gz", trim / f"{sid}_R2.fastq.gz"

    print(f"[{sid}] {len(r1)} lane-pair(s) -> concat -> trim -> quant")
    # 1. concatenate -- gzip files can be cat'd end-to-end and stay valid gzip
    run(["bash", "-c", "cat " + " ".join(r1) + f" > {c1}"], dry)
    run(["bash", "-c", "cat " + " ".join(r2) + f" > {c2}"], dry)
    # 2. trim + per-sample QC
    run(["fastp", "-i", str(c1), "-I", str(c2), "-o", str(t1), "-O", str(t2),
         "-j", str(qc / f"{sid}.fastp.json"),
         "-h", str(qc / f"{sid}.fastp.html"),
         "-w", str(min(threads, 16))], dry)
    # 3. quantify (-l A = auto-detect library type)
    run(["salmon", "quant", "-i", str(index_dir), "-l", "A",
         "-1", str(t1), "-2", str(t2), "-p", str(threads),
         "--validateMappings", "-o", str(quant / sid)], dry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--outdir", default="results", type=Path)
    ap.add_argument("--index", default="salmon_index", type=Path)
    ap.add_argument("--txome", help="transcriptome FASTA; if given, (re)build index first")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", help="comma-separated sample IDs to run (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()

    samples = read_manifest(args.manifest)
    if args.only:
        keep = set(args.only.split(","))
        samples = [s for s in samples if s["sample"] in keep]
    if not samples:
        sys.exit("no samples selected -- check --manifest / --only")

    if args.txome:
        build_index(args.txome, args.index, args.threads, args.dry_run)
    elif not args.index.exists() and not args.dry_run:
        sys.exit(f"index '{args.index}' not found -- pass --txome to build it first")

    for s in samples:
        process(s, args.index, args.outdir, args.threads, args.dry_run)

    # 4. aggregate all fastp + salmon reports into one page
    run(["multiqc", str(args.outdir), "-o", str(args.outdir / "multiqc")], args.dry_run)

    print("\n[ok] done. per-sample counts are in results/quant/<sample>/quant.sf")
    print("     next step (differential expression) reads those with pydeseq2.")


if __name__ == "__main__":
    main()
