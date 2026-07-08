
Here's the email in plain English, then how to actually work with the files.

## What they sent you

They ran **RNA-seq** (RNA sequencing — measures which genes are switched on and how strongly in each sample) on two mouse experiments. Both test **R36**, a fragment of the reelin protein, against **GFP** (green fluorescent protein — an inert marker used as the placebo/control).

**RSQ experiment** — normal mice, R36 injected straight into the brain at a *high* dose vs. GFP. Simple two-group comparison.

**ART experiment** — three groups:
- Normal untreated mice (baseline control)
- AD-model mice + GFP (control treatment)
- AD-model mice + R36 (reelin treatment, *lower* dose)

The "AD-model" mice are **APP/PS1 transgenics** (genetically engineered to grow **amyloid plaques**, one hallmark of Alzheimer's) that were *also* given a virus adding human **tau** (produces **tau tangles**, the second hallmark). Their lab calls this combo an "amyloid-enhanced tauopathy." So ART answers two questions: (1) what does the disease model change vs. normal mice, and (2) does R36 reverse any of it.

The **CSV** maps each mouse to its group and sex. The **aducanumab paper** (an anti-amyloid antibody drug) is just a published dataset Dave wants to compare against later — a separate step, ignore it for now.

## The one file gotcha

They sequenced **paired-end** (each RNA fragment read from both ends → two files per run, usually labeled `_R1` and `_R2`). Critically: each sample was **run multiple times** to hit their target depth (50M reads). So one sample = several R1/R2 pairs. You'll **concatenate the technical replicates per sample** (all R1s together, all R2s together) before or during alignment. The CSV + filenames are how you'll figure out which files belong to which mouse.

## The workflow

Standard RNA-seq pipeline, roughly:

1. **QC** — `FastQC` + `MultiQC` to check read quality
2. **Trim** adapters/low-quality bases — `fastp`
3. **Concatenate** the per-sample runs (the gotcha above)
4. **Align/quantify** against the mouse genome (GRCm39) — `STAR` (aligner) or `Salmon`/`kallisto` (faster, pseudo-alignment) to get a gene-count table
5. **Differential expression** — find genes that differ between groups

## Python vs Julia

Use **Python**. Steps 1–4 are standalone command-line tools (language-agnostic — you're just orchestrating them). For step 5, `pydeseq2` keeps you in Python. Julia offers no real advantage here; the bioinformatics ecosystem is overwhelmingly Python/R. (Note: the field-standard stats package is `DESeq2` in R — worth knowing exists, but `pydeseq2` is a faithful port if you'd rather not touch R.)

Want me to sketch the actual pipeline script for steps 1–4 next, or start with a quick script to inventory the fastq files and match them to the CSV?

------------------------------

Read both. Here's your experimental design at a glance.

**RSQ** — 8 samples, two groups, balanced by sex:
- **R36** (reelin treatment): RSQ01, 02, 05, 07
- **GFP** (control): RSQ03, 04, 06, 08

**ART** — 12 samples, three groups (4 each), also sex-balanced:
- **NonTg / None** (normal mice, baseline): ART03, 08, 13, 16
- **APP+PS1 / GFP** (disease + control): ART27, 39, 51, 87
- **APP+PS1 / R36** (disease + reelin treatment): ART14, 25, 46, 50

"NonTg" = non-transgenic (normal mouse). "APP+PS1" = the Alzheimer's model genotype. "Treatment" is which virus they got.

One thing worth confirming: your **3 ART folders are almost certainly the 3 sequencing runs**, not 3 groups. The lab said they re-ran the ART samples several times to hit read depth — so each folder likely holds one run's worth of R1/R2 files for all 12 samples. That's the concatenation step from before: for each sample you'll merge its R1s across all three folders (and R2s separately). The single RSQ folder suggests those hit depth in one run.

Want me to write a quick script that scans all four folders and verifies this — i.e. confirms each ART sample appears once per folder and each RSQ sample once total, and flags anything missing?

-----------------------------

Inventory works and correctly confirms folder = group. Now the pipeline script.Both tested and working. Copying to outputs.Two scripts, run in order. I tested both against a mock copy of your folder layout.

**`inventory.py`** — scans `alz/{rsq,art1,art2,art3}`, matches every `.fastq.gz` to your CSVs, and checks each sample has equal R1/R2 counts (paired-end must stay balanced). It prints a per-sample table, confirms each folder is one group, and writes **`manifest.csv`** (sample → group → its file lists) that feeds the next script.

```
python inventory.py --root alz
```

**`pipeline.py`** — reads that manifest and, per sample: concatenates the per-lane files → `fastp` (trims adapters + junk bases) → `salmon` (counts reads per transcript) → `multiqc` (one QC summary). Output lands in `results/quant/<sample>/quant.sf`.

```
python pipeline.py --txome <mouse_transcriptome.fa.gz> --threads 8   # first run builds the index
python pipeline.py --threads 8                                        # later runs reuse it
```

Two quick notes:

- **`salmon`, `fastp`, `multiqc` aren't pip packages** — they're standalone bioinformatics tools. Install via conda/mamba (command's in the script header). This is why Python-vs-Julia doesn't matter here: you're just orchestrating external tools.
- **You'll need a mouse transcriptome FASTA** (the reference list of all known mouse RNAs) for salmon to match reads against. GENCODE mouse, genome build **GRCm39**, the "transcripts" file. Grab that and pass it to `--txome` once.

Use `--dry-run` first to see every command without executing — good for sanity-checking paths before committing compute.

That gets you from raw reads to a count table. The next stage is **differential expression** (finding genes that differ between groups) with `pydeseq2` — want that when you're ready?

----------------------------------------

The core idea: **read raw data from the drive, write everything else to your internal SSD, and don't make duplicate copies.**

Concretely:

1. **Treat the external drive as a read-only source.** Point `--root` at its mounted path. Don't copy 70GB inward, and don't write outputs back to it.

2. **Put the salmon index, temp files, and results on your internal SSD** — that's where the fast, repeated I/O should live.

3. **Skip the concatenation step.** Right now the pipeline `cat`s each sample's lanes into a new file — a second full ~70GB copy for no benefit. salmon accepts multiple inputs directly (`-1 lane1_R1 lane2_R1 -2 lane1_R2 lane2_R2`) and merges them on the fly. This is the biggest single saving in both space and I/O.

4. **Process samples one at a time, not in parallel.** If it's a spinning USB drive, parallel reads make the head thrash and everything crawls. Give each salmon run many threads (that's compute), but only read one sample's files at a time. (If it's an external SSD, parallel is fine.)

5. **Make it resumable.** salmon's per-sample output is tiny — have the pipeline skip any sample that already has a `quant.sf`. If the drive disconnects mid-run, you just rerun and it picks up where it left off.

One hardware check: plug into a **USB 3+ port** (blue connector, or "SS" logo). USB 2 caps around 40 MB/s and will dominate your runtime.

Net result: your disk footprint is just the raw data (external, untouched) + the index (~1 GB) + tiny outputs. No giant intermediates.

This also drops the separate `fastp` trimming step — salmon's mapping tolerates leftover adapters well, so trimming is optional for quantification. If you do want it, we trim per-lane to the SSD and delete as we go.

Want me to rewire `pipeline.py` to this streamed, resumable, no-concat design?

--------------------------------

The heredoc didn't write — the file is still the old version. Let me recreate it cleanly.All confirmed working: no-concat lane merging, re-rooting onto the drive, optional trimming, and resume/skip. Copying out.Rewired and tested. Key changes from before:

- **No concatenation.** salmon now takes each sample's lanes as multiple inputs in one call (`-1 laneA laneB -2 laneA laneB`) and merges internally. No 70GB duplicate.
- **Reads from the drive, writes to SSD.** Point `--root` at the drive's mount; `--index` and `--outdir` go on your internal disk.
- **Resumable.** Any sample with an existing `quant.sf` is skipped. Drive disconnects → just rerun, it continues. `--force` overrides.
- **Sequential by default** (one sample's reads off the drive at a time), each salmon run using all `--threads`.
- **Trimming is optional** via `--trim` — off by default, and when on it writes per-lane temp to the SSD and deletes it right after each sample.

Run it like:

```
# first time (builds the index on your SSD, then processes everything):
python pipeline.py --txome GRCm39_transcripts.fa.gz \
    --root /media/you/EXTDRIVE/alz \
    --outdir ~/rnaseq/results --index ~/rnaseq/salmon_index --threads 8

# rerun anytime — index reused, finished samples skipped:
python pipeline.py --root /media/you/EXTDRIVE/alz \
    --outdir ~/rnaseq/results --index ~/rnaseq/salmon_index --threads 8
```

Add `--dry-run` to preview every command first (paths are easy to get wrong with a mounted drive). One reminder from before: confirm it's a **USB 3+** port, or the drive's read speed becomes your bottleneck.

The `--root` re-rooting assumes your `manifest.csv` paths start with `alz/` — which they will if you ran `inventory.py` from the folder containing `alz/`. If your manifest already holds absolute drive paths, just omit `--root`.

Ready to move to differential expression with `pydeseq2` whenever you are.