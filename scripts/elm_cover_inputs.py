#!/usr/bin/env python3
"""Stage the inputs of the ELM cover search (notebook 250) for qfo-pfam-region-benchmark.

Two modes.

--write-assets (Mac, once): from Stage 0's usable instances
(scripts/elm_stage0_flanks.py -> stage0/stage0_instances.parquet) write, into the pipeline's
assets/ (in git, so Sherlock gets them by `git pull`):
  elm_cover_query_accessions.txt   the 1_303 human proteins with a usable ELM instance
  elm_cover_instances.tsv          the 2_160 instances: class, accession, start, end
                                   (0-based, end-exclusive), motif length, length bin, regex

--stage DATA_DIR (Mac or Sherlock, in the pipeline directory): build the qfo-shaped input
folder the pipeline reads, from files already there:
  <out>/qfo/Eukaryota/UP000005640_9606.fasta   the query proteins, cut from the QfO human proteome
  <out>/qfo/<subdir>/<proteome>_<taxon>.fasta   links to the nine full QfO target proteomes
  <out>/annotations/human_pfam_domains.parquet  the human Pfam table cut to the queries
                                                (buildDomainTruth needs one)
  <out>/annotations/no_target_pfam_domains.parquet  empty table for a label never searched: the
                                                targets get no Pfam table, so the pipeline
                                                searches them and scores nothing
  <out>/structures/human/AF-<acc>-F1.cif        links to the query proteins' models
  <out>/structures/<species>                    links to the target species' model folders
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import polars as pl

PIPE = Path(__file__).resolve().parents[1] / "nextflow-runs" / "qfo-pfam-region-benchmark"
ASSETS = PIPE / "assets"
NINE = ["mouse", "chicken", "zebrafish", "ciona", "fly", "worm", "yeast", "arabidopsis", "ecoli"]


def write_assets(stage0: Path) -> None:
    inst = pl.read_parquet(stage0 / "stage0_instances.parquet")
    accs = sorted(inst["accession"].unique().to_list())
    (ASSETS / "elm_cover_query_accessions.txt").write_text("\n".join(accs) + "\n")
    inst.select("elm_instance", "elm_class", "accession", "start", "end", "motif_length", "length_bin",
                "in_midi_plus", "regex").sort("accession", "start", "elm_instance").write_csv(
        ASSETS / "elm_cover_instances.tsv", separator="\t")
    print(f"wrote {len(accs)} query accessions and {inst.height} instances to {ASSETS}")


def read_fasta(path: Path, keep: set[str]) -> list[tuple[str, str]]:
    out, header, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                out.append((header, "".join(buf)))
            header, buf = line[1:], []
        else:
            buf.append(line.strip())
    if header is not None:
        out.append((header, "".join(buf)))
    return [(h, s) for h, s in out if h.split()[0].split("|")[1] in keep]


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def stage(data: Path, out: Path) -> None:
    accs = set((ASSETS / "elm_cover_query_accessions.txt").read_text().split())
    reg = {r["label"]: r for r in csv.DictReader(open(ASSETS / "qfo_species.tsv"), delimiter="\t")}
    (out / "qfo" / "Eukaryota").mkdir(parents=True, exist_ok=True)

    human = read_fasta(data / "qfo" / "Eukaryota" / "UP000005640_9606.fasta", accs)
    with open(out / "qfo" / "Eukaryota" / "UP000005640_9606.fasta", "w") as fh:
        for h, s in human:
            fh.write(f">{h}\n{s}\n")
    missing = accs - {h.split()[0].split("|")[1] for h, _ in human}
    print(f"queries: {len(human)} of {len(accs)} accessions found in the QfO human proteome"
          + (f"; missing: {sorted(missing)[:10]}" if missing else ""))

    for sp in NINE:
        r = reg[sp]
        (out / "qfo" / r["subdir"]).mkdir(parents=True, exist_ok=True)
        name = f"{r['proteome']}_{r['taxon']}.fasta"
        link(data / "qfo" / r["subdir"] / name, out / "qfo" / r["subdir"] / name)

    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    hp = pl.read_parquet(data / "annotations" / "human_pfam_domains.parquet")
    hp.filter(pl.col("accession").is_in(list(accs))).write_parquet(ann / "human_pfam_domains.parquet")
    hp.clear().write_parquet(ann / "no_target_pfam_domains.parquet")

    sdir = out / "structures"
    (sdir / "human").mkdir(parents=True, exist_ok=True)
    n = 0
    for acc in accs:
        src = data / "structures" / "human" / f"AF-{acc}-F1.cif"
        if src.exists():
            link(src, sdir / "human" / src.name)
            n += 1
    for sp in NINE:
        src = data / "structures" / sp
        if src.exists():
            link(src, sdir / sp)
    print(f"structures: {n} of {len(accs)} query models; target folders: "
          f"{[sp for sp in NINE if (sdir / sp).exists()]}")
    print(f"staged {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-assets", action="store_true")
    ap.add_argument("--stage0", type=Path, default=Path("/Users/olga/data/elm-motif-transfer/stage0"))
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--data", type=Path, default=Path("data"), help="the pipeline's data/ folder")
    ap.add_argument("--out", type=Path, default=Path("data/elm-cover"))
    args = ap.parse_args()
    if args.write_assets:
        write_assets(args.stage0)
    if args.stage:
        stage(args.data, args.out)


if __name__ == "__main__":
    main()
