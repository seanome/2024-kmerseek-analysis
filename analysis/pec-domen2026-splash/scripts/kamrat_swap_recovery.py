#!/usr/bin/env python3
"""The paper's lists reached by KaMRaT + DESeq2 with real and swapped labels (notebook 361).

Usage: kamrat_swap_recovery.py <pec_root> <manifest.tsv> <out_dir>

Runs paper_recovery.py on the real labels and on each swap, against the real contrast's list and
the same 1,000 random gene sets, by pointing it at a folder that holds one route's search. A swap
is scored as if its contrast were the real one; `reached_either_way` also counts list genes reached
only by contigs going the other way, since flipping a pair can turn a change around.

Writes to <out_dir>:
  kamrat_swap_recovery.tsv  one row per label set and list (paper_recovery.py's columns, plus run)
  kamrat_swap_kept.tsv      one row per label set: contigs kept, up, down, peptides, genes reached
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

root, manifest, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
SCRIPTS = Path(__file__).resolve().parent
ROUTE = "kamrat_deseq2_p05"
man = pl.read_csv(manifest, separator="\t")
id_map = root / "db/botznik.id_map.tsv"
bz = (pl.read_csv(id_map, separator="\t").filter(pl.col("original_id").str.contains(r"^g\d+\.t\d+$"))
      .select(pl.col("stable_id").alias("target"), pl.col("original_id").str.replace(r"\.t\d+$", "").alias("gene")))

runs = [(c, "real", root / "contrasts" / c) for c in man["contrast"].unique(maintain_order=True)]
runs += [(r["contrast"], r["flip"], root / "label_swap" / r["contrast"] / r["flip"]) for r in man.iter_rows(named=True)]

scores, kept = [], []
with tempfile.TemporaryDirectory(dir=out) as tmp:
    for contrast, run, d in runs:
        search = d / f"search_{ROUTE}"
        assert (search / "search.done").exists(), search
        view = Path(tmp) / f"{contrast}__{run}"
        (view / contrast).mkdir(parents=True)
        (view / contrast / f"search_{ROUTE}").symlink_to(search)
        tsv = view / "recovery.tsv"
        subprocess.run([sys.executable, str(SCRIPTS / "paper_recovery.py"), str(view), str(root / "paper/paper_deseq_all.parquet"),
                        str(root / "paper/paper_named_lists.tsv"), str(id_map), str(tsv)],
                       check=True, stdout=subprocess.DEVNULL)
        scores.append(pl.read_csv(tsv, separator="\t").filter(pl.col("channel") == ROUTE)
                      .with_columns(pl.lit(run).alias("run")))
        s = json.load(open(d / ROUTE / "summary.json"))
        genes = (pl.read_parquet(search / "align.botznik.parquet", columns=["target", "supported"]).filter(pl.col("supported"))
                 .join(bz, on="target")["gene"].n_unique())
        peptides = pl.read_csv(search / "pep.peptides.tsv", separator="\t", columns=["peptide_id"]).height
        kept.append(dict(contrast=contrast, run=run, contigs_fitted=s["contigs_fitted"], contigs_kept=s["p_lt_0.05"],
                         up=s["up"], down=s["down"], corrected_p_le_0_05=s["q_bh_le_0.05"], peptides=peptides, genes_reached=genes))
        print(kept[-1], flush=True)
pl.concat(scores).write_csv(out / "kamrat_swap_recovery.tsv", separator="\t")
pl.DataFrame(kept).write_csv(out / "kamrat_swap_kept.tsv", separator="\t")
print("wrote", out)
