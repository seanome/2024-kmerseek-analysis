"""Choose which sequences get protein searches (fixed before results).

Comparable review set, the same rule for both methods: the 500 top units by
each method's own ranking (SPLASH anchors: pval_opt_corrected ascending, then
effect_size_bin descending; DE-kupl contigs: adjusted p ascending, then
|log2FC| descending). For SPLASH every compactor of a chosen anchor is
searched.

Added on top, and flagged separately so they never enter the comparable
counts: every unit with no 31-mer in any oyster_v9 transcript (up to the
2,000 best ranked per method), every unit with a strict gene in the
published list, and every unit in an aligned DE-kupl/SPLASH pair.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

TOP, ABSENT_CAP = 500, 2000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--published", default=None)
    a = p.parse_args()
    w = Path(a.work)
    anchors = pd.read_csv(w / "splash_anchors_graded.tsv", sep="\t", dtype={"gene": str})
    contigs = pd.read_csv(w / "dekupl_contigs_graded.tsv", sep="\t", dtype={"gene": str})
    genes = pd.read_csv(w / "gene_table.tsv", sep="\t", dtype=str).fillna("")
    published = set(genes.loc[genes.published != "", "gene"])
    pairs_path = w / "aligned_pairs.tsv"
    pairs = pd.read_csv(pairs_path, sep="\t") if pairs_path.stat().st_size > 1 else pd.DataFrame(columns=["dekupl", "splash"])
    sp_seqs = pd.read_csv(w / "splash_sequences.tsv", sep="\t")
    paired_groups = set(sp_seqs.set_index("seq_id").loc[pairs["splash"], "group_id"]) if len(pairs) else set()

    def reasons(df, idcol, paired):
        r = pd.DataFrame(index=df.index)
        r["comparable_top500"] = df["rank"] <= TOP
        absent_rank = df["status"].eq("absent").cumsum().where(df["status"].eq("absent"))
        r["absent_from_reference"] = absent_rank.le(ABSENT_CAP).fillna(False)
        r["published_gene"] = df["gene"].isin(published) & df["status"].eq("strict")
        r["aligned_pair"] = df[idcol].isin(paired)
        return r

    anchors = anchors.sort_values("rank")
    contigs = contigs.sort_values("rank")
    ra = reasons(anchors, "group_id", paired_groups)
    rc = reasons(contigs, "contig_id", set(pairs["dekupl"]))
    sel_a = anchors[ra.any(axis=1)].join(ra)
    sel_c = contigs[rc.any(axis=1)].join(rc)
    sel_a.to_csv(w / "review_splash_units.tsv", sep="\t", index=False)
    sel_c.to_csv(w / "review_dekupl_units.tsv", sep="\t", index=False)
    sp_seqs[sp_seqs.group_id.isin(set(sel_a.group_id))].to_csv(w / "review_splash_sequences.tsv", sep="\t", index=False)
    sel_c.rename(columns={"contig_id": "seq_id", "contig": "dna"})[["seq_id", "dna"]].to_csv(
        w / "review_dekupl_sequences.tsv", sep="\t", index=False)
    summary = dict(rule=__doc__.strip(), splash_units=len(sel_a), dekupl_units=len(sel_c),
                   splash_reasons=ra.sum().astype(int).to_dict(), dekupl_reasons=rc.sum().astype(int).to_dict())
    (w / "review_selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
