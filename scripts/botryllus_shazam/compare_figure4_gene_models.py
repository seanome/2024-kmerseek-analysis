"""Compare saved read-based gene models with published tables by exact gene ID."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))
import pubfig as pf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", type=Path)
    parser.add_argument("paper", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mapping = {
        "young_2h": "Group1_2Hr_young_DEseqres.tsv.gz",
        "young_24h": "group2_24Hr_Young_DEseqresult.tsv.gz",
    }
    summary, comparisons = [], []
    for contrast in ["young_2h", "young_24h", "young_312h"]:
        folder = args.models / contrast / "genes/result"
        assert (folder / "model.complete.txt").is_file()
        path = folder / "all-features.tsv.gz"
        model = pd.read_csv(path, sep="\t")
        assert model.tag.is_unique
        assert set(model.contrast) == {contrast} and set(model.assay) == {"genes"}
        valid = model.pvalue.notna()
        p = model.loc[valid, "pvalue"].to_numpy()
        order = np.argsort(p)
        adjusted = np.minimum.accumulate(
            (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
        )[::-1].clip(0, 1)
        np.testing.assert_allclose(
            model.loc[valid, "padj"].to_numpy()[order], adjusted, rtol=1e-10, atol=1e-12
        )
        record = dict(
            contrast=contrast,
            eligible_genes=len(model),
            finite_p=int(valid.sum()),
            significant_genes=int((model.padj < 0.05).sum()),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            bh_independently_verified=True,
        )
        if contrast in mapping:
            published = pd.read_csv(args.paper / mapping[contrast], sep="\t")
            assert published.Gene_ID.is_unique
            joined = model.merge(
                published,
                left_on="tag",
                right_on="Gene_ID",
                suffixes=("_reads", "_published"),
                validate="one_to_one",
            )
            finite = np.isfinite(joined.log2FoldChange_reads) & np.isfinite(
                joined.log2FoldChange_published
            )
            comparable = joined.loc[finite]
            nonzero = comparable[
                (comparable.log2FoldChange_reads != 0)
                & (comparable.log2FoldChange_published != 0)
            ]
            record.update(
                published_rows=len(published),
                published_significant=int((published.padj < 0.05).sum()),
                exact_id_matches=len(joined),
                finite_effect_pairs=len(comparable),
                published_ids_without_eligible_read_model=int(
                    (~published.Gene_ID.isin(model.tag)).sum()
                ),
                read_ids_absent_from_published=int(
                    (~model.tag.isin(published.Gene_ID)).sum()
                ),
                nonzero_effect_pairs=len(nonzero),
                same_effect_direction=int(
                    (
                        (nonzero.log2FoldChange_reads > 0)
                        == (nonzero.log2FoldChange_published > 0)
                    ).sum()
                ),
                spearman_rho=float(
                    spearmanr(
                        comparable.log2FoldChange_reads,
                        comparable.log2FoldChange_published,
                    ).statistic
                ),
            )
            joined.to_csv(
                args.output / f"{contrast}-exact-gene-comparison.tsv.gz",
                sep="\t",
                index=False,
            )
            comparisons.append((contrast, comparable, record))
        else:
            record["published_comparison"] = (
                "not matched: published group 4 combines 13 and 49 days; this model uses 312 hours only"
            )
        summary.append(record)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pf.use_style()
    fig, axes = pf.figure(pf.TWO_COLUMN_MM, 88, ncols=2)
    for ax, (contrast, table, record) in zip(axes, comparisons):
        ax.scatter(
            table.log2FoldChange_published,
            table.log2FoldChange_reads,
            s=2,
            alpha=0.25,
            color=pf.OKABE_ITO["blue"],
            edgecolors="none",
            rasterized=True,
        )
        ax.axhline(0, color=pf.GREY)
        ax.axvline(0, color=pf.GREY)
        ax.set_xlabel("Published treatment / control (log2 fold change)")
        ax.set_ylabel("Read-based treatment / control (log2 fold change)")
        ax.set_title(
            f"{contrast.replace('young_', '')}: {len(table):,} genes; rank correlation {record['spearman_rho']:.2f}"
        )
    fig.suptitle(
        "Each blue dot = one exact gene-ID match; grey lines = no change\nDifferent counts and statistical models: preliminary comparison"
    )
    pf.save(fig, args.output / "read-based-versus-published-gene-effects")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
