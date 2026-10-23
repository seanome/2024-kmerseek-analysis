from pathlib import Path
import polars as pl
import pubfig as pf

def run_status_figure(data_dir, output):
    rows = []
    for run in ("exact", "extend", "decoy"):
        files = sorted((Path(data_dir) / run / "kmerseek").glob("human_vs_*.regions.parquet"))
        extended = 0
        for path in files:
            x = pl.scan_parquet(path).select(((pl.col("region_evalue").is_finite()) | (pl.col("region_n_mismatches") > 0)).any()).collect().item()
            extended += bool(x)
        rows.append((run, len(files), extended))
    fig, ax = pf.figure(width_mm=89, height_mm=65)
    labels = [r[0] for r in rows]
    ax.barh(labels, [r[2] for r in rows], color="#0072B2", label="Extended regions present")
    ax.barh(labels, [r[1]-r[2] for r in rows], left=[r[2] for r in rows], color="#999999", label="Exact regions only")
    ax.set_xlabel("Input tables (count)")
    ax.set_xlim(0, 70)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False)
    for i, (_, total, ext) in enumerate(rows):
        ax.text(total+1, i, str(total), va="center")
    fig.tight_layout()
    pf.save(fig, Path(output).with_suffix(""))
    print("Input tables, extended:", rows)
    return fig
