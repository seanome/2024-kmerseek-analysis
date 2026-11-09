"""Totals and the two figures for the expression-sample census, from the files in counts/.

Writes counts/census_summary.json and figures/geo_samples_per_year.{svg,png},
figures/expression_inventory.{svg,png}. Run from the repository root:

  python geo2kmers/census/make_figures.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "notebooks"))
import pubfig as pf  # noqa: E402

COUNTS = HERE / "counts"
FIGURES = HERE / "figures"

# Meta-datasets rebuilt from GEO data, removed from the ArrayExpress total
# (list from the ChatGPT census of 2026-11-08; each study's description says it pools GEO series).
AE_GEO_COMPENDIA = [
    "E-MTAB-3732", "E-MTAB-6703", "E-MTAB-6698", "E-MTAB-6699",
    "E-MTAB-6043", "E-MTAB-1788", "E-MTAB-62", "E-MTAB-27",
]
GEA_ARRAY_ASSAYS = 8428  # DDBJ GEA expression-array assays, ChatGPT census 2026-11-08 (all 770 studies read)

ARRAY_COLOUR = pf.OKABE_ITO["blue"]
RNASEQ_COLOUR = pf.OKABE_ITO["vermillion"]


def load(name):
    return json.loads((COUNTS / name).read_text())


def summary():
    platform = load("geo_platform.json")
    arrays = load("geo_array_series.json")
    rnaseq = load("geo_rnaseq_series.json")
    ae = load("arrayexpress_rna_array_not_geo.json")
    sra = json.loads((HERE / "sra_2026-11-06" / "sra_sample_counts.json").read_text())
    ae_compendia = sum(ae["per_study"][a] for a in AE_GEO_COMPENDIA)
    ae_kept = ae["non_geod_samples"] - ae_compendia
    array_total = platform["array_gsm_rna"] + ae_kept + GEA_ARRAY_ASSAYS
    sra_public = sra["strict_RNA_seq"]["public"]["sample_acc_distinct"]
    return {
        "geo_array_rna_gsm": platform["array_gsm_rna"],
        "geo_expression_array_series_unique_gsm": arrays["expression profiling by array"]["unique_gsm"],
        "geo_rnaseq_series_unique_gsm": rnaseq["expression profiling by high throughput sequencing"]["unique_gsm"],
        "arrayexpress_not_geo_studies": ae["non_geod_studies"],
        "arrayexpress_not_geo_samples": ae["non_geod_samples"],
        "arrayexpress_geo_compendia_samples": ae_compendia,
        "arrayexpress_kept_samples": ae_kept,
        "gea_array_assays": GEA_ARRAY_ASSAYS,
        "expression_array_total": array_total,
        "sra_rnaseq_public_samples": sra_public,
        "rnaseq_over_arrays": round(sra_public / array_total, 3),
        "arrays_added_fraction": round(array_total / sra_public, 3),
    }


def per_year_figure():
    arrays_all = load("geo_array_series.json")["expression profiling by array"]
    rnaseq_all = load("geo_rnaseq_series.json")["expression profiling by high throughput sequencing"]
    arrays, rnaseq = arrays_all["by_year"], rnaseq_all["by_year"]
    years = list(range(2001, 2027))
    for name, series in (("array", arrays_all), ("RNA-seq", rnaseq_all)):
        plotted = sum(series["by_year"].get(str(y), 0) for y in years)
        assert plotted == series["unique_gsm"], f"{name}: years {years[0]}-{years[-1]} hold {plotted:,} of {series['unique_gsm']:,} GSMs"
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 55)
    ax.plot(years, [arrays.get(str(y), 0) / 1e3 for y in years], color=ARRAY_COLOUR, marker="o",
            markersize=3, label=f"Expression microarray ({arrays_all['unique_gsm'] / 1e6:.2f} M samples)")
    ax.plot(years, [rnaseq.get(str(y), 0) / 1e3 for y in years], color=RNASEQ_COLOUR, marker="^",
            markersize=3, linestyle="--", label=f"RNA-seq ({rnaseq_all['unique_gsm'] / 1e6:.2f} M samples)")
    ax.set_xlabel("Year the series was released (2026 to 8 Nov)")
    ax.set_ylabel("New GEO samples (thousands)")
    ax.set_xticks([2002, 2006, 2010, 2014, 2018, 2022, 2026])
    ax.set_ylim(0, None)
    pf.shared_legend(fig, ncol=1)
    pf.save(fig, FIGURES / "geo_samples_per_year")


def inventory_figure(s):
    fig, ax = pf.figure(pf.ONE_COLUMN_MM, 40)
    parts = [
        ("GEO arrays", s["geo_array_rna_gsm"], ARRAY_COLOUR),
        ("ArrayExpress, not from GEO", s["arrayexpress_kept_samples"], pf.OKABE_ITO["sky_blue"]),
        ("DDBJ GEA", s["gea_array_assays"], pf.OKABE_ITO["bluish_green"]),
    ]
    left = 0
    for label, n, colour in parts:
        ax.barh(1, n / 1e6, left=left, color=colour, height=0.6, edgecolor="white", linewidth=0.5,
                label=f"{label} ({n:,})")
        left += n / 1e6
    ax.barh(0, s["sra_rnaseq_public_samples"] / 1e6, color=RNASEQ_COLOUR, height=0.6,
            label=f"SRA RNA-seq ({s['sra_rnaseq_public_samples']:,})")
    ax.text(left + 0.08, 1, f"{s['expression_array_total'] / 1e6:.2f} M", va="center")
    ax.text(s["sra_rnaseq_public_samples"] / 1e6 + 0.08, 0, f"{s['sra_rnaseq_public_samples'] / 1e6:.2f} M",
            va="center")
    ax.set_yticks([1, 0], ["Expression arrays", "RNA-seq"])
    ax.set_xlim(0, 7.2)
    ax.set_xlabel("Public samples (millions)")
    pf.shared_legend(fig, ncol=2)
    pf.save(fig, FIGURES / "expression_inventory")


if __name__ == "__main__":
    pf.use_style()
    s = summary()
    (COUNTS / "census_summary.json").write_text(json.dumps(s, indent=1) + "\n")
    print(json.dumps(s, indent=1))
    per_year_figure()
    inventory_figure(s)
