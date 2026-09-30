"""Human protein feature classes for BHF's matches (notebook 258).

For each canonical human protein (GENCODE v49, the database notebooks 241 and 256
searched) this builds residue intervals of four classes from the reviewed human UniProt
entries (UniProtKB/Swiss-Prot, organism 9606, downloaded 2025-06-04):

  zinc finger    `zinc finger region` features, and `domain` features whose name says
                 zinc finger
  membrane       `transmembrane region`, `intramembrane region`, `region of interest`
                 whose name mentions membrane, and each `lipid moiety-binding region`
                 (a lipid anchor such as myristoylation or palmitoylation) widened by
                 5 residues on each side
  extracellular  `topological domain` Extracellular; for a protein whose subcellular
                 location is Secreted and that has no transmembrane region, the whole
                 chain after the signal peptide
  disordered     `region of interest` Disordered (MobiDB-lite, as UniProt reports it)

A GENCODE protein takes the features of the UniProt entry with the identical sequence.
Proteins with no identical reviewed entry get no features and are counted, not guessed.

UniProt positions are 1-based and inclusive; the table stores 0-based, end-exclusive
intervals, the convention of kmerseek's target_start/target_end.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

UNIPROT_XML = Path("/Users/olga/data/uniprot/uniprotkb_organism_name_Human_AND_revie_2025_06_04.xml.gz")
GENCODE = Path("/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa")
CACHE = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases/human_feature_classes.parquet")
MATCHED = CACHE.with_name("human_feature_classes.matched_proteins.parquet")
CLASSES = ["zinc finger", "membrane", "extracellular", "disordered"]
LIPID_PAD = 5
NS = "{http://uniprot.org/uniprot}"


def _pos(loc):
    """(begin, end) 1-based inclusive from a UniProt <location>, or None when unknown."""
    b, e, p = loc.find(NS + "begin"), loc.find(NS + "end"), loc.find(NS + "position")
    try:
        if p is not None:
            v = int(p.get("position"))
            return v, v
        return int(b.get("position")), int(e.get("position"))
    except (TypeError, ValueError, AttributeError):
        return None


def parse_uniprot() -> pl.DataFrame:
    """One row per (accession, class, start, end), 0-based end-exclusive, plus sequence."""
    from lxml import etree
    rows, seqs = [], []
    for _, e in etree.iterparse(gzip.open(UNIPROT_XML), tag=NS + "entry"):
        tax = e.find(f"{NS}organism/{NS}dbReference[@type='NCBI Taxonomy']")
        if tax is None or tax.get("id") != "9606":
            e.clear()
            continue
        acc = e.findtext(NS + "accession")
        seq = "".join((e.findtext(NS + "sequence") or "").split())
        gene = e.find(f"{NS}gene/{NS}name[@type='primary']")
        seqs.append(dict(accession=acc, gene_uniprot=gene.text if gene is not None else None, sequence=seq))
        L = len(seq)
        locs = [x.text or "" for x in e.iter(NS + "subcellularLocation") for x in x.iter(NS + "location")]
        has_tm, signal_end = False, 0
        for ft in e.iter(NS + "feature"):
            t, d = ft.get("type"), (ft.get("description") or "")
            loc = ft.find(NS + "location")
            pos = _pos(loc) if loc is not None else None
            if pos is None:
                continue
            b, en = pos
            cls = None
            if t == "zinc finger region" or (t == "domain" and "zinc finger" in d.lower()):
                cls = "zinc finger"
            elif t in ("transmembrane region", "intramembrane region"):
                cls, has_tm = "membrane", has_tm or t == "transmembrane region"
            elif t == "region of interest" and "membrane" in d.lower():
                cls = "membrane"
            elif t == "lipid moiety-binding region":
                cls, b, en = "membrane", max(1, b - LIPID_PAD), min(L, en + LIPID_PAD)
            elif t == "topological domain" and d == "Extracellular":
                cls = "extracellular"
            elif t == "region of interest" and d == "Disordered":
                cls = "disordered"
            elif t == "signal peptide":
                signal_end = en
            if cls:
                rows.append(dict(accession=acc, cls=cls, start=b - 1, end=en, feature=t, description=d))
        if any(l.strip() == "Secreted" for l in locs) and not has_tm:
            rows.append(dict(accession=acc, cls="extracellular", start=signal_end, end=L,
                             feature="subcellular location", description="Secreted, no transmembrane region"))
        e.clear()
    return pl.DataFrame(rows), pl.DataFrame(seqs)


def gencode_sequences() -> pl.DataFrame:
    out, name, buf = [], None, []
    for line in open(GENCODE):
        if line.startswith(">"):
            if name:
                out.append((name, "".join(buf)))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    out.append((name, "".join(buf)))
    return pl.DataFrame(out, schema=["target_name", "sequence"], orient="row").with_columns(
        pl.col("target_name").str.split("|").list.get(6).alias("gene"))


def feature_table(rebuild: bool = False) -> tuple[pl.DataFrame, dict]:
    """Feature intervals keyed by GENCODE gene symbol, and counts of how many GENCODE
    proteins matched a reviewed UniProt entry by identical sequence."""
    if CACHE.exists() and not rebuild:
        t = pl.read_parquet(CACHE)
        return t, dict(t.select("n_gencode", "n_matched").row(0, named=True)) if "n_gencode" in t.columns else {}
    feats, seqs = parse_uniprot()
    g = gencode_sequences()
    m = g.join(seqs, on="sequence", how="inner").unique("target_name", keep="first", maintain_order=True)
    t = m.select("target_name", "gene", "accession").join(feats, on="accession", how="inner")
    stats = {"n_gencode": g.height, "n_matched": m.height}
    t = t.with_columns(pl.lit(stats["n_gencode"]).alias("n_gencode"), pl.lit(stats["n_matched"]).alias("n_matched"))
    t.write_parquet(CACHE)
    m.select("target_name", "gene", "accession").write_parquet(MATCHED)
    return t, stats


def matched_proteins() -> pl.DataFrame:
    """The GENCODE proteins with an identical reviewed UniProt entry (feature_table writes it)."""
    if not MATCHED.exists():
        feature_table(rebuild=True)
    return pl.read_parquet(MATCHED)


def merged_features(feats: pl.DataFrame) -> pl.DataFrame:
    """Feature intervals merged per (protein, class), so overlapping or touching
    features of one class count each residue once."""
    f = feats.select("target_name", "cls", "start", "end").sort("target_name", "cls", "start")
    f = f.with_columns(pl.col("end").cum_max().shift(1).over("target_name", "cls").alias("prev_end"))
    f = f.with_columns((pl.col("prev_end").is_null() | (pl.col("start") > pl.col("prev_end"))).cast(pl.Int32)
                         .cum_sum().over("target_name", "cls").alias("block"))
    return f.group_by("target_name", "cls", "block").agg(pl.col("start").min(), pl.col("end").max()).drop("block")


def label_matches(matches: pl.DataFrame, feats: pl.DataFrame, min_frac: float = 0.5) -> pl.DataFrame:
    """For each match (target_name, target_start, target_end; 0-based end-exclusive), the
    share of the human region inside each class (frac_<class>), and on_<class>: at least
    `min_frac` of the human region lies inside that class. `annotated` says whether the
    human protein has an identical reviewed UniProt entry at all; shares are 0 for a
    protein without one, so compare classes on annotated rows only."""
    mf = merged_features(feats)
    m = matches.with_row_index("match_id")
    ov = (m.select("match_id", "target_name", "target_start", "target_end")
           .join(mf, on="target_name", how="inner")
           .with_columns((pl.min_horizontal("target_end", "end") - pl.max_horizontal("target_start", "start"))
                         .clip(lower_bound=0).alias("ov"))
           .group_by("match_id", "cls").agg(pl.col("ov").sum()))
    wide = ov.pivot(on="cls", index="match_id", values="ov")
    for c in CLASSES:
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(c))
    length = (pl.col("target_end") - pl.col("target_start")).clip(lower_bound=1)
    out = (m.join(wide, on="match_id", how="left")
            .with_columns([(pl.col(c).fill_null(0) / length).alias(f"frac_{c.replace(' ', '_')}") for c in CLASSES])
            .drop(CLASSES))
    out = out.with_columns([(pl.col(f"frac_{c.replace(' ', '_')}") >= min_frac).alias(f"on_{c.replace(' ', '_')}")
                            for c in CLASSES])
    annotated = matched_proteins().select("target_name").with_columns(pl.lit(True).alias("annotated"))
    return (out.join(annotated, on="target_name", how="left")
               .with_columns(pl.col("annotated").fill_null(False)).drop("match_id"))


# ---------------------------------------------------------------- BHF against its copies
REGIONS_RUN = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases/null_bhf_dipeptide_regions")
MAIN = ["zinc finger", "membrane", "extracellular"]


def load_regions(run: Path = REGIONS_RUN) -> pl.DataFrame:
    """Top-10 regions of BHF and its 300 copies (256_bhf_dipeptide_shuffle_null.py
    --keep-regions), one row per (query, alphabet-ksize pair, metric, human protein)."""
    return pl.read_parquet(sorted((run / "regions").glob("*.parquet"))).with_columns(
        (pl.col("query_name") == "BHF").alias("is_bhf"))


def class_shares(lab: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """Per query (and `by`): among top-10 matches in a human protein with a UniProt
    entry, the share on each class, and on each class outside disordered regions
    (frac_disordered < 0.5)."""
    by = by or []
    a = lab.filter(pl.col("annotated"))
    aggs = [pl.len().alias("n_matches")]
    for c in MAIN:
        k = c.replace(" ", "_")
        aggs += [(100 * pl.col(f"on_{k}").mean()).alias(f"pct_{k}"),
                 (100 * (pl.col(f"on_{k}") & ~pl.col("on_disordered")).mean()).alias(f"pct_{k}_ordered")]
    aggs.append((100 * pl.col("on_disordered").mean()).alias("pct_disordered"))
    return a.group_by(["query_name", "is_bhf"] + by).agg(aggs)


def class_test(shares: pl.DataFrame, by: list[str] | None = None) -> pl.DataFrame:
    """BHF's share against the 300 copies' shares, per class (and `by`): p is
    (1 + copies with a share at least BHF's) / (1 + copies)."""
    by = by or []
    cols = [c for c in shares.columns if c.startswith("pct_")]
    long = shares.unpivot(index=["query_name", "is_bhf"] + by, on=cols, variable_name="measure", value_name="pct")
    b = long.filter(pl.col("is_bhf")).select(by + ["measure", pl.col("pct").alias("bhf_pct")])
    c = long.filter(~pl.col("is_bhf")).join(b, on=by + ["measure"], how="inner")
    return (c.group_by(by + ["measure"]).agg(
                pl.col("bhf_pct").first().round(2),
                pl.col("pct").median().round(2).alias("copies_median_pct"),
                pl.col("pct").quantile(0.025).round(2).alias("copies_p2_5"),
                pl.col("pct").quantile(0.975).round(2).alias("copies_p97_5"),
                pl.len().alias("n_copies"),
                ((1 + (pl.col("pct") >= pl.col("bhf_pct")).sum()) / (1 + pl.len())).round(4).alias("p"))
              .sort(by + ["measure"]))


def bhf_class_matches(lab: pl.DataFrame, cls: str, ordered_only: bool = True) -> pl.DataFrame:
    """BHF's top-10 matches on one class, collapsed per (human protein, BHF stretch):
    how many (alphabet-ksize pair, metric) cells found it, and the residues on both sides."""
    k = cls.replace(" ", "_")
    s = lab.filter(pl.col("is_bhf") & pl.col(f"on_{k}"))
    if ordered_only:
        s = s.filter(~pl.col("on_disordered"))
    return (s.group_by("gene", "target_name", "region_start", "region_end", "target_start", "target_end")
             .agg(pl.len().alias("n_cells"), pl.col("alphabet").unique().sort().alias("alphabets"),
                  pl.col("metric").unique().sort().alias("metrics"))
             .sort("n_cells", "gene", descending=[True, False]))


def with_residues(t: pl.DataFrame, bhf: str) -> pl.DataFrame:
    seqs = dict(gencode_sequences().select("target_name", "sequence").iter_rows())
    return t.with_columns(
        pl.struct("region_start", "region_end").map_elements(lambda r: bhf[r["region_start"]:r["region_end"]],
                                                             return_dtype=pl.Utf8).alias("bhf_residues"),
        pl.struct("target_name", "target_start", "target_end").map_elements(
            lambda r: seqs[r["target_name"]][r["target_start"]:r["target_end"]], return_dtype=pl.Utf8).alias("human_residues"))


# ---------------------------------------------------------------- figures
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from bhf_shuffle_null_utils import PURPLE, GREY, _figure_with_legend_row, finish_figure  # noqa: E402

TOOLS = ("kmerseek 0.4.0 (982a055) search, 256_bhf_dipeptide_shuffle_null.py --keep-regions: BHF and 300 "
         "dipeptide-shuffled copies against the 19_732 GENCODE v49 canonical human proteins, 152 alphabet-ksize "
         "pairs; human features from reviewed UniProt human entries (2025-06-04)")
MEASURE_LABEL = {
    "pct_zinc_finger_ordered": "zinc finger", "pct_membrane_ordered": "membrane",
    "pct_extracellular_ordered": "extracellular", "pct_disordered": "disordered (MobiDB-lite)",
}


def fig_class_shares(shares: pl.DataFrame, test: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """One row per class: % of top-10 matches that land on it (outside disordered
    regions, except the disordered row), BHF (diamond) against 300 copies (grey dots,
    middle 95% as a pale band drawn first)."""
    handles = [
        Patch(color="#e4e4e4", label="middle 95% of the 300 shuffled copies"),
        Line2D([], [], marker="o", ls="", ms=4, color=GREY, alpha=0.6,
               label="one shuffled copy of BHF (same length, residue counts and adjacent-pair counts)"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="BHF; p on the right = share of copies at least as high"),
    ]
    rows = list(MEASURE_LABEL)
    fig, (ax,) = _figure_with_legend_row(1, (10, 0.7 * len(rows) + 2.3), handles)
    rng = np.random.default_rng(0)
    for y, m in enumerate(rows):
        r = test.filter(pl.col("measure") == m).row(0, named=True)
        ax.barh(y, r["copies_p97_5"] - r["copies_p2_5"], left=r["copies_p2_5"], height=0.7, color="#e4e4e4", zorder=1)
        nul = shares.filter(~pl.col("is_bhf"))[m].to_numpy()
        ax.scatter(nul, y + rng.uniform(-0.25, 0.25, len(nul)), s=6, color=GREY, alpha=0.5, lw=0, zorder=2)
        ax.scatter([r["bhf_pct"]], [y], marker="D", s=55, color=PURPLE, zorder=4)
        ax.text(1.01, y, f"p = {r['p']:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=8, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([MEASURE_LABEL[m] for m in rows], fontsize=9)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(left=0)
    ax.set_xlabel("% of the query's top-10 matches whose human region lies at least half inside that class\n"
                  "(zinc finger, membrane, extracellular: outside disordered regions)")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.02, tight=False)


CLASS_COLOR = {"zinc finger": "#1b7837", "membrane": "#b35900", "extracellular": "#2166ac"}


def fig_where_on_bhf(lab: pl.DataFrame, bhf: str, path: Path, hypothesis: str, conclusion: str) -> pl.DataFrame:
    """BHF as a line (1-252). Below it, one panel per class: for every BHF residue, the
    number of BHF top-10 matches (alphabet-ksize pair x metric x human protein) on that
    class, outside disordered regions, whose BHF stretch covers the residue."""
    L = len(bhf)
    x = np.arange(1, L + 1)
    b = lab.filter(pl.col("is_bhf") & pl.col("annotated") & ~pl.col("on_disordered"))
    handles = [Line2D([], [], color=CLASS_COLOR[c], lw=2, label=f"BHF top-10 matches on a human {c} region")
               for c in MAIN] + [Line2D([], [], color="#333333", lw=2, label="BHF, residues 1-252")]
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(len(MAIN) + 1, 1, height_ratios=[0.32] + [1] * len(MAIN), hspace=0.3,
                          top=0.97, bottom=0.08, left=0.08, right=0.98)
    lax = fig.add_subplot(gs[0]); lax.axis("off")
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9, ncol=2)
    rows = []
    for i, c in enumerate(MAIN):
        ax = fig.add_subplot(gs[i + 1])
        cov = np.zeros(L)
        s = b.filter(pl.col(f"on_{c.replace(' ', '_')}"))
        for a_, e_ in zip(s["region_start"], s["region_end"]):
            cov[a_:e_] += 1
        ax.fill_between(x, cov, step="mid", color=CLASS_COLOR[c], alpha=0.85, lw=0)
        top = max(cov.max(), 1)
        ax.plot([1, L], [-0.12 * top] * 2, color="#333333", lw=2, solid_capstyle="butt", clip_on=False)
        ax.set_ylim(-0.2 * top, top * 1.1)
        ax.set_xlim(0, L + 1)
        ax.set_ylabel(f"{c}\n(matches)", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if i < len(MAIN) - 1:
            ax.set_xticklabels([])
        rows.append(pl.DataFrame({"residue": x, "cls": c, "n_matches": cov.astype(int)}))
    ax.set_xlabel("BHF residue")
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=1.01, footer_y=-0.03, tight=False)
    return pl.concat(rows)


# ---------------------------------------------------------------- BHF's own sequence
KYTE_DOOLITTLE = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2, I=4.5, L=3.8,
                      K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9, Y=-1.3, V=4.2)
EISENBERG = dict(A=0.62, R=-2.53, N=-0.78, D=-0.90, C=0.29, Q=-0.85, E=-0.74, G=0.48, H=-0.40, I=1.38,
                 L=1.06, K=-1.50, M=0.64, F=1.19, P=0.12, S=-0.18, T=-0.05, W=0.81, Y=0.26, V=1.08)
TM_WINDOW, TM_CUTOFF = 19, 1.6      # Kyte & Doolittle 1982: a 19-residue average above 1.6 suggests a membrane-spanning helix
HELIX_WINDOW, HELIX_ANGLE = 18, 100  # Eisenberg 1984: hydrophobic moment of an 18-residue alpha helix


def hydrophobic_moment(s: str) -> float:
    import math
    a = math.radians(HELIX_ANGLE)
    x = sum(EISENBERG[c] * math.cos(i * a) for i, c in enumerate(s))
    y = sum(EISENBERG[c] * math.sin(i * a) for i, c in enumerate(s))
    return math.hypot(x, y) / len(s)


def helix_profiles(seq: str) -> pl.DataFrame:
    """Per window start (1-based): Kyte-Doolittle average over 19 residues and the
    Eisenberg hydrophobic moment over 18 residues."""
    n = len(seq)
    kd = [sum(KYTE_DOOLITTLE[c] for c in seq[i:i + TM_WINDOW]) / TM_WINDOW for i in range(n - TM_WINDOW + 1)]
    mh = [hydrophobic_moment(seq[i:i + HELIX_WINDOW]) for i in range(n - HELIX_WINDOW + 1)]
    return pl.DataFrame({"start": range(1, len(kd) + 1), "kyte_doolittle": kd}).join(
        pl.DataFrame({"start": range(1, len(mh) + 1), "hydrophobic_moment": mh}), on="start", how="full", coalesce=True
    ).sort("start")


def helix_null(seq: str, n: int = 1_000, seed: int = 0) -> pl.DataFrame:
    """Max Kyte-Doolittle window and max hydrophobic moment of `n` residue-shuffled copies."""
    import random
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = list(seq); rng.shuffle(s); s = "".join(s)
        p = helix_profiles(s)
        rows.append((p["kyte_doolittle"].max(), p["hydrophobic_moment"].max()))
    return pl.DataFrame(rows, schema=["max_kyte_doolittle", "max_hydrophobic_moment"], orient="row")


def fig_helix_profiles(seq: str, prof: pl.DataFrame, null: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """Two panels along BHF: the 19-residue Kyte-Doolittle average with the 1.6 cutoff,
    and the 18-residue hydrophobic moment with the 95th percentile of the shuffled
    copies' maxima. Each value is drawn at the middle residue of its window."""
    L = len(seq)
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.3, 1, 1], hspace=0.35, top=0.97, bottom=0.1, left=0.08, right=0.98)
    handles = [Line2D([], [], color=PURPLE, lw=2, label="BHF"),
               Line2D([], [], color="#555555", ls="--", lw=1.2, label="Kyte-Doolittle cutoff for a membrane-spanning helix (1.6)"),
               Line2D([], [], color=GREY, ls=":", lw=1.5, label="95th percentile of the highest hydrophobic moment in 1_000 residue-shuffled BHF copies")]
    lax = fig.add_subplot(gs[0]); lax.axis("off")
    lax.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)
    kd = prof.drop_nulls("kyte_doolittle")
    mh = prof.drop_nulls("hydrophobic_moment")
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(kd["start"] + TM_WINDOW // 2, kd["kyte_doolittle"], color=PURPLE, lw=2)
    ax1.axhline(TM_CUTOFF, color="#555555", ls="--", lw=1.2)
    ax1.set_ylabel(f"Kyte-Doolittle,\n{TM_WINDOW}-residue average", fontsize=9)
    ax1.set_xlim(0, L + 1); ax1.set_xticklabels([])
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(mh["start"] + HELIX_WINDOW // 2, mh["hydrophobic_moment"], color=PURPLE, lw=2)
    ax2.axhline(float(null["max_hydrophobic_moment"].quantile(0.95)), color=GREY, ls=":", lw=1.5)
    ax2.set_ylabel(f"hydrophobic moment,\n{HELIX_WINDOW}-residue helix", fontsize=9)
    ax2.set_xlim(0, L + 1); ax2.set_xlabel("BHF residue (middle of the window)")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    finish_figure(fig, path, tools="no kmerseek: BHF sequence only; Kyte-Doolittle and Eisenberg hydrophobicity scales",
                  hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.03, tight=False)


def fig_membrane_by_alphabet(shares: pl.DataFrame, test: pl.DataFrame, path: Path, hypothesis: str, conclusion: str):
    """Like fig_class_shares, one row per alphabet, for the membrane class only."""
    m = "pct_membrane_ordered"
    handles = [
        Patch(color="#e4e4e4", label="middle 95% of the 300 shuffled copies"),
        Line2D([], [], marker="o", ls="", ms=4, color=GREY, alpha=0.6, label="one shuffled copy of BHF"),
        Line2D([], [], marker="D", ls="", ms=8, color=PURPLE, label="BHF; p on the right = share of copies at least as high"),
    ]
    t = test.filter(pl.col("measure") == m).sort("p")
    alphas = t["alphabet"].to_list()
    fig, (ax,) = _figure_with_legend_row(1, (10, 0.36 * len(alphas) + 2.3), handles)
    rng = np.random.default_rng(0)
    for y, a in enumerate(alphas):
        r = t.filter(pl.col("alphabet") == a).row(0, named=True)
        ax.barh(y, r["copies_p97_5"] - r["copies_p2_5"], left=r["copies_p2_5"], height=0.7, color="#e4e4e4", zorder=1)
        nul = shares.filter(~pl.col("is_bhf") & (pl.col("alphabet") == a))[m].to_numpy()
        ax.scatter(nul, y + rng.uniform(-0.25, 0.25, len(nul)), s=5, color=GREY, alpha=0.5, lw=0, zorder=2)
        ax.scatter([r["bhf_pct"]], [y], marker="D", s=45, color=PURPLE, zorder=4)
        ax.text(1.01, y, f"p = {r['p']:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=8, clip_on=False)
    ax.set_yticks(range(len(alphas))); ax.set_yticklabels(alphas, fontsize=8.5)
    ax.set_ylim(len(alphas) - 0.5, -0.5); ax.set_xlim(left=0)
    ax.set_xlabel("% of the query's top-10 matches on a human membrane region (outside disordered regions), all k-mer sizes and metrics of the alphabet")
    ax.grid(axis="x", color="#eeeeee", zorder=0)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion, header_y=1.01, footer_y=-0.02, tight=False)


# Reduced alphabets as kmerseek defines them (src/rust/alphabets.rs, commit 982a055), for
# showing a match in the alphabet it was found in.
CLUSTERS = {
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
}


def encode(seq: str, alphabet: str) -> str:
    """Each residue as the index of its class (0, 1, 2, ...) in `alphabet`."""
    m = {a: str(i) for i, c in enumerate(CLUSTERS[alphabet]) for a in c}
    return "".join(m.get(a, "?") for a in seq)


def class_agreement(t: pl.DataFrame, alphabet: str) -> pl.DataFrame:
    """For matches found in `alphabet`: both sides encoded, positions in the same class,
    and positions with the identical residue."""
    return t.with_columns(
        pl.col("bhf_residues").map_elements(lambda s: encode(s, alphabet), return_dtype=pl.Utf8).alias("bhf_encoded"),
        pl.col("human_residues").map_elements(lambda s: encode(s, alphabet), return_dtype=pl.Utf8).alias("human_encoded"),
    ).with_columns(
        pl.struct("bhf_encoded", "human_encoded").map_elements(
            lambda r: sum(a == b for a, b in zip(r["bhf_encoded"], r["human_encoded"])), return_dtype=pl.Int64).alias("same_class"),
        pl.struct("bhf_residues", "human_residues").map_elements(
            lambda r: sum(a == b for a, b in zip(r["bhf_residues"], r["human_residues"])), return_dtype=pl.Int64).alias("identical"),
        pl.col("bhf_residues").str.len_chars().alias("length"),
    )
