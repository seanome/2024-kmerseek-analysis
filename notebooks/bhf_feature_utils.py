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
    return t, stats


def label_matches(matches: pl.DataFrame, feats: pl.DataFrame, min_frac: float = 0.5) -> pl.DataFrame:
    """For each match (target_name, target_start, target_end; 0-based end-exclusive), the
    share of the human region inside each class, and a boolean per class: at least
    `min_frac` of the human region lies inside that class."""
    m = matches.with_row_index("match_id")
    j = (m.select("match_id", "target_name", "target_start", "target_end")
          .join(feats.select("target_name", "cls", "start", "end"), on="target_name", how="inner")
          .with_columns((pl.min_horizontal("target_end", "end") - pl.max_horizontal("target_start", "start"))
                        .clip(lower_bound=0).alias("ov"))
          .filter(pl.col("ov") > 0))
    # overlapping features of one class must not be counted twice: merge per (match, class)
    # by marking covered residues
    cover = {}
    for mid, cls, ts, a, b in j.select("match_id", "cls", "target_start", "start", "end").iter_rows():
        s = cover.setdefault((mid, cls), set())
        s.update(range(max(a, ts), b))
    lens = dict(zip(m["match_id"], (m["target_end"] - m["target_start"]).to_list()))
    ends = dict(zip(m["match_id"], m["target_end"].to_list()))
    fr = {c: [0.0] * m.height for c in CLASSES}
    for (mid, cls), s in cover.items():
        n = sum(1 for r in s if r < ends[mid])
        fr[cls][mid] = n / lens[mid] if lens[mid] else 0.0
    return m.with_columns(
        *[pl.Series(f"frac_{c.replace(' ', '_')}", fr[c]) for c in CLASSES],
        *[(pl.Series(fr[c]) >= min_frac).alias(f"on_{c.replace(' ', '_')}") for c in CLASSES],
    ).drop("match_id")
