#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
format_orthoxml.py
------------------
Converts kmerseek pairwise search CSVs (QfO all-vs-all) into OrthoXML v0.3.

Filtering:
  Bonferroni-corrected Poisson p-value < --pvalue, where n_tests = rows in
  that CSV file.  No Jaccard floor.

Ortholog-group formation:
  Reciprocal Best Hit (RBH) per species pair (best = highest Jaccard).
  RBH pairs across all species pairs are joined via union-find into
  multi-species ortholog groups.

CSV filenames must follow the pattern:
    {PROTEOME1}_{TAXID1}_vs_{PROTEOME2}_{TAXID2}[.kN].csv[.gz]

Usage
-----
    format_orthoxml.py --results /path/to/csvs/ --output out.orthoxml \\
                       --pvalue 0.05 --ksize 26 --moltype hp_pbotc_1st_ed \\
                       --scaled 1 [--workers N]
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent

import polars as pl


# ---------------------------------------------------------------------------
# QfO 2020-04 species table  (taxid → (species_name, proteome_id))
# ---------------------------------------------------------------------------
QFO_SPECIES = {
    36329:   ("Plasmodium falciparum", "UP000001450"),
    7070:    ("Tribolium castaneum", "UP000007266"),
    188937:  ("Methanosarcina acetivorans", "UP000002487"),
    83332:   ("Mycobacterium tuberculosis", "UP000001584"),
    10090:   ("Mus musculus", "UP000000589"),
    7719:    ("Ciona intestinalis", "UP000008144"),
    81824:   ("Monosiga brevicollis", "UP000001357"),
    321614:  ("Phaeosphaeria nodorum", "UP000001055"),
    9595:    ("Gorilla gorilla gorilla", "UP000001519"),
    374847:  ("Korarchaeum cryptofilum", "UP000001686"),
    44689:   ("Dictyostelium discoideum", "UP000002195"),
    7165:    ("Anopheles gambiae", "UP000007062"),
    9606:    ("Homo sapiens", "UP000005640"),
    284812:  ("Schizosaccharomyces pombe", "UP000002485"),
    243232:  ("Methanocaldococcus jannaschii", "UP000000805"),
    9598:    ("Pan troglodytes", "UP000002277"),
    208964:  ("Pseudomonas aeruginosa", "UP000002438"),
    9615:    ("Canis lupus familiaris", "UP000002254"),
    684364:  ("Batrachochytrium dendrobatidis", "UP000007241"),
    8090:    ("Oryzias latipes", "UP000001038"),
    243274:  ("Thermotoga maritima", "UP000008183"),
    8364:    ("Xenopus tropicalis", "UP000008143"),
    184922:  ("Giardia intestinalis", "UP000001548"),
    418459:  ("Puccinia graminis", "UP000008783"),
    272561:  ("Chlamydia trachomatis", "UP000000431"),
    100226:  ("Streptomyces coelicolor", "UP000001973"),
    189518:  ("Leptospira interrogans", "UP000001408"),
    83333:   ("Escherichia coli", "UP000000625"),
    559292:  ("Saccharomyces cerevisiae", "UP000002311"),
    122586:  ("Neisseria meningitidis", "UP000000425"),
    243230:  ("Deinococcus radiodurans", "UP000002524"),
    3218:    ("Physcomitrella patens", "UP000006727"),
    665079:  ("Sclerotinia sclerotiorum", "UP000001312"),
    224911:  ("Bradyrhizobium diazoefficiens", "UP000002526"),
    164328:  ("Phytophthora ramorum", "UP000005238"),
    1111708: ("Synechocystis sp. PCC 6803", "UP000001425"),
    7227:    ("Drosophila melanogaster", "UP000000803"),
    324602:  ("Chloroflexus aurantiacus", "UP000002008"),
    7955:    ("Danio rerio", "UP000000437"),
    35128:   ("Thalassiosira pseudonana", "UP000001449"),
    224308:  ("Bacillus subtilis", "UP000001570"),
    6945:    ("Ixodes scapularis", "UP000001555"),
    251221:  ("Gloeobacter violaceus", "UP000000557"),
    224324:  ("Aquifex aeolicus", "UP000000798"),
    214684:  ("Cryptococcus neoformans", "UP000002149"),
    436308:  ("Nitrosopumilus maritimus", "UP000000792"),
    64091:   ("Halobacterium salinarum", "UP000000554"),
    284591:  ("Yarrowia lipolytica", "UP000001300"),
    85962:   ("Helicobacter pylori", "UP000000429"),
    5722:    ("Trichomonas vaginalis", "UP000001542"),
    9913:    ("Bos taurus", "UP000009136"),
    367110:  ("Neurospora crassa", "UP000001805"),
    6239:    ("Caenorhabditis elegans", "UP000001940"),
    190304:  ("Fusobacterium nucleatum", "UP000000521"),
    4577:    ("Zea mays", "UP000007305"),
    273057:  ("Saccharolobus solfataricus", "UP000001974"),
    3702:    ("Arabidopsis thaliana", "UP000006548"),
    243090:  ("Rhodopirellula baltica", "UP000001025"),
    13616:   ("Monodelphis domestica", "UP000002280"),
    5664:    ("Leishmania major", "UP000000542"),
    45351:   ("Nematostella vectensis", "UP000001593"),
    9031:    ("Gallus gallus", "UP000000539"),
    237561:  ("Candida albicans", "UP000000559"),
    39947:   ("Oryza sativa subsp. japonica", "UP000059680"),
    330879:  ("Neosartorya fumigata", "UP000002530"),
    243231:  ("Geobacter sulfurreducens", "UP000000577"),
    69014:   ("Thermococcus kodakarensis", "UP000000536"),
    243273:  ("Mycoplasma genitalium", "UP000000807"),
    3055:    ("Chlamydomonas reinhardtii", "UP000006906"),
    6412:    ("Helobdella robusta", "UP000015101"),
    5888:    ("Paramecium tetraurelia", "UP000000600"),
    10116:   ("Rattus norvegicus", "UP000002494"),
    515635:  ("Dictyoglomus turgidum", "UP000007719"),
    237631:  ("Ustilago maydis", "UP000000561"),
    7739:    ("Branchiostoma floridae", "UP000001554"),
    226186:  ("Bacteroides thetaiotaomicron", "UP000001414"),
    289376:  ("Thermodesulfovibrio yellowstonii", "UP000000718"),
    7918:    ("Lepisosteus oculatus", "UP000018468"),
}


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PVAL_COLUMNS = ("poisson_pvalue", "prob_overlap", "prob_overlap_adjusted")


def _acc_expr(col: str) -> pl.Expr:
    """Vectorized: 'sp|ACC|ENTRY description' → 'ACC', or plain accession."""
    first_tok = pl.col(col).str.split_exact(" ", 1).struct.field("field_0")
    return (
        pl.when(first_tok.str.contains("|", literal=True))
        .then(first_tok.str.split_exact("|", 2).struct.field("field_1"))
        .otherwise(first_tok)
    )


def parse_filename(csv_path: str):
    """Return (query_proteome, query_taxid, target_proteome, target_taxid)."""
    stem = os.path.basename(csv_path)
    while "." in stem:
        stem, _ = os.path.splitext(stem)
        if "_vs_" in stem:
            try:
                left, right = stem.split("_vs_")
                qproteome, qtaxid = left.rsplit("_", 1)
                tproteome, ttaxid = right.rsplit("_", 1)
                return qproteome, int(qtaxid), tproteome, int(ttaxid)
            except (ValueError, TypeError):
                continue
    raise ValueError(f"Cannot parse species pair from: {os.path.basename(csv_path)}")


# ---------------------------------------------------------------------------
# Per-file worker (runs in a subprocess)
# ---------------------------------------------------------------------------
def _process_file(task: tuple) -> list[tuple]:
    """
    Read one CSV, apply per-file Bonferroni filter on poisson_pvalue, compute
    RBH pairs using vectorised Polars operations.

    Returns list of (qtaxid, qproteome, ttaxid, tproteome, qacc, tacc).
    """
    csv_path, qtaxid, qproteome, ttaxid, tproteome, pvalue = task

    try:
        df = pl.read_csv(csv_path, infer_schema_length=200)
    except Exception as e:
        print(f"WARNING: cannot read {csv_path}: {e}", file=sys.stderr)
        return []

    if len(df) == 0:
        return []

    n_tests = len(df)
    pvc = next((c for c in PVAL_COLUMNS if c in df.columns), None)
    qcol = "query_name" if "query_name" in df.columns else None
    tcol = next((c for c in ("target_name", "match_name") if c in df.columns), None)

    if pvc is None or qcol is None or tcol is None:
        print(f"WARNING: {csv_path} missing required columns; skipping.", file=sys.stderr)
        return []

    # Ensure numeric types (schema inference may read as Utf8 in edge cases)
    for col in ("jaccard", pvc):
        if df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Float64))

    # Parse accessions, apply Bonferroni filter, drop unneeded columns
    df = (df
          .with_columns([
              _acc_expr(qcol).alias("qacc"),
              _acc_expr(tcol).alias("tacc"),
          ])
          .filter(
              (pl.col(pvc) * n_tests < pvalue) &
              (pl.col("qacc") != pl.col("tacc"))
          )
          .select(["qacc", "tacc", "jaccard"]))

    if len(df) == 0:
        return []

    # Best hit per query protein = target with highest Jaccard
    best_q = (df.group_by("qacc")
               .agg(pl.col("tacc").sort_by("jaccard", descending=True).first()
                    .alias("best_tacc")))

    # Best query per target protein = query with highest Jaccard
    best_t = (df.group_by("tacc")
               .agg(pl.col("qacc").sort_by("jaccard", descending=True).first()
                    .alias("best_qacc")))

    # RBH = mutual best hits
    rbh = (best_q
           .join(best_t, left_on="best_tacc", right_on="tacc", how="inner")
           .filter(pl.col("best_qacc") == pl.col("qacc"))
           .select(["qacc", "best_tacc"])
           .rows())

    return [(qtaxid, qproteome, ttaxid, tproteome, qacc, tacc)
            for qacc, tacc in rbh]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",  required=True)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--pvalue",   type=float, default=0.05)
    parser.add_argument("--ksize",    type=int,   default=24)
    parser.add_argument("--moltype",  default="hp")
    parser.add_argument("--scaled",   type=int,   default=1)
    parser.add_argument("--workers",  type=int,   default=min(8, os.cpu_count() or 4))
    args = parser.parse_args()

    csv_files = [
        os.path.join(dp, f)
        for dp, _, fns in os.walk(args.results)
        for f in fns
        if f.endswith(".csv.gz") or (f.endswith(".csv") and not f.endswith(".csv.gz"))
    ]

    if not csv_files:
        print("WARNING: no CSV files found in results directory; writing empty OrthoXML.",
              file=sys.stderr)

    tasks = []
    for p in sorted(csv_files):
        try:
            qprot, qtax, tprot, ttax = parse_filename(p)
            tasks.append((p, qtax, qprot, ttax, tprot, args.pvalue))
        except Exception as e:
            print(f"WARNING: skipping {p}: {e}", file=sys.stderr)

    print(f"Processing {len(tasks)} CSV files with {args.workers} workers ...",
          file=sys.stderr)

    gene_info: dict[str, tuple[int, str]] = {}
    uf = UnionFind()
    n_rbh = 0
    n_done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_file, t): t for t in tasks}
        for fut in as_completed(futures):
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(tasks)} files done, {n_rbh} RBH pairs so far ...",
                      file=sys.stderr)
            try:
                pairs = fut.result()
            except Exception as e:
                print(f"WARNING: worker error: {e}", file=sys.stderr)
                continue
            for qtaxid, qproteome, ttaxid, tproteome, qacc, tacc in pairs:
                gene_info.setdefault(qacc, (qtaxid, qproteome))
                gene_info.setdefault(tacc, (ttaxid, tproteome))
                uf.union(qacc, tacc)
                n_rbh += 1

    print(f"Read {len(tasks)} CSV files, found {n_rbh} RBH pairs "
          f"(Bonferroni p<{args.pvalue}) spanning {len(gene_info)} proteins.",
          file=sys.stderr)

    components: dict[str, list[str]] = defaultdict(list)
    for acc in gene_info:
        components[uf.find(acc)].append(acc)

    ortholog_groups = [
        sorted(accs)
        for accs in components.values()
        if len({gene_info[a][0] for a in accs}) >= 2
    ]

    print(f"Formed {len(ortholog_groups)} ortholog groups.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Build OrthoXML
    # -----------------------------------------------------------------------
    genes_in_groups: set[str] = {a for grp in ortholog_groups for a in grp}
    gene_id: dict[str, int] = {
        acc: i + 1 for i, acc in enumerate(sorted(genes_in_groups))
    }

    root = Element("orthoXML")
    root.set("xmlns",         "http://orthoXML.org/2011/")
    root.set("version",       "0.3")
    root.set("origin",        "kmerseek")
    root.set("originVersion",
             f"k{args.ksize}_{args.moltype}_scaled{args.scaled}"
             f"_bonferroni_p{args.pvalue}_rbh")

    proteome_genes: dict[tuple[int, str], list[str]] = defaultdict(list)
    for acc in sorted(genes_in_groups):
        taxid, proteome_id = gene_info[acc]
        proteome_genes[(taxid, proteome_id)].append(acc)

    for (taxid, proteome_id) in sorted(proteome_genes):
        species_name, _ = QFO_SPECIES.get(taxid, (f"taxid_{taxid}", proteome_id))
        sp_el = SubElement(root, "species")
        sp_el.set("name",      species_name)
        sp_el.set("NCBITaxId", str(taxid))
        db_el = SubElement(sp_el, "database")
        db_el.set("name",    proteome_id)
        db_el.set("version", "QfO-2020")
        genes_el = SubElement(db_el, "genes")
        for acc in sorted(proteome_genes[(taxid, proteome_id)]):
            g = SubElement(genes_el, "gene")
            g.set("id",     str(gene_id[acc]))
            g.set("protId", acc)

    groups_el = SubElement(root, "groups")
    for grp_idx, accs in enumerate(ortholog_groups, start=1):
        og = SubElement(groups_el, "orthologGroup")
        og.set("id", str(grp_idx))
        for acc in accs:
            gr = SubElement(og, "geneRef")
            gr.set("id", str(gene_id[acc]))

    tree = ElementTree(root)
    try:
        indent(tree, space="  ")
    except AttributeError:
        pass

    with open(args.output, "wb") as fh:
        fh.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
        tree.write(fh, encoding="utf-8", xml_declaration=False)

    print(f"Wrote OrthoXML to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
