#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
format_orthoxml.py
------------------
Converts kmerseek pairwise search CSVs (QfO all-vs-all) into OrthoXML v0.3.

Filtering:
  1. Poisson p-value < --pvalue (computed from intersect_hashes + containment
     when poisson_pvalue column is absent).
  2. Jaccard >= --min-jaccard (default 0.01).

Ortholog-group formation:
  Reciprocal Best Hit (RBH) per species pair.  For each CSV (species A vs B):
    - best_hit_of[a] = argmax_{b} jaccard for each query protein a
    - best_query_for[b] = argmax_{a} jaccard for each target protein b
    - RBH pairs: (a, b) where best_hit_of[a]==b AND best_query_for[b]==a
  RBH pairs across all species pairs are then joined via union-find to form
  multi-species ortholog groups.

CSV filenames must follow the pattern produced by the Nextflow pipeline:
    {PROTEOME1}_{TAXID1}_vs_{PROTEOME2}_{TAXID2}.csv[.gz]

Usage
-----
    format_orthoxml.py --results /path/to/csvs/ --output out.orthoxml \\
                       --pvalue 0.05 --min-jaccard 0.01 \\
                       --ksize 24 --moltype hp --scaled 1
"""

import argparse
import csv
import gzip
import math
import os
import sys
from collections import defaultdict
from xml.etree.ElementTree import Element, SubElement, ElementTree, indent


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
# Poisson p-value (computed from containment columns when poisson_pvalue absent)
# ---------------------------------------------------------------------------
_MAX_HASH = 2 ** 64
PVAL_COLUMNS = ("poisson_pvalue", "prob_overlap", "prob_overlap_adjusted")


def compute_poisson_pvalue(intersect_hashes: float, containment: float,
                           containment_target: float, scaled: int = 1) -> float:
    """P(X >= intersect_hashes | Poisson(lambda)) where lambda = n_q * n_t * scaled / 2^64."""
    n = int(round(intersect_hashes))
    if n <= 0 or containment <= 0 or containment_target <= 0:
        return 1.0
    query_hashes  = intersect_hashes / containment
    target_hashes = intersect_hashes / containment_target
    lam = query_hashes * target_hashes * scaled / _MAX_HASH
    if lam <= 0:
        return 0.0

    def log_poisson_cdf(k: int, lam: float) -> float:
        log_terms = [-lam]
        log_term  = -lam
        for i in range(1, k + 1):
            log_term += math.log(lam) - math.log(i)
            log_terms.append(log_term)
        max_lt  = max(log_terms)
        log_sum = max_lt + math.log(sum(math.exp(lt - max_lt) for lt in log_terms))
        return log_sum

    try:
        return max(0.0, 1.0 - math.exp(log_poisson_cdf(n - 1, lam)))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_accession(seq_name: str) -> str:
    """Extract UniProtKB accession from sp|ACC|ENTRY or plain accession."""
    name  = seq_name.strip().split()[0]
    parts = name.split("|")
    return parts[1] if len(parts) >= 2 else name


def parse_filename(csv_path: str):
    """Return (query_proteome, query_taxid, target_proteome, target_taxid)."""
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    stem = os.path.splitext(stem)[0]
    left, right = stem.split("_vs_")
    qproteome, qtaxid = left.rsplit("_", 1)
    tproteome, ttaxid = right.rsplit("_", 1)
    return qproteome, int(qtaxid), tproteome, int(ttaxid)


def find_pval_column(header: list[str]) -> str | None:
    for col in PVAL_COLUMNS:
        if col in header:
            return col
    return None


def find_name_column(header: list[str], role: str) -> str | None:
    candidates = ["match_name", "target_name"] if role == "target" else ["query_name"]
    for col in candidates:
        if col in header:
            return col
    return None


# ---------------------------------------------------------------------------
# Per-file RBH extraction
# ---------------------------------------------------------------------------
def extract_rbh_pairs(csv_path: str, qtaxid: int, ttaxid: int,
                      qproteome: str, tproteome: str,
                      pvalue: float, min_jaccard: float,
                      scaled: int) -> list[tuple[str, str]]:
    """Return list of (query_acc, target_acc) reciprocal best-hit pairs.

    Best hit is defined by highest Jaccard score among hits passing both
    the p-value and min-jaccard filters.
    """
    opener = gzip.open if csv_path.endswith(".gz") else open

    # best_of_query[qacc]  = (best_jaccard, tacc)
    # best_of_target[tacc] = (best_jaccard, qacc)
    best_of_query:  dict[str, tuple[float, str]] = {}
    best_of_target: dict[str, tuple[float, str]] = {}

    with opener(csv_path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return []

        pval_col  = find_pval_column(reader.fieldnames)
        qname_col = find_name_column(reader.fieldnames, "query")
        tname_col = (find_name_column(reader.fieldnames, "target")
                     or find_name_column(reader.fieldnames, "match"))
        compute_pval = pval_col is None

        if compute_pval:
            need = ("intersect_hashes", "containment", "containment_target_in_query")
            if not all(c in reader.fieldnames for c in need):
                print(f"WARNING: {csv_path} missing p-value and containment columns; "
                      f"skipping.", file=sys.stderr)
                return []

        if qname_col is None or tname_col is None:
            print(f"WARNING: {csv_path} missing query/match name columns; "
                  f"skipping.", file=sys.stderr)
            return []

        for row in reader:
            try:
                jaccard = float(row["jaccard"])
                if jaccard < min_jaccard:
                    continue

                if compute_pval:
                    pval = compute_poisson_pvalue(
                        float(row["intersect_hashes"]),
                        float(row["containment"]),
                        float(row["containment_target_in_query"]),
                        scaled=scaled,
                    )
                else:
                    pval = float(row[pval_col])

                if pval >= pvalue:
                    continue

            except (ValueError, KeyError):
                continue

            qacc = parse_accession(row[qname_col])
            tacc = parse_accession(row[tname_col])
            if qacc == tacc:
                continue

            prev_q = best_of_query.get(qacc)
            if prev_q is None or jaccard > prev_q[0]:
                best_of_query[qacc] = (jaccard, tacc)

            prev_t = best_of_target.get(tacc)
            if prev_t is None or jaccard > prev_t[0]:
                best_of_target[tacc] = (jaccard, qacc)

    rbh = []
    for qacc, (_, best_tacc) in best_of_query.items():
        best_back = best_of_target.get(best_tacc)
        if best_back is not None and best_back[1] == qacc:
            rbh.append((qacc, best_tacc))

    return rbh


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",     required=True)
    parser.add_argument("--output",      required=True)
    parser.add_argument("--pvalue",      type=float, default=0.05)
    parser.add_argument("--min-jaccard", type=float, default=0.01,
                        dest="min_jaccard")
    parser.add_argument("--ksize",       type=int,   default=24)
    parser.add_argument("--moltype",     default="hp")
    parser.add_argument("--scaled",      type=int,   default=1)
    args = parser.parse_args()

    csv_files = []
    for dirpath, _, filenames in os.walk(args.results):
        for f in filenames:
            if f.endswith(".csv.gz") or (f.endswith(".csv") and not f.endswith(".csv.gz")):
                csv_files.append(os.path.join(dirpath, f))

    if not csv_files:
        print("WARNING: no CSV files found in results directory; writing empty OrthoXML.",
              file=sys.stderr)

    gene_info: dict[str, tuple[int, str]] = {}
    uf = UnionFind()
    n_rbh   = 0
    n_files = 0

    for csv_path in sorted(csv_files):
        try:
            qproteome, qtaxid, tproteome, ttaxid = parse_filename(csv_path)
        except Exception as e:
            print(f"WARNING: skipping {csv_path}: {e}", file=sys.stderr)
            continue

        n_files += 1
        pairs = extract_rbh_pairs(
            csv_path, qtaxid, ttaxid, qproteome, tproteome,
            pvalue=args.pvalue, min_jaccard=args.min_jaccard, scaled=args.scaled,
        )

        for qacc, tacc in pairs:
            gene_info.setdefault(qacc, (qtaxid, qproteome))
            gene_info.setdefault(tacc, (ttaxid, tproteome))
            uf.union(qacc, tacc)
            n_rbh += 1

    print(f"Read {n_files} CSV files, found {n_rbh} RBH pairs "
          f"(p<{args.pvalue}, jaccard>={args.min_jaccard}) "
          f"spanning {len(gene_info)} proteins.",
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
    gene_id: dict[str, int]   = {
        acc: i + 1 for i, acc in enumerate(sorted(genes_in_groups))
    }
    active_taxids: set[int] = {gene_info[a][0] for a in genes_in_groups}

    XMLNS = "http://orthoXML.org/2011/"
    root  = Element("orthoXML")
    root.set("xmlns",         XMLNS)
    root.set("version",       "0.3")
    root.set("origin",        "kmerseek")
    root.set("originVersion",
             f"k{args.ksize}_{args.moltype}_scaled{args.scaled}"
             f"_p{args.pvalue}_j{args.min_jaccard}_rbh")

    proteome_genes: dict[tuple[int, str], list[str]] = defaultdict(list)
    for acc in sorted(genes_in_groups):
        taxid, proteome_id = gene_info[acc]
        proteome_genes[(taxid, proteome_id)].append(acc)

    for (taxid, proteome_id) in sorted(proteome_genes):
        species_name, _ = QFO_SPECIES.get(taxid, (f"taxid_{taxid}", proteome_id))
        sp_el = SubElement(root, "species")
        sp_el.set("name",       species_name)
        sp_el.set("NCBITaxId",  str(taxid))
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
