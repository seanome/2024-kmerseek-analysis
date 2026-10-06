#!/usr/bin/env python3
"""ELM Stage 0: do the flanks of a motif keep their hydrophobic-polar pattern across species?

No search. Runs on a laptop. Must finish, and be read, before the ELM search is wired.

Why: an ELM motif is 3 to 15 residues and a hydrophobic-polar (HP) seed is 19 to 30, so an
exact seed can only cover a motif by also matching 8 to 20 flanking residues. Whether that
can happen depends on whether the flanks keep their HP pattern in the ortholog. This script
measures it on real ortholog pairs.

Inputs
  queries    human ELM "true positive" instances (scripts/fetch_elm.py) on the QfO 2020_04
             human proteome, with coordinates checked against that sequence: an instance is
             kept only if it lies inside it and its class regex matches a stretch
             overlapping it. `in_midi_plus` marks the instances on the 998 midi-plus human
             proteins, the set the search runs on; only 100 instances on 67 proteins are
             there, too few for per-species numbers, so this no-search stage uses them all.
  orthologs  1:1 OMA pairs from the QfO release's .idmapping files, the source
             nextflow-runs/qfo-dnds-omega/bin/build_ortholog_pairs.py uses: a human protein
             and a target protein in the same OMA group, each the group's only member from
             its species. ecoli has no OMA cross-references in this release, so no pairs.
  alignment  MAFFT (L-INS-i, `mafft --localpair --maxiterate 1000`) on each human-ortholog
             pair.

Per (instance, species) the script reports, over three windows of the human protein:
  motif   the ELM instance's own residues
  flanks  the 10 residues on each side of the motif (fewer at a protein end)
  rest    every other residue of the human protein
the number of alignment columns where both proteins have a residue, the share of those
columns with the same hp_pbotc_1st_ed2 class, Pr(same class), and the share with the same
residue (percent identity). A gap column counts in neither.

The chance level is Pr(same class | chance) = h_human * h_ortholog + p_human * p_ortholog,
with h the hydrophobic share and p the polar share of each whole protein.

It also reports the longest run of consecutive alignment columns, with no gap on either
side, where the two proteins have the same HP class, among the runs that cover at least 80%
of the motif's residues (the COVER rule of the ELM scoring). An exact k-mer seed of length
k that covers the motif can exist only if that run is at least k long.

Outputs, in --out-dir:
  stage0_instances.parquet  one row per query instance, with pLDDT and metapredict disorder
                            over the motif and over the flanks, and overlapping Swiss-Prot
                            features
  stage0_pairs.parquet      one row per (instance, species) with an ortholog
  alignments/<species>/<human>__<ortholog>.fasta   the MAFFT alignments
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PIPELINE_BIN = REPO / "nextflow-runs" / "qfo-pfam-region-benchmark" / "bin"
sys.path.insert(0, str(PIPELINE_BIN))
from build_query_covariates import read_plddt  # noqa: E402

MIDI = Path("/Users/olga/data/qfo-pfam-region-midi-plus")
ELM_DIR = Path("/Users/olga/data/elm-motif-transfer")
QFO = Path(
    "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
)
AF_CACHE = Path("/Users/olga/data/alphafold_structures")
MAFFT = "/Users/olga/anaconda3/envs/orthofinder/bin/mafft"
HUMAN = "Homo sapiens"
FLANK = 10
COVER_MIN = 0.8
#: hp_pbotc_1st_ed2, as in notebooks/hero_example_utils.HP_CLASSES.
HYDROPHOBIC = set("ACFILMPVWY")
POLAR = set("DEGHKNQRST")
NINE = [
    "mouse",
    "chicken",
    "zebrafish",
    "ciona",
    "fly",
    "worm",
    "yeast",
    "arabidopsis",
    "ecoli",
]


def load_species() -> dict[str, tuple[str, str, str, int]]:
    """label -> (subdir, proteome, taxon, Mya), from the region pipeline's species registry."""
    reg = pl.read_csv(
        REPO
        / "nextflow-runs"
        / "qfo-pfam-region-benchmark"
        / "assets"
        / "qfo_species.tsv",
        separator="\t",
        infer_schema_length=0,
    )
    rows = {r["label"]: r for r in reg.iter_rows(named=True)}
    return {
        sp: (
            rows[sp]["subdir"],
            rows[sp]["proteome"],
            rows[sp]["taxon"],
            int(rows[sp]["mya"]),
        )
        for sp in NINE
    }


SPECIES = load_species()


def read_fasta(path: Path) -> dict[str, str]:
    seqs, acc, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if acc:
                seqs[acc] = "".join(buf)
            parts = line[1:].split()[0].split("|")
            acc, buf = (parts[1] if len(parts) >= 2 else parts[0]), []
        else:
            buf.append(line.strip())
    if acc:
        seqs[acc] = "".join(buf)
    return seqs


def hp(c: str) -> str | None:
    return "H" if c in HYDROPHOBIC else ("P" if c in POLAR else None)


def regex_at_instance(regex: str, seq: str, start: int, end: int) -> bool:
    pat = re.compile(regex)
    for i in range(max(0, start - 60), min(len(seq), end)):
        m = pat.match(seq, i)
        if m and m.end() > start and m.start() < end and m.end() > m.start():
            return True
    return False


def oma_pairs(species: str, human_keep: set[str]) -> dict[str, str]:
    """human accession -> ortholog accession, 1:1 OMA, as build_ortholog_pairs.py pairs them."""
    subdir, proteome, taxon, _ = SPECIES[species]
    t_fasta = QFO / subdir / f"{proteome}_{taxon}.fasta"
    t_keep = set(read_fasta(t_fasta))

    def groups(path: Path, keep: set[str]) -> dict[str, set[str]]:
        g = defaultdict(set)
        with open(path) as fh:
            for line in fh:
                acc, db, value = line.rstrip("\n").split("\t")
                if db == "OMA" and acc in keep:
                    g[value].add(acc)
        return g

    hg = groups(QFO / "Eukaryota" / "UP000005640_9606.idmapping", human_keep)
    tg = groups(QFO / subdir / f"{proteome}_{taxon}.idmapping", t_keep)
    out = {}
    for grp in set(hg) & set(tg):
        if len(hg[grp]) == 1 and len(tg[grp]) == 1:
            out[next(iter(hg[grp]))] = next(iter(tg[grp]))
    return out


def align(h_acc: str, h_seq: str, t_acc: str, t_seq: str, out: Path) -> tuple[str, str]:
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        inp = out.with_suffix(".in.fasta")
        inp.write_text(f">{h_acc}\n{h_seq}\n>{t_acc}\n{t_seq}\n")
        res = subprocess.run(
            # --anysymbol: selenocysteine (U) is a real residue in some human proteins.
            [
                MAFFT,
                "--anysymbol",
                "--localpair",
                "--maxiterate",
                "1000",
                "--quiet",
                "--thread",
                "1",
                str(inp),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        tmp = out.with_suffix(".tmp")
        tmp.write_text(res.stdout)
        tmp.rename(out)
        inp.unlink()
    recs, name, buf = [], None, []
    for line in out.read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                recs.append("".join(buf))
            name, buf = line, []
        else:
            buf.append(line.strip())
    recs.append("".join(buf))
    return recs[0].upper(), recs[1].upper()


def window_stats(
    ha: str,
    ta: str,
    hpos: list[int | None],
    lo: int,
    hi: int,
    exclude: tuple[int, int] | None = None,
) -> dict:
    """Columns whose human residue index is in [lo, hi) (minus `exclude`), both residues present."""
    n = same_cls = same_res = 0
    for col, i in enumerate(hpos):
        if i is None or not (lo <= i < hi):
            continue
        if exclude and exclude[0] <= i < exclude[1]:
            continue
        a, b = ha[col], ta[col]
        if b == "-" or hp(a) is None or hp(b) is None:
            continue
        n += 1
        same_cls += hp(a) == hp(b)
        same_res += a == b
    return {
        "n_cols": n,
        "pr_same_class": same_cls / n if n else None,
        "identity": same_res / n if n else None,
    }


def longest_covering_run(
    ha: str, ta: str, hpos: list[int | None], start: int, end: int
) -> int:
    """Longest gap-free run of same-HP-class columns covering >= COVER_MIN of [start, end)."""
    need = COVER_MIN * (end - start)
    best = run = covered = 0
    for col in range(len(ha)):
        a, b, i = ha[col], ta[col], hpos[col]
        ok = a != "-" and b != "-" and hp(a) is not None and hp(a) == hp(b)
        if ok:
            run += 1
            covered += i is not None and start <= i < end
            if covered >= need:
                best = max(best, run)
        else:
            run = covered = 0
    return best


def chance_same(h_seq: str, t_seq: str) -> float:
    def shares(s):
        c = [hp(x) for x in s if hp(x)]
        h = sum(x == "H" for x in c) / len(c)
        return h, 1 - h

    hh, hp_ = shares(h_seq)
    th, tp = shares(t_seq)
    return hh * th + hp_ * tp


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=ELM_DIR / "stage0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- queries: human TP instances on midi-plus proteins, checked on the searched sequence
    hseq = read_fasta(QFO / "Eukaryota" / "UP000005640_9606.fasta")
    midi = set(read_fasta(MIDI / "extract" / "244_human_queries.fasta"))
    inst = pl.read_parquet(ELM_DIR / "elm_instances_tp.parquet").filter(
        pl.col("organism") == HUMAN
    )
    n_all_human = inst.height
    q = inst.filter(pl.col("accession").is_in(list(hseq))).with_columns(
        in_midi_plus=pl.col("accession").is_in(list(midi))
    )
    rows = []
    for r in q.iter_rows(named=True):
        s = hseq[r["accession"]]
        ok = r["end"] <= len(s) and regex_at_instance(
            r["regex"], s, r["start"], r["end"]
        )
        rows.append({"elm_instance": r["elm_instance"], "usable_on_searched_seq": ok})
    q = q.join(pl.DataFrame(rows), on="elm_instance").with_columns(
        length_bin=pl.when(pl.col("motif_length") <= 6)
        .then(pl.lit("3-6 aa"))
        .when(pl.col("motif_length") <= 10)
        .then(pl.lit("7-10 aa"))
        .when(pl.col("motif_length") <= 15)
        .then(pl.lit("11-15 aa"))
        .otherwise(pl.lit("16+ aa"))
    )
    print(
        f"human true-positive instances: {n_all_human:_}; on the QfO human proteome: {q.height:_} "
        f"on {q['accession'].n_unique():_} proteins; usable there: {q['usable_on_searched_seq'].sum():_}; "
        f"of those on the 998 midi-plus proteins: "
        f"{q.filter(pl.col('usable_on_searched_seq') & pl.col('in_midi_plus')).height}"
    )
    q = q.filter("usable_on_searched_seq")
    print(
        f"  classes: {q['elm_class'].n_unique()}, motif length 2 aa (below the 3-6 bin): "
        f"{(q['motif_length'] < 3).sum()}"
    )
    print(q.group_by("length_bin").agg(n=pl.len()).sort("length_bin"))

    # ---- covariates: pLDDT from the pipeline's human models, metapredict motif and flanks
    # The pipeline's own human models for the midi-plus proteins; for the rest, the flat
    # AlphaFold cache, used only when the model is as long as the QfO sequence.
    plddt = {}
    for acc in q["accession"].unique():
        f = MIDI / "structures_human" / f"AF-{acc}-F1.cif"
        cands = (
            [f]
            if f.exists()
            else sorted(AF_CACHE.glob(f"AF-{acc}-F1-model_v*.cif"))[-1:]
        )
        if cands:
            t = read_plddt(cands[0])
            if t and len(t) == len(hseq[acc]):
                plddt[acc] = t
    dis = (
        pl.read_parquet(
            MIDI / "extract" / "244_human_queries.disorder_metapredict.parquet"
        )
        if (
            MIDI / "extract" / "244_human_queries.disorder_metapredict.parquet"
        ).exists()
        else None
    )
    work = out / "metapredict"
    work.mkdir(exist_ok=True)
    win = []
    for r in q.iter_rows(named=True):
        L = len(hseq[r["accession"]])
        win.append((r["accession"], r["start"] + 1, r["end"]))
        if r["start"] > 0:
            win.append((r["accession"], max(0, r["start"] - FLANK) + 1, r["start"]))
        if r["end"] < L:
            win.append((r["accession"], r["end"] + 1, min(L, r["end"] + FLANK)))
    pl.DataFrame(
        win, schema=["accession", "domain_start", "domain_end"], orient="row"
    ).unique().write_parquet(work / "windows.parquet")
    fa = work / "queries.fasta"
    fa.write_text("".join(f">sp|{a}|q\n{hseq[a]}\n" for a in q["accession"].unique()))
    dom_out = work / "window_disorder.parquet"
    if not dom_out.exists():
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE_BIN / "predict_disorder_metapredict.py"),
                "--fasta",
                str(fa),
                "--out",
                str(work / "protein.parquet"),
                "--domains",
                str(work / "windows.parquet"),
                "--domains-out",
                str(dom_out),
                "--threshold",
                "0.5",
            ],
            check=True,
            capture_output=True,
        )
    wd = {
        (r["accession"], r["domain_start"], r["domain_end"]): (
            r["disorder_fraction_region"],
            r["n_residues_region"],
        )
        for r in pl.read_parquet(dom_out).iter_rows(named=True)
    }
    sp = pl.read_parquet(MIDI / "truth_swissprot" / "human_swissprot_truth.parquet")
    cov = []
    for r in q.iter_rows(named=True):
        a, s, e = r["accession"], r["start"], r["end"]
        L = len(hseq[a])
        t = plddt.get(a)
        w = t[s:e] if t and e <= len(t) else []
        m = wd.get((a, s + 1, e))
        fl = [
            wd.get((a, max(0, s - FLANK) + 1, s)),
            wd.get((a, e + 1, min(L, e + FLANK))),
        ]
        fl = [x for x in fl if x]
        n_fl = sum(x[1] for x in fl)
        feats = sp.filter(
            (pl.col("accession") == a)
            & (pl.col("domain_start") <= e)
            & (pl.col("domain_end") > s)
        )
        cov.append(
            {
                "elm_instance": r["elm_instance"],
                "mean_plddt_motif": sum(w) / len(w) if w else None,
                "frac_disordered_motif": m[0] if m else None,
                "frac_disordered_flanks": (
                    sum(x[0] * x[1] for x in fl) / n_fl if n_fl else None
                ),
                "swissprot_features": sorted(set(feats["pfam_id"].to_list())),
            }
        )
    q = q.join(pl.DataFrame(cov), on="elm_instance")
    q.write_parquet(out / "stage0_instances.parquet")

    # ---- orthologs, alignments, windows
    human_keep = set(read_fasta(QFO / "Eukaryota" / "UP000005640_9606.fasta"))
    tasks = []
    for species, (subdir, proteome, taxon, mya) in SPECIES.items():
        pairs = oma_pairs(species, human_keep)
        tseq = read_fasta(QFO / subdir / f"{proteome}_{taxon}.fasta")
        n_q = sum(a in pairs for a in q["accession"].unique())
        print(
            f"{species:12s} 1:1 OMA pairs with a human protein: {len(pairs):_}; "
            f"human ELM proteins with one: {n_q} of {q['accession'].n_unique()}"
        )
        for a in q["accession"].unique():
            if a in pairs and pairs[a] in tseq:
                tasks.append((species, mya, a, pairs[a], tseq[pairs[a]]))

    def run(t):
        species, mya, a, ta, ts = t
        return t, align(
            a, hseq[a], ta, ts, out / "alignments" / species / f"{a}__{ta}.fasta"
        )

    with ThreadPoolExecutor(args.workers) as ex:
        aligned = list(ex.map(run, tasks))

    rows = []
    for (species, mya, a, ta, ts), (ha, tal) in aligned:
        hpos, i = [], 0
        for c in ha:
            hpos.append(None if c == "-" else i)
            i += c != "-"
        L = len(hseq[a])
        chance = chance_same(hseq[a], ts)
        for r in q.filter(pl.col("accession") == a).iter_rows(named=True):
            s, e = r["start"], r["end"]
            mo = window_stats(ha, tal, hpos, s, e)
            fl = window_stats(
                ha, tal, hpos, max(0, s - FLANK), min(L, e + FLANK), exclude=(s, e)
            )
            rest = window_stats(
                ha, tal, hpos, 0, L, exclude=(max(0, s - FLANK), min(L, e + FLANK))
            )
            rows.append(
                {
                    "elm_instance": r["elm_instance"],
                    "elm_class": r["elm_class"],
                    "in_midi_plus": r["in_midi_plus"],
                    "accession": a,
                    "species": species,
                    "mya": mya,
                    "ortholog": ta,
                    "motif_length": r["motif_length"],
                    "length_bin": r["length_bin"],
                    **{f"motif_{k}": v for k, v in mo.items()},
                    **{f"flanks_{k}": v for k, v in fl.items()},
                    **{f"rest_{k}": v for k, v in rest.items()},
                    "pr_same_class_chance": chance,
                    "longest_hp_run_covering_motif": longest_covering_run(
                        ha, tal, hpos, s, e
                    ),
                }
            )
    pairs_df = pl.DataFrame(rows)
    pairs_df.write_parquet(out / "stage0_pairs.parquet")
    print(
        f"\nwrote {out / 'stage0_instances.parquet'} ({q.height} instances) and "
        f"{out / 'stage0_pairs.parquet'} ({pairs_df.height} instance-species pairs)"
    )


if __name__ == "__main__":
    main()
