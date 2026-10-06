#!/usr/bin/env python3
"""Add two bonus cases to the notebook-244 export, with tier = "bonus".

Case A: Ced-9 (C. elegans, P41958) against human BCL2 (P10415). The feature is BCL2's
Swiss-Prot "Motif BH1", 136-155 (UniProt P10415, entry version 279). Swiss-Prot has no
feature for the BH3-binding groove as one range; BH1 is one of the three motifs that line
it and the one kmerseek's region sits on. kmerseek's call is the one region
`kmerseek pair` finds at hp_lehninger2 k=17 (notebook 241's arm for this pair).

Case B: BHF (Botryllus schlosseri histocompatibility factor, 252 aa), the sequence notebook
241 searched: record "BHF" of /Users/olga/data/botryllus/Bs_proteins.fa, identical to
/Users/olga/data/botryllus/alphabet-ranking-three-cases/queries.fa. No true feature, so
feature_type = "NONE" and no coordinates. Each tool's best human hit (lowest E-value, E <= 10)
is recorded with its box on BHF and on the human protein, or "no call". kmerseek uses
notebook 241's BHF arm, hp_lehninger2 k=24, with 241's index and search settings.

Both cases search the whole QfO human proteome, the human proteome of notebook 244. In 244
human was the query; here it is the target, so the feature of case A sits on the target.

The bonus rows go after the 183 rows of notebook 244, whose lines are kept byte for byte.
Four columns are added to both CSVs: tier ("bonus" on the new rows, empty on the old),
query_species, feature_on (which protein carries the feature) and note. 244_case_calls.csv
also gains score, evalue and partner_rank (case A: BCL2's rank among the tool's human hits).
case_id continues the row index: 183 = case A, 184 = case B.

Steps:
    kmerseek  run kmerseek pair (case A) and index + search (case B) into --data-dir
    collect   merge those with the six tools' output from Sherlock
              (scripts/bonus_244_tools.sbatch) into tables/244_bonus_tool_hits.tsv
    write     add the bonus rows to the three tables from that file (the default;
              scripts/export_244_case_calls.py runs this too, after it rewrites its files)

Run with the 2025-kmerseek-analysis env:
    python scripts/export_244_bonus_cases.py kmerseek
    python scripts/export_244_bonus_cases.py collect --tools-dir <copy of $SCRATCH/244-bonus/out>
    python scripts/export_244_bonus_cases.py write
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import hero_example_utils as he  # noqa: E402

TAB = ROOT / "tables"
HITS = TAB / "244_bonus_tool_hits.tsv"
#: The two queries, as scripts/bonus_244_tools.sbatch reads them on Sherlock.
QUERIES = TAB / "244_bonus_queries.fasta"
DATA_DIR = Path("/Users/olga/data/qfo-pfam-region-midi-plus/244_bonus")
KMERSEEK = Path("/Users/olga/code/kmerseek-ka-lambda-region/target/release/kmerseek")
BHF_FASTA = Path("/Users/olga/data/botryllus/Bs_proteins.fa")
BHF_LENGTH = 252

#: notebook 244's report cutoff (params.evalue_report); every tool reports at it.
EVALUE_MAX = 10.0
#: The region-extension penalty and X-drop notebook 241 used for hp_lehninger2.
HP_PENALTY, HP_XDROP = "1.51", "6.03"

#: Tool label -> (arm, file the Sherlock job writes).
TOOLS = {
    "phmmer": ("hmmer3_phmmer.default", "phmmer.tsv"),
    "MMseqs2": ("mmseqs2_seqseq.s7", "mmseqs2_seqseq.tsv"),
    "MMseqs2 iterative": ("mmseqs2_iterative.s7", "mmseqs2_iterative.tsv"),
    "Foldseek": ("foldseek.3di_aa", "foldseek.tsv"),
    "ProstT5": ("prostt5.3di_from_seq", "prostt5.tsv"),
    "Reseek": ("reseek.verysensitive", "reseek.tsv"),
}
STRUCTURE_TOOLS = {"Foldseek", "Reseek"}

CASE_A = dict(
    case_id=183,
    gene="ced-9",
    query="P41958",
    query_species="worm",
    species="human",
    target="P10415",
    feature_type="MOTIF",
    swissprot_description="BH1",
    feature_start=136,
    feature_end=155,
    feature_on="target",
    kmerseek_arm="kmerseek.hp_lehninger2_k17_pair",
    k=17,
)
CASE_B = dict(
    case_id=184,
    gene="BHF",
    query="BHF",
    query_species="botryllus",
    species="human",
    target=None,
    feature_type="NONE",
    swissprot_description=None,
    feature_start=None,
    feature_end=None,
    feature_on="none",
    kmerseek_arm="kmerseek.hp_lehninger2_k24_lcTrue",
    k=24,
)

NEW_CASE_COLS = ["tier", "query_species", "feature_on", "note"]
NEW_CALL_COLS = NEW_CASE_COLS + ["score", "evalue", "partner_rank"]


# ---------------------------------------------------------------------------
# kmerseek
# ---------------------------------------------------------------------------
def human_fasta() -> Path:
    return he.proteome_fasta("human")


def write_record(src: Path, match, dst: Path, rename: str | None = None) -> None:
    """Copy the one FASTA record whose header satisfies ``match`` to ``dst``."""
    out, keep = [], False
    with open(src) as fh:
        for line in fh:
            if line.startswith(">"):
                keep = match(line)
                if keep and rename:
                    line = f">{rename}\n"
            if keep:
                out.append(line)
    assert sum(x.startswith(">") for x in out) == 1, (src, len(out))
    dst.write_text("".join(out))


def query_records() -> list[tuple[str, str, str]]:
    """(accession, rest of header, sequence) of the two queries."""
    ced9 = he.sequences("worm", {"P41958"})["P41958"]
    bhf = he.su.read_fasta(BHF_FASTA, {"BHF"})["BHF"]
    return [
        ("P41958", f"species=worm {he.protein_name('worm', 'P41958')}", ced9),
        (
            "BHF",
            "species=botryllus Botryllus histocompatibility factor (Bs_proteins.fa)",
            bhf,
        ),
    ]


def run_kmerseek(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    ced9, bcl2, bhf = data_dir / "ced9.fa", data_dir / "bcl2.fa", data_dir / "bhf.fa"
    write_record(he.proteome_fasta("worm"), lambda h: "|P41958|" in h, ced9)
    write_record(human_fasta(), lambda h: "|P10415|" in h, bcl2)
    write_record(BHF_FASTA, lambda h: h.split()[0] == ">BHF", bhf, rename="BHF")
    assert len(he.su.read_fasta(bhf)["BHF"]) == BHF_LENGTH
    with open(QUERIES, "w") as fh:
        for acc, rest, seq in query_records():
            fh.write(f">{acc} {rest}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
    print(f"wrote {QUERIES}")

    k = str(CASE_A["k"])
    subprocess.run(
        [KMERSEEK, "pair", "-q", ced9, "-t", bcl2, "-k", k, "-a", "hp_lehninger2",
         "-o", data_dir / "ced9_bcl2.hp_lehninger2.k17.pair.json"],
        check=True,
    )  # fmt: skip

    k = str(CASE_B["k"])
    idx = data_dir / f"human.hp_lehninger2.k{k}.rocksdb"
    ext = ["--extend-mismatch-penalty", HP_PENALTY, "--extend-xdrop", HP_XDROP]
    if not (idx / "CURRENT").exists():
        subprocess.run(
            [KMERSEEK, "index", "-i", human_fasta(), "-o", idx, "-k", k, "-s", "1",
             "-a", "hp_lehninger2", "--remove-low-complexity", *ext,
             "--ka-queries", "200", "--ka-survival-out", data_dir / f"ka_survival.k{k}.csv"],
            check=True,
        )  # fmt: skip
    subprocess.run(
        [KMERSEEK, "search", "-q", bhf, "-t", idx, "-k", k, "-a", "hp_lehninger2",
         "--threshold", "0", "--min-shared-kmers", "1", "--max-query-pvalue", "1",
         "--min-region-score", "0", *ext, "-o", data_dir / f"bhf.hp_lehninger2.k{k}.csv"],
        check=True,
    )  # fmt: skip
    version = subprocess.run(
        [KMERSEEK, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "-C", KMERSEEK.parents[2], "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    (data_dir / "kmerseek_version.txt").write_text(f"{version} ({commit})\n")
    print(f"kmerseek {version} ({commit}) outputs in {data_dir}")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------
HIT_COLS = [
    "tool",
    "arm",
    "query",
    "target",
    "qstart",
    "qend",
    "tstart",
    "tend",
    "score",
    "evalue",
]


def accession(name: str) -> str:
    """sp|P10415|BCL2_HUMAN -> P10415; a bare accession stays as it is."""
    return name.split("|")[1] if "|" in name else name.split()[0]


def read_tool(label: str, path: Path) -> pl.DataFrame:
    arm = TOOLS[label][0]
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(schema={c: pl.Utf8 for c in HIT_COLS})
    d = pl.read_csv(path, separator="\t", has_header=False, new_columns=HIT_COLS[2:],
                    infer_schema_length=None)  # fmt: skip
    return d.select(
        pl.lit(label).alias("tool"),
        pl.lit(arm).alias("arm"),
        pl.col("query").map_elements(accession, return_dtype=pl.Utf8),
        pl.col("target").map_elements(accession, return_dtype=pl.Utf8),
        *[pl.col(c).cast(pl.Int64) for c in ["qstart", "qend", "tstart", "tend"]],
        pl.col("score").cast(pl.Float64),
        pl.col("evalue").cast(pl.Float64),
    )


def read_kmerseek(data_dir: Path) -> pl.DataFrame:
    """The pair region (case A) and the E <= 10 search regions (case B), 1-based inclusive."""
    pj = json.loads((data_dir / "ced9_bcl2.hp_lehninger2.k17.pair.json").read_text())
    assert len(pj["regions"]) == 1, pj["regions"]
    r = pj["regions"][0]
    pair = pl.DataFrame(
        [dict(tool="kmerseek chosen arm", arm=CASE_A["kmerseek_arm"], query="P41958",
              target="P10415", qstart=r["query_start"] + 1, qend=r["query_end"],
              tstart=r["target_start"] + 1, tend=r["target_end"], score=None, evalue=None)]
    )  # fmt: skip
    s = pl.read_csv(
        data_dir / f"bhf.hp_lehninger2.k{CASE_B['k']}.csv", infer_schema_length=None
    )
    search = s.filter(pl.col("region_evalue") <= EVALUE_MAX).select(
        pl.lit("kmerseek chosen arm").alias("tool"),
        pl.lit(CASE_B["kmerseek_arm"]).alias("arm"),
        pl.lit("BHF").alias("query"),
        pl.col("target_name")
        .map_elements(accession, return_dtype=pl.Utf8)
        .alias("target"),
        (pl.col("region_start") + 1).alias("qstart"),
        pl.col("region_end").alias("qend"),
        (pl.col("target_start") + 1).alias("tstart"),
        pl.col("target_end").alias("tend"),
        pl.col("region_ka_bits").alias("score"),
        pl.col("region_evalue").alias("evalue"),
    )
    return pl.concat([pair, search], how="vertical_relaxed")


def rank_hits(d: pl.DataFrame) -> pl.DataFrame:
    """Order every (tool, query)'s hits and rank its human targets by their best hit:
    lowest E-value, then highest score. Ties left after that are broken by accession and
    start, so the order is the same on every run."""
    d = d.sort(["tool", "query", "evalue", "score", "target", "qstart", "tstart"],
               descending=[False, False, False, True, False, False, False],
               nulls_last=True, maintain_order=True)  # fmt: skip
    first = (
        d.group_by(["tool", "query", "target"], maintain_order=True)
        .head(1)
        .with_columns(
            pl.int_range(1, pl.len() + 1).over(["tool", "query"]).alias("target_rank")
        )
        .select("tool", "query", "target", "target_rank")
    )
    return d.join(
        first, on=["tool", "query", "target"], how="left", maintain_order="left"
    )


def collect(tools_dir: Path, data_dir: Path) -> pl.DataFrame:
    parts = [read_kmerseek(data_dir)]
    for label, (_arm, fname) in TOOLS.items():
        t = read_tool(label, tools_dir / fname)
        print(
            f"{label}: {t.height} hits ({', '.join(f'{q} {n}' for q, n in t.group_by('query').len().rows())})"
        )
        parts.append(t)
    d = rank_hits(pl.concat(parts, how="vertical_relaxed"))
    d.write_csv(HITS, separator="\t")
    print(f"wrote {HITS}: {d.height} rows")
    return d


# ---------------------------------------------------------------------------
# write
# ---------------------------------------------------------------------------
def interval_stats(s: int, e: int, fs: int, fe: int) -> tuple[float, float, float]:
    """IoU, share of the call inside the feature, share of the feature covered (1-based,
    inclusive on both)."""
    ov = max(0, min(e, fe) - max(s, fs) + 1)
    union = (e - s + 1) + (fe - fs + 1) - ov
    return ov / union, ov / (e - s + 1), ov / (fe - fs + 1)


def classify(n: int, iou: float, inside_half: bool, km_iou: float) -> str:
    """hero_example_utils.classify_comparison, for one instance."""
    if n == 0:
        return "no call"
    if iou >= km_iou:
        return "equal or higher IoU"
    return "inside, lower IoU" if inside_half else "spills"


def n_targets(h: pl.DataFrame) -> str:
    n = h["target"].n_unique()
    return f"{n} human target" + ("" if n == 1 else "s")


def lengths(accs: set[str]) -> dict[str, int]:
    return {a: len(s) for a, s in he.sequences("human", accs).items()}


def base_call(case: dict, tool: str, arm: str) -> dict:
    return dict(
        case_id=case["case_id"], gene=case["gene"], query=case["query"],
        feature_type=case["feature_type"], feature_start=case["feature_start"],
        feature_end=case["feature_end"], species=case["species"], target=case["target"],
        tool=tool, arm=arm, tier="bonus", query_species=case["query_species"],
        feature_on=case["feature_on"],
    )  # fmt: skip


def case_a_calls(hits: pl.DataFrame) -> tuple[list[dict], dict]:
    c = CASE_A
    fs, fe = c["feature_start"], c["feature_end"]
    tlen = lengths({c["target"]})[c["target"]]
    km = hits.filter(
        (pl.col("tool") == "kmerseek chosen arm") & (pl.col("query") == c["query"])
    )
    assert km.height == 1, km
    k = km.row(0, named=True)
    km_iou, km_inside, km_cover = interval_stats(k["tstart"], k["tend"], fs, fe)
    lands = km_inside >= he.INSIDE_MIN and km_cover >= he.COVER_MIN
    rows = [
        base_call(c, "kmerseek chosen arm", c["kmerseek_arm"])
        | dict(
            outcome="lands" if lands else "does not land",
            outcome_on_human="lands" if lands else "does not land",
            query_start=k["qstart"], query_end=k["qend"], target_start=k["tstart"],
            target_end=k["tend"], iou=round(km_iou, 3), target_protein_length=tlen,
            note="kmerseek pair region (no E-value); the whole-proteome search is in the PR",
        )
    ]  # fmt: skip
    summary = dict(
        kmerseek_chosen_arm=c["kmerseek_arm"], kmerseek_iou=round(km_iou, 3),
        frac_call_inside_feature=round(km_inside, 3), frac_feature_covered=round(km_cover, 3),
    )  # fmt: skip
    for label, (arm, _f) in TOOLS.items():
        h = hits.filter((pl.col("tool") == label) & (pl.col("query") == c["query"]))
        on_bcl2 = h.filter(pl.col("target") == c["target"])
        partner_rank = on_bcl2["target_rank"].min() if on_bcl2.height else None
        n_hit = h["target"].n_unique()
        scored = [
            (interval_stats(r["tstart"], r["tend"], fs, fe), r)
            for r in on_bcl2.iter_rows(named=True)
            if min(r["tend"], fe) >= max(r["tstart"], fs)
        ]
        if scored:
            (iou, inside, _cov), r = max(
                scored, key=lambda x: (x[0][0], -x[1]["evalue"])
            )
            cat = classify(
                len(scored), iou, any(s[0][1] >= 0.5 for s in scored), km_iou
            )
            call = dict(query_start=r["qstart"], query_end=r["qend"], target_start=r["tstart"],
                        target_end=r["tend"], iou=round(iou, 3), score=r["score"],
                        evalue=r["evalue"])  # fmt: skip
        else:
            cat, call = "no call", {}
        if not h.height:
            note = "no human hit reported"
        elif on_bcl2.height and not scored:
            spans = ", ".join(
                f"{a}-{b}" for a, b in on_bcl2.select("tstart", "tend").rows()
            )
            note = f"BCL2 is hit at {spans}, not on BH1"
        elif not on_bcl2.height:
            note = f"BCL2 not among its {n_targets(h)} reported"
        else:
            note = f"BCL2 is target {partner_rank} of {n_hit} by E-value"
        rows.append(
            base_call(c, label, arm)
            | dict(outcome=cat, outcome_on_human=cat, target_protein_length=tlen,
                   partner_rank=partner_rank, note=note, **call)
        )  # fmt: skip
        summary[f"{label}_iou"] = call.get("iou")
        summary[f"{label}_category"] = cat
    return rows, summary | {"_km": k}


def case_b_calls(hits: pl.DataFrame) -> tuple[list[dict], dict, set[str]]:
    c = CASE_B
    rows, summary, targets = [], {"kmerseek_chosen_arm": c["kmerseek_arm"]}, set()
    for label, arm in [("kmerseek chosen arm", c["kmerseek_arm"])] + [
        (lab, a) for lab, (a, _f) in TOOLS.items()
    ]:
        h = hits.filter((pl.col("tool") == label) & (pl.col("query") == c["query"]))
        row = base_call(c, label, arm)
        if label in STRUCTURE_TOOLS:
            row |= dict(outcome="no call", outcome_on_human="no call",
                        note="not run: BHF has no AlphaFold model")  # fmt: skip
        elif not h.height:
            row |= dict(outcome="no call", outcome_on_human="no call",
                        note="no human hit reported")  # fmt: skip
        else:
            r = h.row(0, named=True)  # rank_hits sorted it: lowest E-value first
            targets.add(r["target"])
            row |= dict(
                outcome="best human hit", outcome_on_human="best human hit",
                target=r["target"], query_start=r["qstart"], query_end=r["qend"],
                target_start=r["tstart"], target_end=r["tend"], score=r["score"],
                evalue=r["evalue"],
                note=f"{n_targets(h)} reported",
            )  # fmt: skip
        rows.append(row)
        if label in TOOLS:  # the candidate table has no category column for kmerseek
            summary[f"{label}_category"] = row["outcome"]
    tl = lengths(targets)
    for row in rows:
        row["target_protein_length"] = tl.get(row["target"]) if row["target"] else None
    return rows, summary, targets


def case_rows(summary_a: dict, summary_b: dict) -> list[dict]:
    a, b = CASE_A, CASE_B
    k = summary_a.pop("_km")
    q = he.sequences("worm", {a["query"]})[a["query"]][k["qstart"] - 1 : k["qend"]]
    t = he.sequences("human", {a["target"]})[a["target"]][k["tstart"] - 1 : k["tend"]]
    n_id = sum(x == y for x, y in zip(q, t))
    common = dict(tier="bonus", species="human")
    row_a = common | dict(
        gene=a["gene"], query=a["query"], feature_type=a["feature_type"],
        swissprot_description=a["swissprot_description"], feature_start=a["feature_start"],
        feature_end=a["feature_end"], feature_length_aa=a["feature_end"] - a["feature_start"] + 1,
        target=a["target"], n_identical=n_id, region_length_aa=len(q),
        identity_pct_region=round(100 * n_id / len(q), 1), query_species=a["query_species"],
        feature_on="target",
        note="Swiss-Prot Motif BH1 of BCL2 (UniProt P10415 v279); the feature is on the target",
    ) | summary_a  # fmt: skip
    row_b = common | dict(
        gene=b["gene"], query=b["query"], feature_type="NONE", query_species=b["query_species"],
        feature_on="none",
        note="BHF, record BHF of Bs_proteins.fa (notebook 241's query); no true feature",
    ) | summary_b  # fmt: skip
    return [row_a, row_b]


def rewrite_csv(path: Path, new_cols: list[str], rows: list[dict]) -> int:
    """Keep every non-bonus line of ``path`` byte for byte (plus empty cells for columns it
    did not have), drop old bonus lines, append ``rows``."""
    lines = path.read_text().splitlines()
    header = next(csv.reader([lines[0]]))
    missing = [c for c in new_cols if c not in header]
    out_header = header + missing
    kept = []
    for ln in lines[1:]:
        rec = dict(zip(header, next(csv.reader([ln]))))
        if rec.get("tier") == "bonus":
            continue
        kept.append(ln + "," * len(missing))
    unknown = {k for r in rows for k in r} - set(out_header)
    assert not unknown, unknown
    with open(path, "w", newline="") as fh:
        fh.write(",".join(out_header) + "\n")
        for ln in kept:
            fh.write(ln + "\n")
        w = csv.writer(fh, lineterminator="\n")
        for r in rows:
            w.writerow(["" if r.get(c) is None else r[c] for c in out_header])
    return len(kept)


def rewrite_fasta(path: Path, records: list[tuple[str, str, str]]) -> int:
    """Drop records tagged tier=bonus, append ``records`` (accession, header rest, seq)
    tagged tier=bonus, skipping accessions the file already has."""
    blocks, cur = [], None
    for line in path.read_text().splitlines(keepends=True):
        if line.startswith(">"):
            cur = [line]
            blocks.append(cur)
        else:
            cur.append(line)
    blocks = [b for b in blocks if "tier=bonus" not in b[0]]
    have = {b[0][1:].split()[0] for b in blocks}
    n = 0
    with open(path, "w") as fh:
        for b in blocks:
            fh.write("".join(b))
        for acc, rest, seq in records:
            if acc in have:
                continue
            have.add(acc)
            fh.write(f">{acc} {rest} tier=bonus\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i : i + 60] + "\n")
            n += 1
    return n


def write_bonus(hits_path: Path = HITS) -> None:
    hits = pl.read_csv(hits_path, separator="\t", infer_schema_length=None)
    calls_a, sum_a = case_a_calls(hits)
    calls_b, sum_b, b_targets = case_b_calls(hits)
    cases = case_rows(sum_a, sum_b)
    n = rewrite_csv(TAB / "244_hero_candidates.csv", NEW_CASE_COLS, cases)
    print(f"244_hero_candidates.csv: {n} rows kept, {len(cases)} bonus rows")
    n = rewrite_csv(TAB / "244_case_calls.csv", NEW_CALL_COLS, calls_a + calls_b)
    print(
        f"244_case_calls.csv: {n} rows kept, {len(calls_a) + len(calls_b)} bonus rows"
    )

    human_accs = {"P10415"} | b_targets
    human = he.sequences("human", human_accs)
    assert (
        set(human) == human_accs
    ), f"not in the QfO human proteome: {human_accs - set(human)}"
    ced9, bhf = query_records()
    records = [ced9]
    records += [
        (a, f"species=human {he.protein_name('human', a)}", human[a])
        for a in sorted(human_accs)
    ]
    records.append(bhf)
    n = rewrite_fasta(TAB / "244_case_sequences.fasta", records)
    print(f"244_case_sequences.fasta: {n} bonus sequences")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "step", nargs="?", default="write", choices=["kmerseek", "collect", "write"]
    )
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--tools-dir", type=Path, help="copy of $SCRATCH/244-bonus/out")
    args = ap.parse_args()
    if args.step == "kmerseek":
        run_kmerseek(args.data_dir)
    elif args.step == "collect":
        if args.tools_dir is None:
            ap.error("collect needs --tools-dir")
        collect(args.tools_dir, args.data_dir)
    else:
        write_bonus()


if __name__ == "__main__":
    main()
