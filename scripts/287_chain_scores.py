"""Notebook 287, part 1: chain scores for notebook 241's three queries against the human
proteome, and where BCL2 (query Ced9) and CD47 (query P66) rank.

Input: notebook 241's regions (kmerseek 982a055, 152 settings, every region kept),
/Users/olga/data/botryllus/alphabet-ranking-three-cases/{regions.parquet,arms.csv,queries.fa},
and the GENCODE v49 canonical human proteome (residues for the chain table).

N for each setting (k-mers in the human index) is rebuilt from the calls themselves:
E = K m N exp(-lambda S), so ln N = ln E + lambda S - ln K - ln m, with K from arms.csv. The
script stops if any setting's calls disagree on N by more than 1e-6 in ln N.

Variants (notebook 287's rules): across settings or within one; each call's E-value times
the number of settings with an E-value fit (71, from notebooks/287_settings_N_human_index_logs.csv,
made by scripts/287_settings_n_from_index_logs.py) or no factor; maximum gap 200 residues (50, 100 and 400 for the main score only).

Also caches HGNC gene group 1277 and UniProt Q08722 and H7C7N8 under 287/ (fetch_reference).

Outputs:
  /Users/olga/data/botryllus/alphabet-ranking-three-cases/287/chain_scores.parquet
      one row per query, variant and human protein (about 19_000 x 3 x 7)
  notebooks/287_partner_rank_human.csv     BCL2 and CD47 under every variant
  notebooks/287_chance_hits_human.csv      proteins at E < 1 outside the known family, per search
  notebooks/287_chain_members_human.csv    the calls in the best chains of BCL2, CD47 and every BCL-2
                                           family member with a chain of two or more calls, with residues
  notebooks/287_settings_N_human.csv       N per setting, and the three checked against index logs
  notebooks/287_named_chain_human.csv      Ced9's BH4 and BH1 calls on BCL2 as one chain, each variant
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import sum_statistics_utils as ss  # noqa: E402

D = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
OUT = D / "287"
HUMAN = Path(
    "/Users/olga/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
)
PARTNER = {"Ced9": "BCL2", "P66": "CD47", "BHF": None}
# HGNC gene groups 1057 (BCL2 family) and 1277 (BH3-only), fetched from rest.genenames.org
# 2026-10-13; the response for 1277 is saved in 287/hgnc_group_1277.json.
BCL2_FAMILY = {
    "BAK1",
    "BAX",
    "BCL2",
    "BCL2A1",
    "BCL2L1",
    "BCL2L10",
    "BCL2L12",
    "BCL2L13",
    "BCL2L14",
    "BCL2L15",
    "BCL2L2",
    "BNIP2",
    "BOK",
    "MCL1",
    "BAD",
    "BBC3",
    "BCL2L11",
    "BID",
    "BIK",
    "BMF",
    "BNIP1",
    "BNIP3",
    "HRK",
    "PMAIP1",
    "RTL10",
}
KNOWN = {"Ced9": BCL2_FAMILY, "P66": {"CD47"}, "BHF": set()}
# "total k-mers found" in notebook 241's index logs on Sherlock
# ($SCRATCH/241-alphabet-ranking/logs/<setting>.index.log), read 2026-10-13
N_FROM_LOGS = {
    ("dayhoff6", 7): 10_513_245,
    ("dayhoff6", 9): 10_862_770,
    ("hp_thomas_dill2", 21): 10_676_958,
}
MAX_GAP = 200
OTHER_GAPS = [50, 100, 400]
# The two calls notebook 284 found in register with UniProt motifs, scored as one chain by
# name whether or not the search picks them: (setting, Ced9 start, BCL2 start), 0-based.
NAMED_CHAIN = (
    "Ced9",
    "BCL2",
    "BH4 + BH1",
    [("wwmj5.k9", 77, 8), ("hp_lehninger2.k19", 162, 138)],
)


def read_fasta(p: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(p):
        if line.startswith(">"):
            name = line[1:].strip()
            out[name] = []
        else:
            out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def fetch_reference() -> None:
    """Cache what the notebook reads from servers (fetched 2026-10-13): HGNC gene group
    1277 (BH3-only) and UniProt Q08722 (CD47) and H7C7N8 (P66). A cached file is reused.
    """
    import urllib.request

    (OUT / "uniprot").mkdir(parents=True, exist_ok=True)
    wanted = {
        OUT
        / "hgnc_group_1277.json": (
            "https://rest.genenames.org/search/gene_group_id/1277",
            {"Accept": "application/json"},
        ),
        OUT
        / "uniprot/Q08722.json": ("https://rest.uniprot.org/uniprotkb/Q08722.json", {}),
        OUT
        / "uniprot/H7C7N8.json": ("https://rest.uniprot.org/uniprotkb/H7C7N8.json", {}),
    }
    for path, (url, headers) in wanted.items():
        if path.exists():
            continue
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
            body = r.read()
        json.loads(body)  # an outage page is not JSON: fail here, not in the notebook
        path.write_bytes(body)


def load_calls(q_len: dict[str, int]) -> tuple[pl.DataFrame, pl.DataFrame, int]:
    arms = pl.read_csv(D / "arms.csv").select(
        "alphabet", pl.col("ksize").alias("ksize_arm"), "penalty", "ka_k"
    )
    r = (
        pl.scan_parquet(D / "regions.parquet")
        .filter(pl.col("region_evalue").is_finite())
        .join(arms.lazy(), on=["alphabet", "ksize_arm"])
        .with_columns(
            m=pl.col("query_name").replace_strict(q_len, return_dtype=pl.Float64),
            S=(pl.col("region_length") - pl.col("region_n_mismatches"))
            - pl.col("penalty") * pl.col("region_n_mismatches"),
            n_t=pl.col("target_name").str.split("|").list.last().cast(pl.Float64),
            setting=pl.col("alphabet") + ".k" + pl.col("ksize_arm").cast(pl.String),
        )
        .with_columns(
            ln_n=pl.col("region_evalue").log()
            + pl.col("region_ka_lambda") * pl.col("S")
            - pl.col("ka_k").log()
            - pl.col("m").log()
        )
        .collect()
    )
    n = (
        r.group_by("alphabet", "ksize_arm", "setting")
        .agg(
            n_calls=pl.len(),
            ln_n_min=pl.col("ln_n").min(),
            ln_n_max=pl.col("ln_n").max(),
        )
        .with_columns(n_db=pl.col("ln_n_min").exp().round(0))
        .sort("setting")
    )
    spread = (n["ln_n_max"] - n["ln_n_min"]).max()
    if spread > 1e-6:
        sys.exit(f"N differs within a setting by {spread} in ln N")
    for (a, k), want in N_FROM_LOGS.items():
        got = n.filter((pl.col("alphabet") == a) & (pl.col("ksize_arm") == k))["n_db"][
            0
        ]
        if abs(got - want) > 1:
            sys.exit(f"N for {a} k{k}: rebuilt {got}, index log {want}")
    n = n.with_columns(
        n_from_index_log=pl.struct("alphabet", "ksize_arm").map_elements(
            lambda s: N_FROM_LOGS.get((s["alphabet"], s["ksize_arm"])),
            return_dtype=pl.Int64,
        )
    )
    calls = r.join(n.select("setting", "n_db"), on="setting")
    return calls, n, n.height


def rank_table(s: pl.DataFrame) -> pl.DataFrame:
    """Ranks of every protein by chain E and by single E, lowest first; ties share the
    best rank (rank 'min') and n_tied counts them."""
    return s.with_columns(
        rank_chain=pl.col("chain_log_e").rank("min"),
        rank_single=pl.col("single_log_e").rank("min"),
        n_tied_chain=pl.len().over("chain_log_e"),
        n_tied_single=pl.len().over("single_log_e"),
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    fetch_reference()
    queries = read_fasta(D / "queries.fa")
    human = read_fasta(HUMAN)
    n_proteome = len(human)
    q_len = {k: len(v) for k, v in queries.items()}
    calls, n_tab, n_settings = load_calls(q_len)
    n_tab.drop("ln_n_min", "ln_n_max").write_csv(
        ROOT / "notebooks/287_settings_N_human.csv"
    )
    print(
        f"{calls.height} calls with an E-value, {n_settings} settings, proteome {n_proteome}"
    )
    calls = ss.add_normalised_score(calls).collect()
    # the search-space factor counts every setting with an E-value fit (71), not only those
    # that gave one of these three queries a finite E-value (69): a shuffled copy can reach
    # the other two
    fit = pl.read_csv(ROOT / "notebooks/287_settings_N_human_index_logs.csv").filter(
        pl.col("fitted")
    )
    n_settings = fit.height
    pen = math.log(n_settings)
    x = f"x{n_settings}"
    variants = [
        (f"across settings, {x}", ss.ChainRules(MAX_GAP, True, pen)),
        ("across settings, no factor", ss.ChainRules(MAX_GAP, True, 0.0)),
        (f"one setting, {x}", ss.ChainRules(MAX_GAP, False, pen)),
        ("one setting, no factor", ss.ChainRules(MAX_GAP, False, 0.0)),
    ] + [
        (f"across settings, {x}, gap {g}", ss.ChainRules(g, True, pen))
        for g in OTHER_GAPS
    ]
    tables, members = [], []
    for q in PARTNER:
        cq = calls.filter(pl.col("query_name") == q)
        for name, rules in variants:
            t0 = time.time()
            s = ss.score_pairs(cq, rules).with_columns(
                gene=pl.col("target_name").str.split("|").list.get(6),
                variant=pl.lit(name),
                max_gap=pl.lit(rules.max_gap),
                across_settings=pl.lit(rules.across_settings),
                settings_factor=pl.lit(round(math.exp(rules.settings_penalty))),
            )
            s = rank_table(s)
            tables.append(s)
            print(
                f"{q} {name}: {s.height} proteins, {time.time() - t0:.0f} s", flush=True
            )
            p = PARTNER[q]
            # the partner's best chain always; for Ced9 also every BCL-2 family member whose
            # best chain has two or more calls
            show = s.filter(
                (pl.col("gene") == p)
                | (pl.col("gene").is_in(list(KNOWN[q])) & (pl.col("chain_r") > 1))
            )
            for row in show.iter_rows(named=True):
                for i, ri in enumerate(row["chain_rows"]):
                    c = (
                        cq.with_row_index("row")
                        .filter(pl.col("row") == ri)
                        .row(0, named=True)
                    )
                    members.append(
                        dict(
                            query=q,
                            partner=row["gene"],
                            variant=name,
                            member=i + 1,
                            setting=c["setting"],
                            query_start=c["qs"] + 1,
                            query_end=c["qe"],
                            target_start=c["ts"] + 1,
                            target_end=c["te"],
                            offset=c["ts"] - c["qs"],
                            evalue=c["region_evalue"],
                            y=c["y"],
                            query_residues=queries[q][c["qs"] : c["qe"]],
                            target_residues=human[c["target_name"]][c["ts"] : c["te"]],
                        )
                    )
    allt = pl.concat(tables, how="diagonal_relaxed")
    allt.write_parquet(OUT / "chain_scores.parquet")
    keys = ["query", "variant", "max_gap", "across_settings", "settings_factor"]
    allt = allt.rename({"query_name": "query"})
    partner_rows = []
    for q, p in PARTNER.items():
        if not p:
            continue
        for row in allt.filter(
            (pl.col("query") == q) & (pl.col("gene") == p)
        ).iter_rows(named=True):
            v = allt.filter(
                (pl.col("query") == q) & (pl.col("variant") == row["variant"])
            )
            partner_rows.append(
                {k: row[k] for k in keys}
                | dict(
                    partner=p,
                    n_proteome=n_proteome,
                    n_proteins_with_a_call=allt.filter(
                        (pl.col("query") == q) & (pl.col("variant") == row["variant"])
                    ).height,
                    chain_e=row["chain_e"],
                    chain_r=row["chain_r"],
                    rank_chain=row["rank_chain"],
                    n_tied_chain=row["n_tied_chain"],
                    single_e=row["single_e"],
                    rank_single=row["rank_single"],
                    n_tied_single=row["n_tied_single"],
                    chain_e_of_rank_20=v.sort("chain_e")["chain_e"][19],
                    single_e_of_rank_20=v.sort("single_e")["single_e"][19],
                )
            )
    pl.DataFrame(partner_rows).write_csv(ROOT / "notebooks/287_partner_rank_human.csv")
    chance = (
        allt.with_columns(
            known=pl.struct("query", "gene").map_elements(
                lambda s: s["gene"] in KNOWN[s["query"]], return_dtype=pl.Boolean
            )
        )
        .group_by(keys)
        .agg(
            n_chain_e_below_1=(pl.col("chain_e") < 1).sum(),
            n_chain_e_below_1_outside_family=(
                (pl.col("chain_e") < 1) & ~pl.col("known")
            ).sum(),
            n_single_e_below_1=(pl.col("single_e") < 1).sum(),
            n_single_e_below_1_outside_family=(
                (pl.col("single_e") < 1) & ~pl.col("known")
            ).sum(),
            genes_chain_e_below_1=pl.col("gene")
            .filter(pl.col("chain_e") < 1)
            .sort()
            .str.join(";"),
        )
        .sort(keys)
    )
    chance.write_csv(ROOT / "notebooks/287_chance_hits_human.csv")
    pl.DataFrame(members).write_csv(ROOT / "notebooks/287_chain_members_human.csv")
    q, p, label, parts = NAMED_CHAIN
    named = []
    for setting, qs0, ts0 in parts:
        c = calls.filter(
            (pl.col("query_name") == q)
            & (pl.col("gene") == p)
            & (pl.col("setting") == setting)
            & (pl.col("qs") == qs0)
            & (pl.col("ts") == ts0)
        )
        if c.height != 1:
            sys.exit(f"named call {setting} {qs0} {ts0}: {c.height} rows")
        named.append(c.row(0, named=True))
    n_t = named[0]["n_t"]
    n_db = max(c["n_db"] for c in named)
    rows = []
    for name, rules in variants[:4]:
        ys = [c["y"] - rules.settings_penalty for c in named]
        rows.append(
            dict(
                query=q,
                partner=p,
                chain=label,
                variant=name,
                settings_factor=round(math.exp(rules.settings_penalty)),
                y_bh4=ys[0],
                y_bh1=ys[1],
                e_bh4=math.exp(-ys[0]) * named[0]["n_db"] / n_t,
                e_bh1=math.exp(-ys[1]) * named[1]["n_db"] / n_t,
                chain_e=ss.chain_evalue(sum(ys), 2, n_db, n_t),
            )
        )
    pl.DataFrame(rows).write_csv(ROOT / "notebooks/287_named_chain_human.csv")
    print(pl.DataFrame(rows).to_pandas().to_string())
    (OUT / "chain_scores.run.json").write_text(
        json.dumps(dict(n_settings=n_settings, n_proteome=n_proteome))
    )
    print(pl.DataFrame(partner_rows).to_pandas().to_string())
    print(chance.drop("genes_chain_e_below_1").to_pandas().to_string())


if __name__ == "__main__":
    main()
