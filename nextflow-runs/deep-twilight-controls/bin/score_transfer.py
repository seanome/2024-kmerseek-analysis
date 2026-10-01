#!/usr/bin/env python3
"""Score whether each tool carries each family protein's functional label onto the others.

A call is one aligned region from one tool: a kmerseek region (ungapped), a phmmer domain,
an MMseqs2 or a Foldseek alignment. Every call is reduced to its list of aligned residue
pairs (query position, target position), 1-based as in UniProt.

For one tool (or one kmerseek alphabet and k), one ordered pair of family proteins and one
label the query carries, with Q the query's labelled residues and T the target's:

  correct          a call with E <= max_evalue places a residue of Q on a residue of T
  misplaced        calls place residues of Q on the target, none of them on T
  no call          no call places any residue of Q on the target
  carried wrongly  T is empty (the target has no such label in Swiss-Prot) and a call places
                   a residue of Q on the target anyway
  not carried      T is empty, the tool has a call on this pair, and none places a
                   residue of Q on the target
  (a pair with no call at all is "no call" whether or not the target has the label)

For an ungapped kmerseek call that covers a residue of Q, placement_null is the share of
positions the same-length window could take on the target at which a residue of Q would
land on T: the chance that the window's placement alone gets the label right.

    score_transfer.py --labels pairs.tsv --family family.fasta --arms arms.tsv \
        --identity identity.tsv --max-evalue 1000 --kmerseek k/*.parquet \
        --pairs p/*.tsv --baselines b/*.tsv --outdir .
"""

import argparse
from pathlib import Path

import polars as pl

ap = argparse.ArgumentParser()
ap.add_argument("--labels", required=True)
ap.add_argument("--family", required=True)
ap.add_argument("--arms", required=True)
ap.add_argument("--identity", required=True)
ap.add_argument("--max-evalue", type=float, required=True)
ap.add_argument("--kmerseek", nargs="*", default=[])
ap.add_argument("--pairs", nargs="*", default=[])
ap.add_argument("--baselines", nargs="*", default=[])
ap.add_argument("--outdir", default=".")
args = ap.parse_args()
outdir = Path(args.outdir)

# --- family proteins and labels ---------------------------------------------------------
seqs, h = {}, None
for line in open(args.family):
    line = line.rstrip("\n")
    if line.startswith(">"):
        h = line[1:].split()[0].split("|")[1]
        seqs[h] = ""
    elif h:
        seqs[h] += line.strip()

labels = pl.read_csv(args.labels, separator="\t", infer_schema_length=0)
family_of = dict(zip(labels["accession"], labels["family"]))
label_pos = {}  # (accession, label) -> set of 1-based positions
for r in labels.iter_rows(named=True):
    label_pos[(r["accession"], r["label"])] = {
        int(p) for p in r["positions"].split(",")
    }
query_labels = {}
for a, lab in label_pos:
    query_labels.setdefault(a, []).append(lab)

ordered_pairs = [
    (q, t) for q in seqs for t in seqs if q != t and family_of[q] == family_of[t]
]

# --- calls ------------------------------------------------------------------------------
calls = []
for p in args.kmerseek:
    k = pl.read_parquet(p)
    if k.height == 0:
        continue
    calls.append(
        k.select(
            tool=pl.lit("kmerseek"),
            alphabet=pl.col("alphabet"),
            ksize=pl.col("ksize").cast(pl.Int64),
            query=pl.col("query"),
            target=pl.col("target"),
            evalue=pl.col("region_evalue"),
            evalue_source=pl.col("region_evalue_source"),
            ka_evalue=pl.col("region_ka_evalue"),
            run_evalue=pl.col("region_run_evalue"),
            mean_idf=pl.col("region_mean_idf"),
            n_mismatches=pl.col("region_n_mismatches").cast(pl.Int64),
            extended=pl.col("extended"),
            qstart=(pl.col("region_start") + 1).cast(pl.Int64),
            qend=pl.col("region_end").cast(pl.Int64),
            tstart=(pl.col("target_start") + 1).cast(pl.Int64),
            tend=pl.col("target_end").cast(pl.Int64),
            qaln=pl.col("region_subseq"),
            taln=pl.col("target_subseq"),
            query_classes=pl.col("moltype_seq"),
        )
    )
for p in args.baselines:
    b = pl.read_csv(p, separator="\t", infer_schema_length=0)
    if b.height == 0:
        continue
    calls.append(
        b.select(
            tool=pl.col("tool"),
            alphabet=pl.lit(None, pl.String),
            ksize=pl.lit(None, pl.Int64),
            query=pl.col("query"),
            target=pl.col("target"),
            evalue=pl.col("evalue").cast(pl.Float64),
            evalue_source=pl.lit(None, pl.String),
            ka_evalue=pl.lit(None, pl.Float64),
            run_evalue=pl.lit(None, pl.Float64),
            mean_idf=pl.lit(None, pl.Float64),
            n_mismatches=pl.lit(None, pl.Int64),
            extended=pl.lit(None, pl.Boolean),
            qstart=pl.col("qstart").cast(pl.Int64),
            qend=pl.col("qend").cast(pl.Int64),
            tstart=pl.col("tstart").cast(pl.Int64),
            tend=pl.col("tend").cast(pl.Int64),
            qaln=pl.col("qaln"),
            taln=pl.col("taln"),
            query_classes=pl.lit(None, pl.String),
        )
    )
calls = pl.concat(calls, how="vertical_relaxed").with_row_index("call_id")


def aligned_pairs(qstart, tstart, qaln, taln):
    q, t, out = qstart - 1, tstart - 1, []
    for a, b in zip(qaln, taln):
        if a != "-":
            q += 1
        if b != "-":
            t += 1
        if a != "-" and b != "-":
            out.append((q, t))
    return out


rows_calls = []
for r in calls.iter_rows(named=True):
    pairs = aligned_pairs(r["qstart"], r["tstart"], r["qaln"], r["taln"])
    qs, ts = seqs[r["query"]], seqs[r["target"]]
    n_ident = sum(qs[a - 1] == ts[b - 1] for a, b in pairs)
    # the aligned strings must be the proteins' own residues at the stated coordinates
    assert (
        r["qaln"].replace("-", "")
        == qs[r["qstart"] - 1 : r["qstart"] - 1 + len(r["qaln"].replace("-", ""))]
    ), (r["tool"], r["query"], r["target"], r["qstart"])
    assert (
        r["taln"].replace("-", "")
        == ts[r["tstart"] - 1 : r["tstart"] - 1 + len(r["taln"].replace("-", ""))]
    ), (r["tool"], r["query"], r["target"], r["tstart"])
    rows_calls.append(
        dict(call_id=r["call_id"], n_aligned=len(pairs), n_identical=n_ident)
    )
calls = calls.join(pl.DataFrame(rows_calls), on="call_id")
calls.write_parquet(outdir / "calls.parquet")

# --- outcomes ---------------------------------------------------------------------------
# Only the arms that produced an output file are scored; an arm with no file (a task that
# never finished) would otherwise show as "no call" on every pair.
arms = pl.read_csv(args.arms, separator="\t")
ran = set()
for p in args.kmerseek:
    alphabet, k = Path(p).name.removesuffix(".hits.parquet").rsplit(".k", 1)
    ran.add((alphabet, int(k)))
missing = [(a, k) for a, k in zip(arms["alphabet"], arms["ksize"]) if (a, k) not in ran]
print(f"{len(ran)} of {arms.height} kmerseek arms have results; missing: {missing}")
arm_keys = [
    ("kmerseek", a, k) for a, k in zip(arms["alphabet"], arms["ksize"]) if (a, k) in ran
] + [(t, None, None) for t in ("phmmer", "mmseqs2", "foldseek")]
passing = calls.filter(pl.col("evalue") <= args.max_evalue)
by_arm_pair = {}
for r in passing.iter_rows(named=True):
    key = (r["tool"], r["alphabet"], r["ksize"], r["query"], r["target"])
    by_arm_pair.setdefault(key, []).append(r)

out = []
for tool, alphabet, ksize in arm_keys:
    for q, t in ordered_pairs:
        for lab in query_labels[q]:
            Q = label_pos[(q, lab)]
            T = label_pos.get((t, lab), set())
            best = None  # (rank, evalue, call row, placement_null)
            for r in by_arm_pair.get((tool, alphabet, ksize, q, t), []):
                pairs = aligned_pairs(r["qstart"], r["tstart"], r["qaln"], r["taln"])
                landed = [(a, b) for a, b in pairs if a in Q]
                if not landed:
                    continue
                hit = any(b in T for a, b in landed)
                rank = 0 if hit else 1
                null = None
                if tool == "kmerseek" and T:
                    L, n = len(pairs), len(seqs[t])
                    offsets = [
                        a - r["qstart"] for a in Q if r["qstart"] <= a <= r["qend"]
                    ]
                    good = sum(
                        any(s + o in T for o in offsets) for s in range(1, n - L + 2)
                    )
                    null = good / (n - L + 1)
                cand = (rank, r["evalue"], r, null)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if T:
                outcome = (
                    "no call"
                    if best is None
                    else ("correct" if best[0] == 0 else "misplaced")
                )
            elif best is not None:
                outcome = "carried wrongly"
            else:
                # "not carried" needs a call on the pair that leaves the label off; with
                # no call on the pair at all the tool has not been tested on it
                on_pair = bool(by_arm_pair.get((tool, alphabet, ksize, q, t)))
                outcome = "not carried" if on_pair else "no call"
            r = best[2] if best else None
            out.append(
                dict(
                    tool=tool,
                    alphabet=alphabet,
                    ksize=ksize,
                    family=family_of[q],
                    query=q,
                    target=t,
                    label=lab,
                    target_has_label=bool(T),
                    outcome=outcome,
                    call_id=r["call_id"] if r else None,
                    call_evalue=r["evalue"] if r else None,
                    placement_null=best[3] if best else None,
                )
            )
outcomes = pl.DataFrame(
    out,
    schema_overrides=dict(
        alphabet=pl.String,
        ksize=pl.Int64,
        call_id=pl.UInt32,
        call_evalue=pl.Float64,
        placement_null=pl.Float64,
    ),
)

identity = pl.read_csv(args.identity, separator="\t")
both = pl.concat(
    [
        identity.rename({"a": "query", "b": "target"}),
        identity.rename({"a": "target", "b": "query"}),
    ],
    how="diagonal",
)
outcomes = outcomes.join(both, on=["query", "target"], how="left")
assert outcomes["needle_identity_pct"].null_count() == 0
outcomes.write_parquet(outdir / "outcomes.parquet")

if args.pairs:
    pl.concat([pl.read_csv(p, separator="\t") for p in args.pairs]).write_parquet(
        outdir / "pair_kmers.parquet"
    )

print(outcomes.group_by("tool", "outcome").len().sort("tool", "outcome"))
