"""Fill report.md's results sections from the scored run.

Keeps report.md up to "## Results under the preregistered rule" (the header and the
changes from the plan, written by hand) and rewrites everything after it.

Usage: python write_report.py <analysis dir> <run dir>
  <run dir> is the copy of $SCRATCH/351_short_feature_boundaries (db/, ledger/, kmerseek/)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CUT = "## Results under the preregistered rule"
EXT = "_extend2"
SEED = 351


def fmt(x, pct=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float) and np.isinf(x):
        return "infinite"
    return f"{x:.1%}" if pct else (f"{x:.3g}" if isinstance(x, float) else str(x))


def md_table(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for row in df.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                fmt(x) if not isinstance(x, (int, np.integer)) else str(x) for x in row
            )
            + " |"
        )
    return "\n".join(lines)


def verdict_table(v):
    rows = [
        "| setting | a: median error (<= 5) | b: within 10 (>= 80%) | c: shuffles placed (<= 5%) | d: profile / kmerseek median (>= 3) | all four |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in v["settings"].items():
        if not s["ran"]:
            rows.append(f"| {name} | not run | | | | |")
            continue
        mark = lambda ok: "pass" if ok else "fail"  # noqa: E731
        rows.append(
            f"| {name} | {fmt(s['a_median_error'])} ({mark(s['a_pass'])}) | {fmt(s['b_within_10'], True)} ({mark(s['b_pass'])}) "
            f"| {fmt(s['c_shuffle_rate'], True)} ({mark(s['c_pass'])}) | {fmt(s['d_profile_over_kmerseek'])} ({mark(s['d_pass'])}) "
            f"| {'yes' if s['all_four'] else 'no'} |"
        )
    return "\n".join(rows)


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def aligned_block(q_name, q_start, q_res, t_name, t_start, t_res):
    """Both stretches, their coordinates, a match line, and the identity count."""
    q_end = q_start + sum(c != "-" for c in q_res) - 1
    t_end = t_start + sum(c != "-" for c in t_res) - 1
    match = "".join("|" if a == b and a != "-" else " " for a, b in zip(q_res, t_res))
    same = match.count("|")
    w = max(len(q_name), len(t_name))
    return (
        f"{q_name:<{w}} {q_start:>5} {q_res} {q_end}\n"
        f"{'':<{w}} {'':>5} {match}\n"
        f"{t_name:<{w}} {t_start:>5} {t_res} {t_end}\n"
        f"{same} of {len(q_res)} aligned positions identical"
    )


def example(kind, r, err, pl, seqs, best_k):
    lines = [f"### {kind}: {r.instance_id}, {r.elm_class} on {r.protein}", ""]
    seq = seqs[r.protein]
    lines.append(
        f"True feature: residues {r.true_start}-{r.true_end} of {r.protein} ({len(seq)} aa): "
        f"`{seq[r.true_start - 1:r.true_end]}`"
    )
    lines.append("")
    for setting in [best_k, "pfam_hmmscan", "blastp"]:
        e = err[
            (err["instance_id"] == r.instance_id) & (err["setting"] == setting)
        ].iloc[0]
        if pd.isna(e["placed_start"]):
            lines.append(f"**{setting}**: no placement passing E <= 0.01.\n")
            continue
        s, t = int(e["placed_start"]), int(e["placed_end"])
        lines.append(
            f"**{setting}**: placed {s}-{t} (E = {e['evalue']:.2g}); start error {int(e['start_error'])}, "
            f"end error {int(e['end_error'])} residues."
        )
        p = pl[
            (pl["setting"] == setting)
            & (pl["query"] == r.protein)
            & (pl["start"] == s)
            & (pl["end"] == t)
            & (pl["target"] == e["target"])
        ]
        if (
            setting == "pfam_hmmscan"
            or p.empty
            or not isinstance(p.iloc[0]["query_residues"], str)
            or not p.iloc[0]["query_residues"]
        ):
            lines.append(
                f"The profile envelope ({e['target']}) has no partner sequence. Query residues {s}-{t}:"
            )
            lines += ["```", seq[s - 1 : t], "```", ""]
            continue
        p = p.iloc[0]
        block = aligned_block(
            r.protein,
            s,
            p["query_residues"],
            str(p["target"]),
            int(p["target_start"]),
            p["target_residues"],
        )
        lines += ["```", block, "```", ""]
    return "\n".join(lines)


def worked_examples(err, pl, seqs, metrics):
    ext = metrics[metrics["setting"].str.endswith(EXT)]
    ran = ext[ext["n_placed"] > 0]
    if ran.empty:
        return "No kmerseek run placed any feature, so there is no example to show."
    best_k = ran.sort_values("median_error").iloc[0]["setting"]
    piv = {}
    for s in [best_k, "pfam_hmmscan", "blastp"]:
        e = err[err["setting"] == s].set_index("instance_id")
        piv[s] = e[["start_error", "end_error"]].max(axis=1)
    df = pd.DataFrame(piv).dropna()
    df = df[np.isfinite(df).all(axis=1)]
    df["other"] = df[["pfam_hmmscan", "blastp"]].min(axis=1)
    kinds = {
        "kmerseek closer by 5 or more residues": df[df[best_k] + 5 <= df["other"]],
        "tie (within 1 residue)": df[(df[best_k] - df["other"]).abs() <= 1],
        "kmerseek further by 5 or more residues": df[df[best_k] >= df["other"] + 5],
    }
    rng = np.random.default_rng(SEED)
    info = err[err["setting"] == best_k].set_index("instance_id")
    out = [
        f"Each example compares {best_k} (the extended kmerseek run with the lowest median error) "
        "with hmmscan and blastp. Error = the larger of the start and end errors. Examples were "
        f"drawn at random (seed {SEED}) among instances that all three placed.",
        "",
    ]
    for kind, sub in kinds.items():
        if sub.empty:
            out.append(f"### {kind}\n\nNo instance of this kind.\n")
            continue
        iid = sub.index[rng.integers(len(sub))]
        r = info.loc[iid]
        r = pd.Series({**r.to_dict(), "instance_id": iid})
        out.append(example(kind, r, err, pl, seqs, best_k))
    return "\n".join(out)


def compute_section(run):
    files = sorted((run / "ledger").glob("*.tsv"))
    if not files:
        return "No ledger found."
    led = pd.concat(
        [
            pd.read_csv(
                f,
                sep="\t",
                header=None,
                names=["job", "task", "step", "seconds", "cpus", "status"],
            )
            for f in files
        ]
    )
    led["step_group"] = led["step"].str.replace(
        r"^(kmerseek|index)-.*", r"\1", regex=True
    )
    by = led.groupby("step_group").agg(
        tasks=("seconds", "size"),
        hours=("seconds", lambda s: s.sum() / 3600),
        cpu_hours=("seconds", lambda s: 0),
    )
    by["cpu_hours"] = (
        led.assign(ch=led["seconds"] * led["cpus"] / 3600)
        .groupby("step_group")["ch"]
        .sum()
    )
    total = led["seconds"].sum() / 3600
    lines = [
        "| step | tasks | wall time (h) | CPU time (h, wall x CPUs asked) |",
        "|---|---|---|---|",
    ]
    for k, row in by.iterrows():
        lines.append(f"| {k} | {row.tasks} | {row.hours:.2f} | {row.cpu_hours:.1f} |")
    lines.append(
        f"\nTotal wall time summed over all {len(led)} jobs and tasks: {total:.2f} h "
        f"(budget 48 h: {'within' if total <= 48 else 'OVER'}). Failed tasks (non-zero status): "
        f"{int((led['status'] != 0).sum())}. No job asked for a GPU; each refused to run if given one."
    )
    return "\n".join(lines)


def main(analysis_dir, run_dir):
    a, run = Path(analysis_dir), Path(run_dir)
    v = json.load(open(a / "results/verdict.json"))
    metrics = pd.read_csv(a / "results/metrics.csv")
    err = pd.read_csv(a / "results/boundary_errors.csv")
    pl = pd.read_csv(a / "results/placements.csv.gz", low_memory=False)
    seqs = read_fasta(a / "data/filtered/query_proteins.fasta")
    head = (a / "report.md").read_text().split(CUT)[0]
    not_run = {
        p.parent.name: p.read_text().strip()
        for p in (run / "kmerseek").glob("*/NOT_RUN")
    }
    excl = (
        (run / "db/exclusion_check.txt").read_text().strip()
        if (run / "db/exclusion_check.txt").exists()
        else "not found"
    )
    n_removed = (
        sum(1 for _ in open(run / "db/removed.tsv")) - 1
        if (run / "db/removed.tsv").exists()
        else "not found"
    )

    cols = [
        "setting",
        "n_placed",
        "median_error",
        "within_10",
        "shuffle_any_placement_rate",
        "shuffle_within_10_rate",
        "blind_median_error",
        "blind_within_10",
        "random_nearest_of_n_median_error",
        "random_nearest_of_n_within_10",
    ]
    s = [
        CUT,
        "",
        f"**Answer: {v['answer']}.**",
        "",
        verdict_table(v),
        "",
        f"Profile method (hmmscan, Pfam-A envelope) median error: {fmt(v['profile_median_error'])} residues. "
        "Median error pools the start and end errors of all 4,006 instances; an unplaced instance counts as infinite.",
        "",
    ]
    if not_run:
        s += (
            ["Runs not made, and why:", ""]
            + [f"- {k}: {t}" for k, t in sorted(not_run.items())]
            + [""]
        )
    s += [
        "### Exclusion check",
        "",
        f"{n_removed} Swiss-Prot sequences removed. Check of the queries against what is left:",
        "",
        "```",
        excl,
        "```",
        "",
        "### Controls and secondary numbers",
        "",
        f"Uniform random placement inside the real protein, length drawn from the true feature lengths (one draw per instance): "
        f"median error {fmt(v['uniform_random']['median_error'])} residues, within 10 residues at both ends "
        f"{fmt(v['uniform_random']['within_10'], True)}.",
        "",
        "Per method: shuffled proteins with any placement (pass/fail c), shuffles placed within 10 residues of the true "
        "feature, the truth-blind choice (lowest-E placement on the protein), and the nearest of n random placements "
        "(n = the method's placement count on that protein):",
        "",
        md_table(metrics[["method"] + cols]),
        "",
        "## Extended-search runs (not preregistered)",
        "",
        f"The same four conditions applied to the `{EXT}` runs give **{v['extended_not_preregistered']['answer']}**. "
        "This does not change the answer above.",
        "",
        verdict_table(v["extended_not_preregistered"]),
        "",
        "## Figures",
        "",
        "- `figures/SF1_boundary_error_cdf_elm_swissprot.png`: share of starts and ends within x residues, per method.",
        "- `figures/SF2_placements_20_random_elm_instances_swissprot.png`: 20 random instances, each method's placement.",
        "- `figures/SF3_boundary_error_by_feature_length_elm_swissprot.png`: error against feature length.",
        "",
        "## Worked examples",
        "",
        worked_examples(err, pl, seqs, metrics),
        "",
        "## Compute",
        "",
        compute_section(run),
        "",
    ]
    (a / "report.md").write_text(head + "\n".join(s))
    print(f"report.md written; answer {v['answer']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
