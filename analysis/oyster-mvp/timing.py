"""Collect measured wall time, CPU time and peak memory for every stage.

Sources, all written during the runs:
  timing/<stage>.time        GNU time -v of each top-level command
  dekupl/<depth>/snakemake-stats.json   Snakemake per-rule durations
  dekupl/<depth>/out/Logs/*exec_time.log  DE-kupl's own start/end stamps
  splash/<depth>/splash.stdout.log      SPLASH stage stamps
Writes timing/stages.tsv and timing/speed.json. Nothing is estimated here.
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_time(path):
    text = path.read_text()

    def grab(label):
        m = re.search(re.escape(label) + r":\s*(.+)", text)
        return m.group(1).strip() if m else None

    wall = grab("Elapsed (wall clock) time (h:mm:ss or m:ss)")
    parts = [float(x) for x in wall.split(":")] if wall else []
    secs = sum(v * 60 ** i for i, v in enumerate(reversed(parts))) if parts else None
    return dict(stage=path.stem, wall_s=secs, user_s=float(grab("User time (seconds)") or 0),
                sys_s=float(grab("System time (seconds)") or 0),
                max_rss_gb=int(grab("Maximum resident set size (kbytes)") or 0) / 1024 ** 2,
                exit_status=int(grab("Exit status") or -1))


def snakemake_rules(path):
    stats = json.loads(path.read_text())
    per_rule = defaultdict(float)
    for f, info in stats.get("files", {}).items():
        per_rule[info.get("rule", "")] += float(info.get("duration", 0))
    # Older layouts key durations by rule only.
    if not per_rule:
        for rule, info in stats.get("rules", {}).items():
            per_rule[rule] = float(info.get("mean-runtime", 0)) * int(info.get("count", 1) or 1)
    return dict(per_rule)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    rows = [parse_time(t) for t in sorted((root / "timing").glob("*.time"))]
    df = pd.DataFrame(rows)
    df["cpu_s"] = df.user_s + df.sys_s
    df.to_csv(root / "timing" / "stages.tsv", sep="\t", index=False)
    speed = {}
    for depth in ("1m", "5m", "full"):
        d = {}
        sub = df.set_index("stage")

        def wall(stage):
            return float(sub.loc[stage, "wall_s"]) if stage in sub.index else None

        d["trim_wall_s_per_library"] = {s: wall(f"trim-{depth}-{s}") for s in ["DW1", "DW2", "DW3", "ZW1", "ZW2", "ZW3"]}
        d["splash_total_wall_s"] = wall(f"splash-{depth}")
        d["dekupl_total_wall_s"] = wall(f"dekupl-{depth}")
        stats = root / "dekupl" / depth / "snakemake-stats.json"
        if stats.exists():
            d["dekupl_rule_seconds_summed"] = snakemake_rules(stats)
        d["kallisto_quant_wall_s"] = {s: wall(f"kallisto-quant-{depth}-{s}") for s in ["DW1", "DW2", "DW3", "ZW1", "ZW2", "ZW3"]}
        d["baseline_aggregate_wall_s"] = wall(f"baseline-aggregate-{depth}")
        d["baseline_de_wall_s"] = wall(f"baseline-de-{depth}")
        d["grade_wall_s"] = wall(f"grade-{depth}")
        d["annotate_wall_s"] = {m: wall(f"annotate-{depth}-{m}") for m in ("splash", "dekupl")}
        speed[depth] = d
    speed["kallisto_index_wall_s"] = float(df.set_index("stage").loc["kallisto-index", "wall_s"]) \
        if "kallisto-index" in set(df.stage) else None
    (root / "timing" / "speed.json").write_text(json.dumps(speed, indent=2))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
