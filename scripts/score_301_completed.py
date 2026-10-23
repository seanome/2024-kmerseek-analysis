"""Score completed settings only, with an strict E cutoff set on decoys alone.

The cutoff is capped at 10 because the retained hits stop there. Boundary error is
conditional on recall, using largest overlap then E-value, target and coordinates.
Also report the previously requested half-inside diagnostic and the E=10 context.
"""

import argparse
import csv
import gzip
import math
import statistics
from pathlib import Path


def read(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def overlap(hit, window):
    return max(
        0,
        min(int(hit["end"]), int(window["end"]))
        - max(int(hit["start"]), int(window["start"]))
        + 1,
    )


def threshold(scores, n):
    """Largest strict threshold <=10 with strictly fewer than 1% decoy hits."""
    assert n > 0
    values = sorted(s for s in scores if math.isfinite(s) and s <= 10)
    allowed = math.ceil(n / 100) - 1
    return 10.0 if len(values) <= allowed else values[allowed]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ids = set((a.base / "step5/subset_query_ids.txt").read_text().split())
    features = [
        dict(r, feature_row=str(i))
        for i, r in enumerate(read(a.base / "step2/truth.tsv"))
        if r["query_id"] in ids
    ]
    decoys = [
        r
        for r in read(a.base / "step3/decoys.tsv")
        if r["query_id"] in ids and r["unchanged"] == "False"
    ]
    summaries = []
    details = []
    a.out.mkdir(parents=True, exist_ok=True)
    for directory in sorted((a.base / "step5").iterdir()):
        if not directory.is_dir() or not (directory / "status.tsv").exists():
            continue
        statuses = [r for r in read(directory / "status.tsv") if r["fit"] == "fitted"]
        if not statuses:
            continue
        status = statuses[-1]
        tag = next(directory.glob(f"*.k{status['ksize']}.*"))
        files = sorted(tag.glob("regions_filtered.chunk*.tsv.gz"))
        chunks = sorted((tag / "chunks").glob("chunk*.fasta"))
        # The downloaded snapshot can omit FASTAs; expected chunk count is recorded in manifest.
        expected = int((directory / "expected_chunks.txt").read_text())
        assert len(files) == expected, (directory, len(files), expected)
        hits = {}
        for f in files:
            for r in read(f):
                hits.setdefault(r["query_id"], []).append(r)
        for rule in ("any_overlap", "half_region_inside"):

            def eligible(h, w):
                ov = overlap(h, w)
                return ov > 0 and (
                    rule == "any_overlap"
                    or 2 * ov >= int(h["end"]) - int(h["start"]) + 1
                )

            best = []
            for d in decoys:
                best.append(
                    min(
                        (
                            float(h["region_evalue"])
                            for h in hits.get(d["decoy_id"], [])
                            if eligible(h, d)
                        ),
                        default=math.inf,
                    )
                )
            cut = threshold(best, len(decoys))
            for cutoff_kind, cutoff in [
                ("decoy_threshold", cut),
                ("context_E10", 10.0),
            ]:
                rows = []
                for w in features:
                    candidates = [
                        h
                        for h in hits.get(w["query_id"], [])
                        if float(h["region_evalue"]) < cutoff and eligible(h, w)
                    ]
                    h = (
                        min(
                            candidates,
                            key=lambda h: (
                                -overlap(h, w),
                                float(h["region_evalue"]),
                                h["target_name"],
                                int(h["start"]),
                                int(h["end"]),
                            ),
                        )
                        if candidates
                        else None
                    )
                    row = dict(
                        setting=tag.name,
                        rule=rule,
                        cutoff_kind=cutoff_kind,
                        feature_row=w["feature_row"],
                        query_id=w["query_id"],
                        feature_type=w["feature_type"],
                        recalled=bool(h),
                        boundary_error_aa=(
                            abs(int(h["start"]) - int(w["start"]))
                            + abs(int(h["end"]) - int(w["end"]))
                            if h
                            else ""
                        ),
                        overlap_fraction=(
                            overlap(h, w) / (int(w["end"]) - int(w["start"]) + 1)
                            if h
                            else 0
                        ),
                        predicted_start=h["start"] if h else "",
                        predicted_end=h["end"] if h else "",
                    )
                    rows.append(row)
                    details.append(row)
                for ft in ["all"] + sorted({w["feature_type"] for w in features}):
                    selected = [
                        r for r in rows if ft == "all" or r["feature_type"] == ft
                    ]
                    recalled = [r for r in selected if r["recalled"]]
                    ds = [
                        s
                        for d, s in zip(decoys, best)
                        if ft == "all" or d["feature_type"] == ft
                    ]
                    rec = len(recalled) / len(selected)
                    med = (
                        statistics.median(r["boundary_error_aa"] for r in recalled)
                        if recalled
                        else None
                    )
                    summaries.append(
                        dict(
                            setting=tag.name,
                            rule=rule,
                            cutoff_kind=cutoff_kind,
                            feature_type=ft,
                            region_evalue_threshold=cutoff,
                            n_features=len(selected),
                            n_recalled=len(recalled),
                            recall_features=rec,
                            n_decoy_windows=len(ds),
                            n_decoys_hit=sum(s < cutoff for s in ds),
                            decoy_hit_rate=(
                                sum(s < cutoff for s in ds) / len(ds) if ds else ""
                            ),
                            median_boundary_error_aa=med if med is not None else "",
                            median_overlap_fraction=(
                                statistics.median(
                                    r["overlap_fraction"] for r in recalled
                                )
                                if recalled
                                else ""
                            ),
                            partial_quality_verdict=(
                                "DROP"
                                if rec < 0.5 or (med is not None and med >= 8)
                                else "timing_not_measured_for_setting"
                            ),
                        )
                    )
    for name, rows in [("summary", summaries), ("features", details)]:
        with open(a.out / f"301_partial_{name}_swissprot.tsv", "w") as f:
            w = csv.DictWriter(
                f, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            w.writeheader()
            w.writerows({k: "NA" if v == "" else v for k, v in r.items()} for r in rows)
    for r in summaries:
        if r["feature_type"] == "all" and r["cutoff_kind"] == "decoy_threshold":
            print(r)


if __name__ == "__main__":
    main()
