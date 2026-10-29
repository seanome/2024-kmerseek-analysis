"""Select every significant SPLASH anchor, without an extension cap."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def select(source, output):
    selected = []
    seen = set()
    tested = 0
    with source.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            anchor = row["anchor"]
            if not anchor or set(anchor) - set("ACGT"):
                raise ValueError("Invalid DNA anchor")
            q = float(row["pval_opt_corrected"])
            if anchor in seen or not math.isfinite(q) or not 0 <= q <= 1:
                raise ValueError("Duplicate anchor or invalid adjusted P-value")
            seen.add(anchor)
            tested += 1
            if q < 0.05:
                selected.append((q, anchor))
    selected.sort()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "selected_anchors.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(["anchor"])
        writer.writerows([anchor] for _, anchor in selected)
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(
        source=str(source),
        source_sha256=source_sha256,
        tested_anchors=tested,
        selected_anchors=len(selected),
        rule="pval_opt_corrected < 0.05",
        extension_cap=None,
        ordering="adjusted P, then anchor",
        note="Use the complete after_correction.all_anchors.tsv as the tested population.",
    )
    (output / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    print(json.dumps(select(a.source, a.output), indent=2))
