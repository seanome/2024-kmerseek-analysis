"""Freeze genotype-level contrasts before reading any abundance results."""

import csv
import hashlib
import json
from pathlib import Path
import sys


def prepare(samples, output):
    with samples.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if len(rows) != 64 or len({r["run"] for r in rows}) != 64:
        raise ValueError("Expected all 64 unique libraries")
    young = {"6996n", "6990b", "6996m", "6941c"}
    contrasts = []
    for hour in (2, 24, 312):
        eligible = [
            r
            for r in rows
            if r["genotype"] in young and int(r["hours_after_treatment"]) == hour
        ]
        strata = {}
        for r in eligible:
            strata.setdefault(
                (r["genotype"], r["age_days"], r["hours_after_treatment"]), []
            ).append(r)
        selected = []
        rejected = []
        for key, members in sorted(strata.items()):
            if sorted(r["treatment"] for r in members) != ["Control", "Treatment"]:
                rejected.extend(
                    dict(
                        run=r["run"],
                        reason="No unique control and treatment at the same genotype, age and post-treatment time",
                    )
                    for r in members
                )
                continue
            selected.extend(
                dict(
                    run=r["run"],
                    genotype=r["genotype"],
                    condition=r["treatment"],
                    collection_stratum="|".join(key),
                    vial=r["vial"],
                )
                for r in members
            )
        genotypes = sorted({r["genotype"] for r in selected})
        if len(genotypes) < 3:
            raise ValueError("Fewer than three genotype units for planned contrast")
        contrasts.append(
            dict(
                name=f"young_{hour}h",
                hours=hour,
                genotypes=genotypes,
                genotype_units=len(genotypes),
                libraries=selected,
                rejected=rejected,
            )
        )
    record = dict(
        samples_sha256=hashlib.sha256(samples.read_bytes()).hexdigest(),
        contrasts=contrasts,
        all_libraries=64,
        model="DESeq2: ~ genotype + condition after summing balanced collection strata within each genotype/condition",
        collection_definition="Operational stratum: same genotype, age in days and hours after treatment. This is not a claim that vial numbers identify pairs or that a calendar collection date is known.",
        independence="One genotype is one biological unit; repeated collections do not increase replicate count.",
        other_times="Retain counts and metadata for all 64; no primary treatment test for unmatched times or the single old genotype.",
        filter="At least 5 pooled counts in at least 2 genotype/condition units, independent of treatment labels",
        kmer_length=31,
        masking="nomask",
        multiple_testing="BH separately over all eligible features per contrast and assay; do not restrict gene tests to SPLASH-selected genes",
        alpha=0.05,
        classification="Gene DE and sequence variation are separate evidence axes; absence of a significant gene result is not evidence for unchanged expression.",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    result = prepare(*map(Path, sys.argv[1:]))
    print(
        [
            (r["name"], r["genotype_units"], len(r["libraries"]))
            for r in result["contrasts"]
        ]
    )
