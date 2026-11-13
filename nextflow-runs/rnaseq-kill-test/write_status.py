"""Write pre-run status without treating software tests as biological evidence."""

import itertools
import json
from pathlib import Path

import protocol

ROOT = Path(__file__).resolve().parent


def main():
    protocol.check_rules()
    target = ROOT / "report.md"
    fingerprint = ROOT / "evidence" / "report.sha256"
    if target.exists():
        if (
            not fingerprint.exists()
            or protocol.digest(target) != fingerprint.read_text().split()[0]
        ):
            raise ValueError("refusing to replace a modified report")
    rows = list(protocol.assignments(list("abcdefgh"), [0] * 4 + [1] * 4))
    independent = [
        bits for bits in itertools.product((0, 1), repeat=8) if sum(bits) == 4
    ]
    assert len(rows) == len(independent)
    splits = {
        min(r["assignment"], "".join(str(1 - int(x)) for x in r["assignment"]))
        for r in rows
    }
    counts = {
        "source": "exhaustive combinatorial enumeration; not a biological measurement",
        "balanced_assignments": len(rows),
        "alternative_labelled_assignments": len(rows) - 1,
        "distinct_splits_ignoring_condition_names": len(splits),
        "alternative_splits_ignoring_condition_names": len(splits) - 1,
    }
    status = protocol.score_claims({})
    evidence = ROOT / "evidence"
    evidence.mkdir(exist_ok=True)
    (evidence / "assignment_counts.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    (evidence / "unmeasured_claims.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )
    rules = (ROOT / "decision_rules.md").read_text()
    report = "# RNA-seq kill test: pre-run report\n\n## 1. Fixed decision rules, verbatim\n\n"
    report += rules
    report += """

## 2. Data chosen

No development or test experiment has been selected. No sequencing reads were
downloaded or analysed. No published DE output was inspected. No biological run began.
No calibration file exists. Mouse preflight is blocked until calibration is frozen.

Public metadata searches found possible stranded zebrafish studies. These are
candidates, not selected inputs. The study described in
[The Genome-Wide Impact of Nipblb Loss-of-Function on Zebrafish Gene Expression](https://pmc.ncbi.nlm.nih.gov/articles/PMC7766774/)
describes stranded paired-end libraries and triplicate conditions. Full sample/run
metadata still need verification. The neuroblastoma sample record
[GSM7016830](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7016830)
also documents stranded paired-end sequencing, but its study includes several
genotypes and has not been accepted as a two-condition development experiment.

Searches also returned mouse papers before development; no experiment was selected,
no sample-level test manifest was read, and no test result was used. Future mouse
selection must follow the committed calibration. Selection dates and complete sample
accessions remain not measured; a search date is not a selection date.

## 3. Fixed claim decisions

### Claim 1: speed

| Rule | Measured value | Decision |
| --- | --- | --- |
| Wall time and CPU-hours below 0.1 of the no-reference route | not measured | not measured |
| Wall time and CPU-hours below 0.5 of the reference route | not measured | not measured |

### Claim 2: concordance

| Rule | Measured value | Decision |
| --- | --- | --- |
| Top 300 units hit at least 70 of the top 100 genes | not measured | not measured |
| Top 20 GO terms overlap by at least 10 | not measured | not measured |

### Claim 3: family tagging

| Rule | Measured value | Decision |
| --- | --- | --- |
| In-domain correct-family fraction at least 0.80 | not measured | not measured |
| Outside-protein false-family fraction at most 0.05 | not measured | not measured |
| Tagging CPU-hours below 0.1 of hmmscan on the same sequences | not measured | not measured |

### Claim 4: amino-acid alphabet

| Rule | Measured value | Decision |
| --- | --- | --- |
| Reduced-alphabet recall gain at least 0.05 at equal false-family rate, each at its best k | not measured | not measured |

The alphabet has not been retained or dropped. A measured failure requires claims
1 to 3 to be reported again using protein20. Missing evidence does not select a winner.

### Claim 5: label swaps

| Rule | Measured value | Decision |
| --- | --- | --- |
| True labels produce more significant units than every balanced swap | not measured | not measured |
| Median swap count below 0.05 of true count | not measured | not measured |

## 4. Required figures and stage measurements

The overlap curve, route timing table, three-class tagging table, alphabet comparison
and swap distribution are not measured. No plot is drawn without measured outputs.
Index-build cost, peak memory, 16 GB allocation outcomes and manual-intervention counts
are not measured. Software unit tests are not biological measurements.

## 5. Closer-distance run

Not measured. Neither the vertebrate-excluded nor mammal-excluded family index exists
for this experiment. No gain from the closer distance has been measured.

## 6. Problems and choices identified before running

The supplied decision rules are unchanged. The separately frozen
[implementation choices](implementation_choices.md) contain the calibration procedure,
ranking ties, lineage exclusions, condition test and missing-value treatment.
Olga approved freezing that procedure first and putting the learned numeric thresholds
in a separate immutable calibration file before mouse access.

The literal swap rule includes an exchange of all condition names. A symmetric
two-condition test gives that exchange the same significant-unit count, so the strict
rule cannot pass in that case. This is a property of the rule, not a measured result.
The primary result will keep the literal comparison even if it fails for this reason.
The statement that four against four gives 18 swaps is inconsistent with enumeration:
"""
    report += f"""
```json
{json.dumps(counts, indent=2)}
```

These counts were printed by the code and checked independently by binary enumeration.
They describe possible assignments, not observed SPLASH results.

SPLASH's default test does not take condition labels. Source inspection found the
fixed-weight condition test and its Cjs_0_pval output in version 2.11.9. The optional
GLM output filters coefficients; it must not be treated as the requested significant
unit count. The condition-test binary and corrected column remain untested.

Sources: [pinned driver](https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/splash.py),
[pinned output writer](https://github.com/refresh-bio/SPLASH/blob/v2.11.9/src/satc_merge/satc_merge.cpp),
[GLM documentation](https://github.com/refresh-bio/SPLASH/wiki/GLM_supervised).

The implementation is incomplete. Missing biological stages and launch gates are
listed in [README.md](README.md). No Nextflow workflow, pinned conda installation,
cluster resource prediction, biological controls or figure notebook has been tested.

## 7. Independent adversarial review

"""
    review = evidence / "adversarial_review.md"
    report += (
        review.read_text()
        if review.exists()
        else "Not yet performed. No biological PASS is asserted.\n"
    )
    report += "\n### Changes in response\n\n"
    response = evidence / "review_response.md"
    report += response.read_text() if response.exists() else "Not yet recorded.\n"
    target.write_text(report)
    fingerprint.write_text(protocol.digest(target) + "  report.md\n")
    print(json.dumps({"claim_status": status, "assignment_counts": counts}, indent=2))


if __name__ == "__main__":
    main()
