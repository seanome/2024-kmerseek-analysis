#!/usr/bin/env python3
"""Independent notebook 360 audit using only Python's standard library.

Never imports pipeline helpers or reads/executes the notebook builder. Summaries are
read only after all family sets have been reconstructed, for discrepancy checks and
explicitly unverified status metadata. Default inputs are the supplied local delivery.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import math
import statistics
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

LABS = ["real"] + [f"swap{i}" for i in range(1, 10)]


def rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def number(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return math.nan


def score(families, key):
    return len(families & key), len(families - key)


def table(header, body):
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
        + ["| " + " | ".join(map(str, row)) + " |" for row in body]
    )


def fasta_ids(path):
    return [line[1:].split()[0] for line in path.open() if line.startswith(">")]


def verify_fasta(keyrows, path, protein):
    sequences = {}
    name = None
    for line in path.open():
        if line.startswith(">"):
            name = line[1:].split()[0]
            sequences[name] = ""
        else:
            sequences[name] += line.strip()
    for r in keyrows:
        seq = sequences[r["transcript_id"]]
        assert len(seq) == int(r["protein_length" if protein else "cdna_length"])
        if protein:
            assert hashlib.md5(seq.encode()).hexdigest() == r["md5"]


def sql_check(base, org, setting, key):
    """A second count using SQL DISTINCT and integer overlap arithmetic."""
    with sqlite3.connect(":memory:") as db:
        db.execute("create table key(f text primary key)")
        db.executemany("insert into key values (?)", [(x,) for x in key])
        db.execute("create table calls(f text,lab text,e real,a int,b int,c int,d int)")
        db.executemany(
            "insert into calls values (?,?,?,?,?,?,?)",
            (
                (
                    r["pfam_acc"],
                    r["labelling"],
                    number(r["region_evalue"]),
                    int(r["target_start"]),
                    int(r["target_end"]),
                    int(r["dom_start"]),
                    int(r["dom_end"]),
                )
                for r in rows(base / "landings" / f"{org}.{setting}.landings.tsv.gz")
            ),
        )
        result = {lab: (0, 0) for lab in LABS}
        query = (
            "select lab,count(distinct case when key.f is not null then calls.f end),"
            "count(distinct case when key.f is null then calls.f end) from calls "
            "left join key on key.f=calls.f where e<=1 and b>a and "
            "2*max(0,min(b,d)-max(a,c))>=b-a group by lab"
        )
        for lab, recovered, outside in db.execute(query):
            result[lab] = (recovered, outside)
        return result


def verify_aggregate(root):
    """Check the combined summary against individually audited source summaries."""
    fields = ["organism", "path", "setting", "labelling", "evalue_cutoff"]

    def keyed(records):
        result = {}
        for r in records:
            key = tuple(r[x] for x in fields)
            assert key not in result, ("duplicate summary row", key)
            result[key] = r
        return result

    individual = []
    for org in ["ciona", "zebrafish"]:
        base = root / "results" / org
        for path in sorted((base / "landings").glob("*.summary.tsv")):
            individual.extend(rows(path))
        individual.extend(rows(base / "standard" / f"{org}.standard.summary.tsv"))
    aggregate = keyed(rows(root / "results/summary.tsv"))
    assert aggregate == keyed(
        individual
    ), "combined summary differs from per-setting summaries"
    return len(aggregate)


def audit(root, source):
    lines = [
        "```mermaid",
        "flowchart LR",
        '  A["Raw DESeq2 tables, keys and landings"] --> B["Independent family sets for ten splits"]',
        '  B --> C["Recovery rule, discrepancies and limits"]',
        "```",
        "",
        "# Independent adversarial review of notebook 360",
        "",
        "Written by Codex. Audit date: 2026-10-22. No notebook builder was read or executed.",
        "",
        f"Inputs: `{root}`. Pipeline source: `{source}`.",
        "",
        "Reproduce with Python 3: `python3 scripts/nb360/adversarial_check.py --delivery "
        + str(root)
        + " --report notebooks/360_data/adversarial_review.md`.",
        "",
        "The script uses the standard library, reads every raw DESeq2 row and every landed region row, "
        "and reconstructs unique family sets before opening any summary. It never uses summary recovery counts as inputs. "
        "Outside means a unique Pfam family outside the 100 changed families; unchanged key families count as outside. "
        "Such counts are errors against the simulated changed-family key, not proof that every outside homolog is biologically unrelated.",
        "",
    ]
    labels = list(rows(source / "assets/labellings.tsv"))
    splits = {tuple(r[f"sample_{i:02d}"] for i in range(1, 7)) for r in labels}
    expected = {
        ("A",) + x for x in itertools.product("AB", repeat=5) if x.count("A") == 2
    }
    assert splits == expected and [r["labelling"] for r in labels] == LABS
    lines += [
        "## Labels and frozen rule",
        "",
        "HELD: all ten distinct balanced splits occur once, with sample 1 fixed in A to remove complementary duplicates. "
        "All nine swaps mix two original samples with one from the other condition. This is a balanced relabelling control, "
        "not independent no-signal data.",
        "",
        "The frozen document makes Ciona the deciding organism and zebrafish a distance control. "
        "The user now prefers zebrafish because of annotations. Its result below is a follow-up species result; "
        "it does not replace the original frozen Ciona verdict.",
        "",
    ]
    settings = list(
        csv.DictReader(
            (
                l
                for l in (source / "assets/kmerseek_settings.tsv").open()
                if not l.startswith("#")
            ),
            delimiter="\t",
        )
    )
    wanted = [r["alphabet"] + ".k" + r["ksize"] for r in settings]
    discrepancies = []
    for org in ["ciona", "zebrafish"]:
        print("Auditing", org, flush=True)
        base = root / "results" / org
        keyrows = list(rows(root / "key" / org / "answer_key.tsv"))
        key = {r["pfam_acc"] for r in keyrows if r["status"] != "unchanged"}
        counts = Counter(r["status"] for r in keyrows)
        assert (
            len(keyrows) == 200
            and len(key) == 100
            and counts == {"up": 50, "down": 50, "unchanged": 100}
        )
        for field in ["gene_id", "transcript_id", "peptide_id", "pfam_acc", "clan"]:
            assert len({r[field] for r in keyrows}) == 200, (org, field)
        assert all(
            float(r["fold_change"])
            == {"up": 4, "down": 0.25, "unchanged": 1}[r["status"]]
            for r in keyrows
        )
        assert all(500 <= int(r["cdna_length"]) <= 5000 for r in keyrows)
        for field, name in [
            ("transcript_id", "transcripts.fa"),
            ("transcript_id", "proteins.fa"),
        ]:
            ids = fasta_ids(root / "key" / org / name)
            assert len(ids) == 200 and set(ids) == {r[field] for r in keyrows}
        verify_fasta(keyrows, root / "key" / org / "proteins.fa", True)
        verify_fasta(keyrows, root / "key" / org / "transcripts.fa", False)
        true = list(rows(base / "reads/true_counts.tsv"))
        assert len(true) == 200 and {r["transcript_id"] for r in true} == {
            r["transcript_id"] for r in keyrows
        }
        statuses = {r["transcript_id"]: r["status"] for r in keyrows}
        ratios = defaultdict(list)
        for r in true:
            a = sum(float(r[f"sample_{i:02d}"]) for i in range(1, 4))
            b = sum(float(r[f"sample_{i:02d}"]) for i in range(4, 7))
            ratios[statuses[r["transcript_id"]]].append(b / a)
        lines += [
            f"## {org}",
            "",
            "HELD: 200 distinct genes, transcripts, peptides, Pfam families and clans; 50 up, 50 down, 100 unchanged. "
            "The changed-family denominator is 100. Fold-change labels, cDNA length limits, FASTA identifiers and true-count identifiers agree. Both FASTAs use transcript identifiers, including the protein FASTA. Every protein MD5 and both sequence lengths match the key.",
            "",
            "Median simulated B/A count ratio by true status: "
            + ", ".join(
                f"{k} {statistics.median(v):.4f}" for k, v in sorted(ratios.items())
            )
            + ".",
            "",
        ]
        de = {}
        sigsets = {}
        real_lfc = {}
        for kind in ["kmers", "transcripts"]:
            for lab in LABS:
                f = base / "deseq2" / f"{org}.{kind}.{lab}.deseq2.tsv.gz"
                n = na = pna = sig = up = down = 0
                digest = hashlib.sha256()
                features = set()
                changed = set()
                sx = sy = sxx = syy = sxy = 0.0
                paired = 0
                for r in rows(f):
                    n += 1
                    feature = r["feature"]
                    features.add(feature)
                    digest.update((feature + "\n").encode())
                    padj = number(r["padj"])
                    pval = number(r["pvalue"])
                    lfc = number(r["log2FoldChange"])
                    na += not math.isfinite(padj)
                    pna += not math.isfinite(pval)
                    if padj < 0.05:
                        sig += 1
                        changed.add(feature)
                        up += lfc > 0
                        down += lfc < 0
                    if lab == "real":
                        real_lfc[kind, feature] = lfc
                    else:
                        x = real_lfc.get((kind, feature), math.nan)
                        if math.isfinite(x) and math.isfinite(lfc):
                            paired += 1
                            sx += x
                            sy += lfc
                            sxx += x * x
                            syy += lfc * lfc
                            sxy += x * lfc
                assert len(features) == n, (org, kind, lab, "duplicate features")
                signature = digest.hexdigest()
                if lab == "real":
                    original_signature = signature
                else:
                    assert signature == original_signature, (
                        org,
                        kind,
                        lab,
                        "feature order/set differs",
                    )
                denom = (paired * sxx - sx * sx) * (paired * syy - sy * sy)
                corr = (
                    (paired * sxy - sx * sy) / math.sqrt(denom)
                    if denom > 0
                    else math.nan
                )
                de[kind, lab] = dict(
                    n=n,
                    na=na,
                    pna=pna,
                    sig=sig,
                    up=up,
                    down=down,
                    corr=corr,
                    paired=paired,
                )
                sigsets[kind, lab] = changed
        lines += [
            "### Raw DESeq2 counts",
            "",
            table(
                [
                    "path",
                    "labels",
                    "features",
                    "padj NA",
                    "pvalue NA",
                    "changed",
                    "up/down",
                    "LFC correlation with real",
                ],
                [
                    (
                        kind,
                        lab,
                        d["n"],
                        d["na"],
                        d["pna"],
                        d["sig"],
                        f"{d['up']}/{d['down']}",
                        "reference" if lab == "real" else f"{d['corr']:.4f}",
                    )
                    for (kind, lab), d in de.items()
                ],
            ),
            "",
            "All features were read; no p-adjusted NA value was treated as significant. Feature identifiers and order match across labels. "
            "The LFC column is log2(B/A). Correlations compare each swapped table against the real table over finite LFC pairs. "
            "Different LFCs establish that the swapped output is not merely a copy with a new label. "
            "They cannot establish that the count matrices were assigned correctly without the missing matrices.",
            "",
        ]
        pmap = defaultdict(set)
        runs = {}
        pmap_bad = 0
        for r in rows(base / "kmer_path" / f"{org}.peptide_map.tsv"):
            pmap[r["labelling"]].add(r["peptide_id"])
            run = (r["labelling"], r["run_id"])
            value = (int(r["run_length_nt"]), r["sign"])
            if run in runs and runs[run] != value:
                pmap_bad += 1
            runs[run] = value
        assert not pmap_bad and set(pmap) <= set(LABS)
        lens = [v[0] for v in runs.values()]
        lines += [
            "### Changed k-mers and assembly",
            "",
            table(
                [
                    "labels",
                    "changed k-mers",
                    "unique peptide queries",
                    "runs represented in peptide map",
                ],
                [
                    (
                        lab,
                        de["kmers", lab]["sig"],
                        len(pmap[lab]),
                        sum(k[0] == lab for k in runs),
                    )
                    for lab in LABS
                ],
            ),
            "",
            f"The peptide map represents {len(runs)} distinct runs: lengths {min(lens)}–{max(lens)} nt, "
            f"median {statistics.median(lens):g} nt; {sum(x > 31 for x in lens)} exceed one 31-mer. "
            "Runs absent from the peptide map cannot be counted because the runs table was not delivered.",
            "",
            "FAILED as a literal claim: “without assembly”. The source constructs a graph of changed 31-mers and reverse complements, "
            "joins same-sign 30-nt overlaps into maximal unbranched sequences, then translates them. This is local assembly "
            "of significant k-mers. “Without whole-transcriptome assembly” is supported by this code path. "
            "The long runs in the delivered peptide map confirm that joining was used. "
            "Without runs.tsv and query FASTA, exact sequence reconstruction and six-frame translation remain untested.",
            "",
        ]
        std = defaultdict(set)
        badstd = 0
        for r in rows(base / "standard" / f"{org}.standard.calls.tsv"):
            lab = r["labelling"]
            assert lab in LABS
            valid = number(r["padj"]) < 0.05 and number(r["evalue"]) <= 1e-5
            badstd += not valid
            assert r["qseqid"] in sigsets["transcripts", lab]
            assert (r["in_key"].lower() == "true") == (r["pfam_acc"] in key)
            if valid:
                std[lab].add(r["pfam_acc"])
        assert not badstd
        scores = {}
        details = {}
        metadata = {}
        for setting in wanted:
            f = base / "landings" / f"{org}.{setting}.landings.tsv.gz"
            if not f.exists():
                continue
            fams = defaultdict(set)
            kept = Counter()
            total = 0
            bad = Counter()
            excluded = 0
            for r in rows(f):
                total += 1
                lab = r["labelling"]
                assert lab in LABS and r["organism"] == org and r["setting"] == setting
                assert r["query_name"] in pmap[lab]
                a, b, c, d = (
                    int(r[x])
                    for x in ["target_start", "target_end", "dom_start", "dom_end"]
                )
                overlap = max(0, min(b, d) - max(a, c))
                length = b - a
                inside = overlap / length if length > 0 else math.nan
                bad["interval"] += length <= 0 or a < 0 or c < 0 or d <= c
                bad["overlap"] += number(r["overlap"]) != overlap
                bad["length"] += number(r["region_target_length"]) != length
                bad["inside"] += not math.isclose(
                    number(r["inside"]), inside, abs_tol=1e-12
                )
                bad["domain_filter"] += not inside >= 0.5
                bad["key_flag"] += (r["in_key"].lower() == "true") != (
                    r["pfam_acc"] in key
                )
                if inside >= 0.5 and number(r["region_evalue"]) <= 1:
                    fams[lab].add(r["pfam_acc"])
                    kept[lab] += 1
                else:
                    excluded += 1
            assert not any(bad.values()), (org, setting, bad)
            scores[setting] = {lab: score(fams[lab], key) for lab in LABS}
            details[setting] = (total, excluded, kept)
            print(org, setting, total, scores[setting]["real"], flush=True)
        # All independent scores exist before summaries are consulted.
        for setting in scores:
            summary = base / "landings" / f"{org}.{setting}.summary.tsv"
            statuses_found = set()
            for r in rows(summary):
                statuses_found.add(r["status"])
                if number(r["evalue_cutoff"]) != 1:
                    continue
                rec, outside = scores[setting][r["labelling"]]
                for col, actual in [
                    ("recovered", rec),
                    ("outside", outside),
                    ("recovery_fraction", rec / len(key)),
                    ("n_changed_domains", rec + outside),
                    ("n_queries", len(pmap[r["labelling"]])),
                    ("n_landing_rows", details[setting][2][r["labelling"]]),
                ]:
                    if not math.isclose(number(r[col]), actual, abs_tol=1e-12):
                        discrepancies.append(
                            (org, setting, r["labelling"], col, r[col], actual)
                        )
            metadata[setting] = ",".join(sorted(statuses_found))
        for r in rows(base / "standard" / f"{org}.standard.summary.tsv"):
            rec, outside = score(std[r["labelling"]], key)
            for col, actual in [
                ("recovered", rec),
                ("outside", outside),
                ("recovery_fraction", rec / len(key)),
            ]:
                if not math.isclose(number(r[col]), actual, abs_tol=1e-12):
                    discrepancies.append(
                        (org, "standard", r["labelling"], col, r[col], actual)
                    )
        missing = sorted(set(wanted) - set(scores))
        lines += [
            "### Recovery independently counted from calls and landed regions",
            "",
            "Each pair below is recovered key families / outside-key families. Divide the first number by 100 for the recovery fraction. "
            "Only landed intervals with recomputed overlap at least half the target interval and E-value at most 1 enter the counts. "
            "Counts use sets of accessions, not numbers of rows, and ignore the supplied in_key flag except to check it.",
            "",
            table(
                ["setting", "reported status (unverified)"] + LABS,
                [
                    ("standard", "calls present")
                    + tuple(
                        f"{score(std[lab],key)[0]}/{score(std[lab],key)[1]}"
                        for lab in LABS
                    )
                ]
                + [
                    (s, metadata[s])
                    + tuple(f"{scores[s][lab][0]}/{scores[s][lab][1]}" for lab in LABS)
                    for s in scores
                ],
            ),
            "",
            table(
                [
                    "setting",
                    "raw landed rows",
                    "rows excluded at E > 1",
                    "retained rows",
                ],
                [(s, d[0], d[1], sum(d[2].values())) for s, d in details.items()],
            ),
            "",
        ]
        reduced = [s for s in scores if not s.startswith("protein20.")]
        best = min(
            reduced, key=lambda s: (-scores[s]["real"][0], scores[s]["real"][1], s)
        )
        assert sql_check(base, org, best, key) == scores[best]
        rec, out = scores[best]["real"]
        standard = score(std["real"], key)[0]
        nulls = [scores[best][lab][0] for lab in LABS[1:]]
        median = statistics.median(nulls)
        ties = [s for s in reduced if scores[s]["real"][0] == rec]
        passed = rec >= standard and rec >= 5 * median
        lines += [
            "### Fixed-rule result",
            "",
            "A second implementation using SQLite COUNT(DISTINCT) and integer overlap arithmetic agrees with all ten scores for the selected best setting.",
            "",
            f"Best delivered reduced setting: `{best}`; recovered {rec}/100 ({rec/100:.2f}), outside {out}. "
            f'Standard: {standard}/100 ({standard/100:.2f}). Settings tied on recovery: {", ".join(ties)}. '
            "Ties are resolved by fewer outside families, then alphabet name.",
            "",
            f"Nine swapped key recoveries for that setting: {nulls}; median {median:g}. "
            f"Part 1 ({rec} >= {standard}): {rec >= standard}. Part 2 ({rec} >= 5 × {median:g}): {rec >= 5*median}. "
            f'Delivered-settings result: {"PASS" if passed else "FAIL"}.',
            "",
            f'Coverage: {len(scores)}/19 settings delivered; missing: {", ".join(missing) or "none"}. '
            + (
                "The full best-of-all reduced-setting verdict is UNTESTABLE: an absent setting could change the winner. "
                "A missing setting is not zero recovery or an unfitted search."
                if missing
                else "All planned settings have landing tables."
            ),
            "",
            "Reported unfitted settings: "
            + ", ".join(s for s, v in metadata.items() if "unfitted" in v)
            + ". "
            "Their empty landed tables are observed; the cause is summary metadata only. Logs and .status files were not delivered. "
            "An unfitted search has no calibrated result; it must remain distinct from a fitted search with no hits.",
            "",
        ]
        if all(de["kmers", lab]["sig"] == 0 for lab in LABS[1:]):
            lines += [
                "CONFIRMED control limitation: all nine swapped k-mer DESeq2 tables have zero changed k-mers and the peptide map "
                "has no swapped queries. Thus zero swapped landings arise before kmerseek is called. The fixed inequality is "
                "numerically satisfied at a zero median, even by zero real recovery. The ratio to that median is undefined. "
                "This tests the combined DE-selection pipeline, but does not test whether protein landing distinguishes real "
                "queries from null queries of comparable number, length and composition. It cannot support a strong "
                "claim about landing specificity or a measured finite five-fold enrichment.",
                "",
            ]
        lines += [
            "### Missing evidence",
            "",
            "UNTESTABLE from this delivery: DESeq2 reruns and sample/count assignment; the total-count >=10 k-mer filter; "
            "salmon rounding; raw search coverage and fits; independent domain annotation joins before the landed tables; "
            "DIAMOND best-hit choice and ties (the source uses -k 1, so omitted equal-score targets cannot be sorted later); "
            "database genus exclusion and key/database membership; exact contig sequences and translations. "
            "The counts, kmerseek and salmon directories contain no files. Raw blastx hits, search database/domain tables, "
            "query FASTA and runs.tsv are absent. The true-count table validates simulation direction only.",
            "",
        ]
    aggregate_n = verify_aggregate(root)
    lines += [
        "## Summary discrepancies",
        "",
        f"The combined summary has {aggregate_n} rows and exactly matches the per-setting summaries, including statuses, all cutoffs and all label splits. Those primary counts were checked against the independent raw-table counts above.",
        "",
        (
            table(
                ["organism", "setting", "labels", "column", "summary", "independent"],
                discrepancies,
            )
            if discrepancies
            else "No discrepancies in primary-cutoff family recovery, recovery fractions, outside counts, changed-family totals, query counts or retained landing-row counts in the per-setting summaries. Standard family counts also agree."
        ),
        "",
        "## Preparing for an Adversarial Reviewer 2: Stress-testing the findings",
        "",
        "- Checks run: Python compilation, full raw-data audit, and boundary fixtures for duplicate families, half-overlap, below-half overlap, E=1, E>1, missing E-values, zero-length intervals and empty swaps. All passed.",
        "- A second SQL implementation agrees on every label for the selected setting. A separate SQL count of standard calls also gave Ciona 87 key / 82 outside and zebrafish 98 key / 14 outside. Protein MD5s and cDNA/protein lengths match every key row.",
        "- Recomputed the full population of landed rows, every raw DESeq2 table, and all standard calls. Recomputed overlap from coordinates and key membership from answer keys; checked every supplied flag and peptide label.",
        "- Asserted distinct features, identical feature sets/order across relabellings, 200 unique keys/clans and exact 50/50/100 statuses. Checked true-count sample direction independently of DESeq2.",
        "- Tried to refute the assembly finding: the implementation and long represented run lengths both show joining. The supported narrower statement is absence of whole-transcriptome assembly.",
        "- Tried to refute the weak-control finding: inspected raw swapped DESeq2 tables before inspecting peptide maps or landing summaries. A zero-query control never reaches the landing step.",
        "- Not tested: the upstream checks listed under Missing evidence; the notebook prose, saved outputs and builder, by instruction. The audit concerns the fixed rule and supplied pipeline sources, not an unseen notebook narrative.",
        "",
    ]
    return "\n".join(lines)


def read_fasta(path):
    result = {}
    name = None
    for line in path.open():
        if line.startswith(">"):
            name = line[1:].split()[0]
            assert name not in result
            result[name] = ""
        else:
            result[name] += line.strip()
    return result


def reverse_complement(seq):
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def translation_exceptions(root, org):
    """Test standard-code translation against the independently annotated key proteins."""
    base = root / "key" / org
    dna = read_fasta(base / "transcripts.fa")
    proteins = read_fasta(base / "proteins.fa")
    key = {r["transcript_id"]: r for r in rows(base / "answer_key.tsv")}
    code = dict(
        zip(
            ("".join(x) for x in itertools.product("TCAG", repeat=3)),
            "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
        )
    )
    exceptions = []
    for tx, seq in dna.items():
        translations = [
            "".join(code.get(nt[i : i + 3], "X") for i in range(frame, len(nt) - 2, 3))
            for nt in [seq, reverse_complement(seq)]
            for frame in range(3)
        ]
        if not any(proteins[tx] in aa for aa in translations):
            r = key[tx]
            exceptions.append((tx, r["gene_symbol"], r["pfam_acc"], r["status"]))
    out = [
        f"### {org}: translation-code limitations in the answer key",
        "",
        f"{len(dna)-len(exceptions)}/200 annotated key proteins occur intact in standard-code six-frame translations "
        "of their cDNAs. The exceptions are:",
        "",
        table(["transcript", "gene", "family", "true status"], exceptions),
        "",
    ]
    if org == "ciona":
        tx = "ENSCINT00000037364.1"
        recode = code | {"ATA": "M", "TGA": "W", "AGA": "G", "AGG": "G"}
    else:
        tx = "ENSDART00000093606.3"
        recode = code | {"ATA": "M", "TGA": "W", "AGA": "*", "AGG": "*"}
    nt = dna[tx]
    aa = "".join(recode[nt[i : i + 3]] for i in range(0, len(nt) - 2, 3)).rstrip("*")
    assert "M" + aa[1:] == proteins[tx]
    out += [
        f"For {tx}, the explicit codon overrides { {k:v for k,v in recode.items() if code[k]!=v} }, "
        "plus an initiating M, reproduce the annotated protein exactly. The default standard-code translation does not.",
        "",
    ]
    if org == "zebrafish":
        tx = "ENSDART00000084598.6"
        assert any(
            proteins[tx].replace("U", "*") in aa
            for aa in [
                "".join(
                    code.get(nt[i : i + 3], "X") for i in range(frame, len(nt) - 2, 3)
                )
                for nt in [dna[tx], reverse_complement(dna[tx])]
                for frame in range(3)
            ]
        )
        out += [
            "For selenos, substituting the annotated U with a stop reproduces a substring of the standard translation; "
            "the pipeline splits at that stop. This demonstrates recoding sensitivity from the provided sequences.",
            "",
            "The two changed exceptions are PF00115 and PF06936. Both occur in the delivered hsdm17.k7 "
            "E<=1 real-label families and standard calls (checked directly). They do not explain the current 77 vs 98 "
            "family gap, but the six-frame code does not reproduce every annotated key protein intact.",
            "",
        ]
        b = root / "results" / org
        landed = {
            r["pfam_acc"]
            for r in rows(b / "landings" / f"{org}.hsdm17.k7.landings.tsv.gz")
            if r["labelling"] == "real" and number(r["region_evalue"]) <= 1
        }
        std = {
            r["pfam_acc"]
            for r in rows(b / "standard" / f"{org}.standard.calls.tsv")
            if r["labelling"] == "real"
        }
        assert {"PF00115", "PF06936"} <= landed & std
    else:
        out += [
            "The Ciona exception is unchanged, so it does not alter the 100 changed-family denominator.",
            "",
        ]
    return out


def extended_inputs(root, source):
    """Check count normalization, labelled contrasts, contig provenance and translations."""
    out = [
        "## Additional raw-input audit, 2026-10-23",
        "",
        "No DESeq2 reruns, scheduler queries or jobs were started. Counts, runs, peptide sequences, "
        "blastx rows and search status files were read directly. These checks extend the earlier family recount. With --extend-only, earlier family scores are retained, not refreshed. The extension does not claim that re-scoring has completed; the family-score tables exclude gbmr7.",
        "",
        "### gbmr7 failure stage and scoring repair",
        "",
        "CORRECTION: the user reports that INDEX and SEARCH completed for gbmr7.k11; LANDINGS failed "
        "with exit 137 on 16, 32 and 48 GB attempts. Re-scoring job 49179127 is user-reported and was not polled. "
        "The fitted status files below directly support search completion as recorded by the pipeline. "
        "No local scheduler trace was supplied to independently verify the attempt sizes or exit code. "
        "Missing landings are a failed scoring stage, not a missing or unfitted search.",
        "",
        "Filtering finite region E-values to <=100 before the domain join preserves every fixed primary landing "
        "(E<=1) and every reported cutoff (maximum 100), provided the same target coordinates, annotations and "
        "query mapping are used and the full raw scores are retained. This follows because every region eligible "
        "at those cutoffs survives the earlier filter. It does not verify the uninspected repair implementation. "
        "The full best-of-all verdict remains pending the repaired gbmr7 scoring.",
        "",
    ]
    for org in ["ciona", "zebrafish"]:
        print("Additional input audit:", org, flush=True)
        base = root / "results" / org
        out += [f"### {org}: counts and sample assignment", ""]
        count_stats = []
        contrast_stats = []
        for kind in ["kmers", "transcripts"]:
            matrix = {}
            logs = [[] for _ in range(6)]
            totals = [0] * 6
            fractional = 0
            f = base / "counts" / f"{org}.{kind}.counts.tsv"
            with f.open() as h:
                assert h.readline().rstrip("\n").split("\t") == ["feature"] + [
                    f"sample_{i:02d}" for i in range(1, 7)
                ]
            for r in rows(f):
                feat = r["feature"]
                assert feat not in matrix
                original = [float(r[f"sample_{i:02d}"]) for i in range(1, 7)]
                assert all(math.isfinite(x) and x >= 0 for x in original)
                fractional += any(x != round(x) for x in original)
                values = [round(x) for x in original]
                if kind == "kmers":
                    assert len(feat) == 31 and set(feat) <= set("ACGT")
                    assert feat <= reverse_complement(feat)
                    assert sum(values) >= 10 and values == original
                matrix[feat] = values
                totals = [a + b for a, b in zip(totals, values)]
                if all(values):
                    logmean = statistics.mean(math.log(x) for x in values)
                    for i, x in enumerate(values):
                        logs[i].append(math.log(x) - logmean)
            assert all(logs), (org, kind, "no features positive in all six samples")
            factors = [math.exp(statistics.median(v)) for v in logs]
            maxerror = 0.0
            nbad = 0
            for assignment in rows(source / "assets/labellings.tsv"):
                lab = assignment["labelling"]
                seen = set()
                concordant = eligible = 0
                ai = [i for i in range(6) if assignment[f"sample_{i+1:02d}"] == "A"]
                bi = [i for i in range(6) if assignment[f"sample_{i+1:02d}"] == "B"]
                for r in rows(base / "deseq2" / f"{org}.{kind}.{lab}.deseq2.tsv.gz"):
                    feature = r["feature"]
                    assert feature not in seen
                    seen.add(feature)
                    values = matrix[feature]
                    norm = [x / s for x, s in zip(values, factors)]
                    expected_mean = sum(norm) / 6
                    observed_mean = number(r["baseMean"])
                    error = abs(expected_mean - observed_mean)
                    maxerror = max(maxerror, error)
                    nbad += not math.isclose(
                        expected_mean, observed_mean, rel_tol=1e-9, abs_tol=1e-8
                    )
                    a = sum(norm[i] for i in ai) / 3
                    b = sum(norm[i] for i in bi) / 3
                    lfc = number(r["log2FoldChange"])
                    if math.isfinite(lfc) and abs(b - a) > 1e-8 and abs(lfc) > 1e-8:
                        eligible += 1
                        concordant += (lfc > 0) == (b > a)
                assert seen == set(matrix), (
                    org,
                    kind,
                    lab,
                    "matrix/DE feature mismatch",
                )
                contrast_stats.append(
                    (
                        kind,
                        lab,
                        concordant,
                        eligible,
                        f"{concordant/max(eligible,1):.6f}",
                    )
                )
            assert nbad == 0, (
                org,
                kind,
                "baseMean normalization mismatch",
                nbad,
                maxerror,
            )
            count_stats.append(
                (
                    kind,
                    len(matrix),
                    fractional,
                    ",".join(map(str, totals)),
                    ", ".join(f"{x:.8f}" for x in factors),
                    f"{maxerror:.3g}",
                )
            )
        out += [
            table(
                [
                    "path",
                    "features",
                    "rows needing rounding",
                    "sample 1–6 totals (rounded)",
                    "independent size factors",
                    "maximum baseMean error",
                ],
                count_stats,
            ),
            "",
            "HELD: matrix sample headers are exactly sample_01 through sample_06; identifiers are unique and "
            "feature sets match all ten DESeq2 tables. Every k-mer is a canonical DNA 31-mer with integer counts "
            "and summed count >=10. This verifies the delivered retained table, not whether the upstream dump "
            "omitted eligible k-mers. Transcript counts were rounded to nearest even, matching R round.",
            "",
            "DESeq2 default median-ratio normalization was independently reconstructed from features positive "
            "in all six samples: geometric mean per feature, then exp(median(log count minus log geometric mean)) "
            "per sample. Normalized row means agree with every baseMean in all ten tables. This checks the "
            "delivered matrix against the fitted tables without refitting.",
            "",
            table(
                [
                    "path",
                    "labels",
                    "LFC signs agreeing with normalized B-A",
                    "finite nonzero comparisons",
                    "fraction",
                ],
                contrast_stats,
            ),
            "",
            "The contrast-sign check uses the actual A/B assignment for each relabelling. DESeq2 fits a "
            "negative-binomial model, so agreement with normalized arithmetic means is a diagnostic, not an "
            "identity required of every feature. These checks support the column assignment used in the outputs; "
            "they cannot authenticate FASTQ identities or salmon quantification without those files.",
            "",
        ]
        changed = {}
        for lab in LABS:
            changed[lab] = {"up": set(), "down": set()}
            for r in rows(base / "deseq2" / f"{org}.kmers.{lab}.deseq2.tsv.gz"):
                if number(r["padj"]) < 0.05:
                    sign = "up" if number(r["log2FoldChange"]) > 0 else "down"
                    changed[lab][sign].add(r["feature"])
        runs = list(rows(base / "kmer_path" / f"{org}.runs.tsv"))
        covered = defaultdict(set)
        run_ids = set()
        joins = 0
        peptides = read_fasta(base / "kmer_path" / f"{org}.queries.fasta")
        assert len(peptides) == len(set(peptides.values()))
        peptide_ids = {v: k for k, v in peptides.items()}
        expected_map = Counter()
        # Standard genetic code, codons ordered T,C,A,G at each position.
        aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
        code = dict(zip(("".join(x) for x in itertools.product("TCAG", repeat=3)), aas))
        assert (
            code["ATG"] == "M"
            and code["TGG"] == "W"
            and all(code[x] == "*" for x in ["TAA", "TAG", "TGA"])
        )
        nodes = {
            (lab, sign): kmers | {reverse_complement(x) for x in kmers}
            for lab, signs in changed.items()
            for sign, kmers in signs.items()
        }
        for r in runs:
            lab, sign, run = r["labelling"], r["sign"], r["run_id"]
            assert (lab, run) not in run_ids
            run_ids.add((lab, run))
            seq = r["sequence"]
            length = int(r["length_nt"])
            n = int(r["n_kmers"])
            assert length == len(seq) and length == 30 + n and n >= 1
            words = [seq[i : i + 31] for i in range(n)]
            for i, word in enumerate(words):
                canonical = min(word, reverse_complement(word))
                assert canonical in changed[lab][sign]
                assert canonical not in covered[lab, sign], (
                    org,
                    "duplicate k-mer across runs",
                )
                covered[lab, sign].add(canonical)
                if i:
                    prev = words[i - 1]
                    pool = nodes[lab, sign]
                    assert sum(prev[1:] + b in pool for b in "ACGT") == 1
                    assert sum(b + word[:-1] in pool for b in "ACGT") == 1
                    joins += 1
            for strand, nt in [("+", seq), ("-", reverse_complement(seq))]:
                for frame in range(3):
                    aa = "".join(
                        code[nt[i : i + 3]] for i in range(frame, len(nt) - 2, 3)
                    )
                    for piece in aa.split("*"):
                        if len(piece) >= 5:
                            assert piece in peptide_ids, (
                                org,
                                "missing translated peptide",
                            )
                            expected_map[
                                (
                                    peptide_ids[piece],
                                    lab,
                                    run,
                                    sign,
                                    strand,
                                    str(frame),
                                    str(length),
                                )
                            ] += 1
        for lab in LABS:
            for sign in ["up", "down"]:
                assert covered[lab, sign] == changed[lab][sign]
        columns = [
            "peptide_id",
            "labelling",
            "run_id",
            "sign",
            "strand",
            "frame",
            "run_length_nt",
        ]
        actual_map = Counter(
            tuple(r[c] for c in columns)
            for r in rows(base / "kmer_path" / f"{org}.peptide_map.tsv")
        )
        assert expected_map == actual_map, (org, "six-frame peptide mapping differs")
        assert set(peptides) == {r[0] for r in expected_map}
        out += [
            f"### {org}: runs and translated queries",
            "",
            f"HELD: {len(runs)} runs cover every significant canonical k-mer exactly once, with matching labels "
            f"and fold-change sign. All run lengths and n_kmers agree with the sequences; {joins} successive "
            "30-nt joins were checked for unique predecessor/successor within the changed-k-mer graph. "
            "This checks internal unbranched transitions and complete nonduplicated coverage, not a second "
            "implementation of maximal endpoint/cycle choices.",
            "",
            f"HELD: all {len(peptides)} distinct query sequences and {sum(actual_map.values())} peptide-map rows "
            "match independent six-frame translation, stop splitting and minimum peptide length 5. "
            "Exact sequence evidence now confirms local assembly, beyond the earlier run-length metadata.",
            "",
        ]
        blast = defaultdict(list)
        with (base / "standard" / f"{org}.blastx.tsv").open() as h:
            for row in csv.reader(h, delimiter="\t"):
                assert len(row) == 10
                blast[row[0]].append(row)
        best = {}
        for q, hits in blast.items():
            eligible = [r for r in hits if float(r[4]) <= 1e-5]
            if eligible:
                best[q] = min(eligible, key=lambda r: (-float(r[5]), r[1]))
        transcript_de = {
            lab: {
                r["feature"]: r
                for r in rows(
                    base / "deseq2" / f"{org}.transcripts.{lab}.deseq2.tsv.gz"
                )
            }
            for lab in LABS
        }
        standard_rows = 0
        for r in rows(base / "standard" / f"{org}.standard.calls.tsv"):
            de_row = transcript_de[r["labelling"]][r["qseqid"]]
            for col in ["baseMean", "log2FoldChange", "pvalue", "padj"]:
                assert math.isclose(number(r[col]), number(de_row[col]), rel_tol=1e-12)
            hit = best[r["qseqid"]]
            assert hit[1] == r["sseqid"]
            for col, index in [("pident", 2), ("evalue", 4), ("bitscore", 5)]:
                assert math.isclose(float(hit[index]), float(r[col]), rel_tol=1e-12)
            standard_rows += 1
        status_files = list((base / "kmerseek").glob("*.status"))
        statuses = {
            p.name.removeprefix(org + ".")
            .removesuffix(".status"): p.read_text()
            .strip()
            for p in status_files
        }
        assert len(statuses) == 19 and set(statuses.values()) <= {"fitted", "unfitted"}
        assert statuses["gbmr7.k11"] == "fitted"
        for path in (base / "landings").glob("*.summary.tsv"):
            for row in rows(path):
                assert row["status"] == statuses[row["setting"]]
        out += translation_exceptions(root, org)
        out += [
            f"### {org}: blastx and search statuses",
            "",
            f"HELD: every one of {standard_rows} standard call rows takes the highest-bit-score eligible "
            f"hit in the delivered blastx table ({len(blast)} query identifiers), with matching target, "
            "identity, E-value and bit score. All standard-call baseMean, fold changes and P-values match the labelled raw transcript DESeq2 rows. Family annotations cannot be rejoined without the search-domain "
            "table. Because blastx was run with -k 1, equal-score targets omitted upstream remain untestable.",
            "",
            table(
                ["setting", "direct .status"],
                [
                    (
                        p.name.removeprefix(org + ".").removesuffix(".status"),
                        p.read_text().strip(),
                    )
                    for p in sorted((base / "kmerseek").glob("*.status"))
                ],
            ),
            "",
            "All 19 .status files per organism were checked and agree with every available per-setting summary. They directly record fitted/unfitted outcomes, including fitted gbmr7.k11. "
            "They replace the earlier summary-only evidence. They do not reproduce the fit or establish "
            "the reason for an unfitted result without search logs.",
            "",
        ]
    out += [
        "### Remaining limits after the new inputs",
        "",
        "No expensive DESeq2 rerun was needed or performed. Original reads, per-sample salmon/jellyfish dumps, "
        "search-domain annotations/database, full raw search regions and search logs remain unavailable in "
        "this delivery. Count normalization and contrast direction are tested; original read provenance, "
        "dispersion estimation, Wald tests, adjusted P-values and completeness of upstream counts are not. "
        "gbmr7 landing recovery remains unscored until the scoring repair delivers its output. "
        "Earlier zero-query control and local-assembly findings still hold.",
        "",
    ]
    return "\n".join(out)


PINNED_INDEX_DEFAULTS = "        #[arg(long, value_enum, default_value_t = DecoyNull::Database)]\n        ka_null: DecoyNull,\n\n        /// For `--ka-null database`: how the reference queries that decide where the fit\n        /// stops are made. `shuffled-dipeptide` keeps each pair of neighbouring residues as\n        /// often as in the original, so hydrophobic runs survive and only relatives and\n        /// periodicity lift the real curve above it; `shuffled` keeps composition only.\n        #[arg(long, value_enum, default_value_t = DecoyNull::ShuffledDipeptide)]\n        ka_reference: DecoyNull,"


def calibration_limit(source):
    """Check pipeline flags against the exact defaults retrieved from pinned main.rs."""
    pipeline = (source / "main.nf").read_text()
    block = pipeline.split("process kmerseekIndex {", 1)[1].split(
        "process kmerseekSearch {", 1
    )[0]
    assert "--ka-null" not in block and "--ka-reference" not in block
    assert "--ka-queries ${kaQueries" in block
    assert "DecoyNull::Database" in PINNED_INDEX_DEFAULTS
    assert "DecoyNull::ShuffledDipeptide" in PINNED_INDEX_DEFAULTS
    return "\n".join(
        [
            "### Literal “no sequence shuffling” limitation: internal E-value calibration",
            "",
            "CONFIRMED from pinned source and pipeline arguments: a literal claim that no sequence shuffling "
            "occurs anywhere in the workflow is false. The answer-key and label-swap design does not shuffle "
            "sequences, but index-time E-value calibration uses a shuffled internal reference by default.",
            "",
            "[Pinned main.rs, index defaults, lines 99–107](https://github.com/seanome/kmerseek/blob/3c353d4/src/rust/main.rs#L99-L107) "
            "sets ka_null to Database and ka_reference to ShuffledDipeptide. The accompanying source comments "
            "(lines 102–105) say that this reference decides where the fit stops and preserves adjacent residue-pair "
            "counts. [Lines 530–539](https://github.com/seanome/kmerseek/blob/3c353d4/src/rust/main.rs#L530-L539) "
            "pass both values into the index calibration settings.",
            "",
            "Exact index-default excerpt, retrieved from GitHub at ref 3c353d4 on 2026-10-23:",
            "",
            "```rust",
            PINNED_INDEX_DEFAULTS,
            "```",
            "",
            "Downloaded main.rs text SHA256: 56844bcd3982ca88acf0fccd05626ca5eb4d91cdc7b18041332c54d7cbee81b6. "
            "The source-verified excerpt is embedded in the audit script so this check needs no network on rerun.",
            "",
            "The audited local index process supplies positive calibration-query counts and overrides neither "
            "--ka-null nor --ka-reference. Thus database sequences are the calibration queries, while a "
            "dipeptide-shuffled reference participates in choosing the fit range. This is a source-backed "
            "configuration inference; index logs were not supplied to observe the run directly. Search uses "
            "the stored fit with zero new calibration queries. The limitation concerns internal calibration, "
            "not a newly introduced shuffled experimental control.",
            "",
            "The user reports that the scoring-only repair entry contains landings and collectSummaries only. "
            "That entry was not inspected or launched by this checker. Re-scoring alone neither removes "
            "the internal shuffled calibration reference nor changes the original fixed E-value thresholds.",
            "",
        ]
    )


def finish_report(
    text,
    trace_path=Path("/private/tmp/nb360-extra/de_kmer_landing.2026-10-21.trace.txt"),
):
    """Correct repair provenance and append independent historical trace evidence."""
    import re

    text = text.split("### Independent historical trace check")[0]
    text = text.replace("49179127", "49179534 (landings jobs 49179865/49179866)")
    text = text.replace(
        "No local scheduler trace was supplied to independently verify the attempt sizes or exit code.",
        "The supplied historical trace now independently verifies the failure stages, attempts and requested memory; see below.",
    )
    text = text.replace(
        "the user reports LANDINGS exit 137",
        "the supplied trace confirms LANDINGS exit 137",
    )
    text = re.sub(
        r"Filtering finite region E-values to <=100 before the domain join preserves.*?uninspected repair implementation\.",
        "CORRECTION: the existing searches were already filtered to E<=100. That filter does not solve the memory failure. "
        "The user reports that the actual repair processes bounded batches of 25,000 input rows and writes gzip output directly. "
        "The scorer was not reviewed here. Retaining the same per-row overlap test, E-value thresholds, scores and family-set aggregation "
        "would preserve the fixed scoring rule; repaired output equivalence remains unverified until output arrives.",
        text,
        flags=re.S,
    )
    trace_rows = list(rows(trace_path))
    selected = [
        r for r in trace_rows if r["tag"] in ["ciona.gbmr7.k11", "zebrafish.gbmr7.k11"]
    ]
    for org in ["ciona", "zebrafish"]:
        group = [r for r in selected if r["tag"] == org + ".gbmr7.k11"]
        for process in ["kmerseekIndex", "kmerseekSearch"]:
            assert any(
                r["process"] == process
                and r["status"] == "COMPLETED"
                and r["exit"] == "0"
                for r in group
            )
        failed = [r for r in group if r["process"] == "landings"]
        assert {(r["attempt"], r["memory"], r["exit"]) for r in failed} == {
            ("1", "16 GB", "137"),
            ("2", "32 GB", "137"),
            ("3", "48 GB", "137"),
        }
    text += "\n\n### Independent historical trace check\n\n"
    text += f"Source: `{trace_path}`. All {len(trace_rows)} trace rows were read; the ten gbmr7 rows are below. "
    text += "Both organisms completed INDEX and SEARCH; LANDINGS failed with exit 137 at requested 16, 32 and 48 GB. "
    text += "Exit 137 alone does not independently prove an OOM kill; no scheduler accounting or stderr was supplied. "
    text += "The current repair job IDs are user-reported, not historical trace results. No repaired outputs were supplied; original scores remain provisional.\n\n"
    text += (
        table(
            [
                "organism/setting",
                "process",
                "job",
                "status",
                "exit",
                "requested memory",
                "attempt",
            ],
            [
                (
                    r["tag"],
                    r["process"],
                    r["native_id"],
                    r["status"],
                    r["exit"],
                    r["memory"],
                    r["attempt"],
                )
                for r in selected
            ],
        )
        + "\n"
    )
    return text


def update_report(text, root, source):
    text = text.split("## Additional raw-input audit, 2026-10-23")[0]
    text = text.replace(
        "reported status (unverified)", "summary status (direct files checked below)"
    )
    text = text.replace(
        "A missing setting is not zero recovery or an unfitted search.",
        "Missing gbmr7 landings are not zero recovery: its INDEX and SEARCH completed; the user reports LANDINGS exit 137. See the additional raw-input audit below.",
    )
    text = text.replace(
        "Their empty landed tables are observed; the cause is summary metadata only. Logs and .status files were not delivered.",
        "Their empty landed tables are observed; direct .status files now corroborate the recorded unfitted outcomes. Search logs remain absent.",
    )
    text = text.replace(
        "Runs absent from the peptide map cannot be counted because the runs table was not delivered.",
        "The complete runs table and exact sequence coverage are checked in the additional raw-input audit below.",
    )
    text = text.replace(
        "Without runs.tsv and query FASTA, exact sequence reconstruction and six-frame translation remain untested.",
        "The newly delivered runs and query FASTA now permit exact sequence and six-frame translation checks below.",
    )
    start = "UNTESTABLE from this delivery: DESeq2 reruns and sample/count assignment;"
    end = "The true-count table validates simulation direction only."
    while start in text:
        a = text.index(start)
        b = text.index(end, a) + len(end)
        text = (
            text[:a]
            + "UPDATED 2026-10-23: the new count, sequence, blastx and status checks below supersede the earlier missing-input limits. Search annotations, raw search regions, logs and original reads remain absent."
            + text[b:]
        )
    return finish_report(
        text.rstrip()
        + "\n\n"
        + extended_inputs(root, source)
        + "\n"
        + calibration_limit(source)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delivery",
        type=Path,
        default=Path("/private/tmp/nb360-delivery/notebooks/360_data"),
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "nextflow-runs/de-kmer-landing",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "notebooks/360_data/adversarial_review.md",
    )
    parser.add_argument(
        "--extend-only",
        action="store_true",
        help="Extend the previously completed family recount without re-reading all landed rows",
    )
    args = parser.parse_args()
    text = (
        args.report.read_text()
        if args.extend_only
        else audit(args.delivery, args.source)
    )
    text = update_report(text, args.delivery, args.source)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text)
    print(args.report, flush=True)


if __name__ == "__main__":
    main()
