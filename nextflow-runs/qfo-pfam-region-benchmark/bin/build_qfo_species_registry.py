#!/usr/bin/env python3
"""
Derive assets/qfo_species.tsv -- the species registry -- from a QfO release directory.

Every consumer of "which species are in this benchmark" reads that TSV: ALL_SPECIES in
main.nf, the `proteomes` map in make_mini_testset.py, and SPECIES_METADATA in
notebooks/build_pfam_architectures.py. Three hardcoded copies of a 10-species list is what
this replaces, and the copies had already drifted -- main.nf carried nine targets,
make_mini_testset.py ten (it also holds human), and the notebook ten with different keys.

Everything except `label` and `mya` is read out of the release itself rather than typed:
the proteome ID and taxon come from the file name, the scientific name and taxon from the
first FASTA header's OS= / OX= fields, and the protein count from the record count. Run it
against a different release and the taxon or the protein count changes on its own. The one
exception is EXTRA_ROWS, for proteomes that are not in any release and are staged locally;
see the comment on it.

LABELS ARE CACHE KEYS. kmerseekIndex stores its target indexes under
${db_cache}/kmerseek_index as <label>.<alphabet>.k<k>.lc<lc>.kmerseek.rocksdb, and
kmerseekSearch stores results as human_vs_<label>.<...>.regions.parquet. Neither name
carries a release, a checksum or a query digest, so renaming a species silently orphans
every index and search already computed for it. The ten labels in LOCKED_LABELS are the
ones run-midi, run-midi-plus and the full run have already populated caches under; they
are pinned here and asserted below rather than left to the generator's naming scheme,
which would have turned "worm" into "celegans" and "ecoli" into "ecoli" by luck.

MYA is divergence from human, in millions of years, and it is the x-axis of every
divergence panel in the report. Only the ten species that already had a value carry one.
The other 68 are written empty ON PURPOSE: evaluate_domain_calls.py takes --species-mya as
an optional float defaulting to None, and build_multiqc_inputs.py skips its divergence
panels when species_mya is absent, so an empty cell degrades to "not plotted on the
divergence axis" rather than to a wrong number. Filling them in needs a real source
(TimeTree node ages); guessing them would put invented coordinates on the axis the whole
experiment is about.

Usage:
    python3 bin/build_qfo_species_registry.py \
        --release ~/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143 \
        --out assets/qfo_species.tsv
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# label -> proteome ID, for every label a cache already exists under. See the module
# docstring: these cannot change without orphaning the stores.
LOCKED_LABELS = {
    "human": "UP000005640",
    "mouse": "UP000000589",
    "chicken": "UP000000539",
    "zebrafish": "UP000000437",
    "ciona": "UP000008144",
    "fly": "UP000000803",
    "worm": "UP000001940",
    "yeast": "UP000002311",
    "arabidopsis": "UP000006548",
    "ecoli": "UP000000625",
}

# Divergence from human in MYA, as main.nf's ALL_SPECIES carried them. Reproduced exactly:
# mya reaches scoreDomainCalls as `--species-mya ${mya}` inside the task script, so a
# changed value rehashes every scored arm for that species.
LOCKED_MYA = {
    "human": 0,
    "mouse": 100,
    "chicken": 300,
    "zebrafish": 430,
    "ciona": 550,
    "fly": 600,
    "worm": 650,
    "yeast": 900,
    "arabidopsis": 1500,
    "ecoli": 2000,
}

# Species that are NOT in the QfO release and are staged locally instead. The release scan
# below cannot see them, so without this list every regeneration would drop them and the
# species would silently un-add itself -- the same class of failure LOCKED_LABELS guards
# against, arriving from the other direction. A label is a cache key whether or not the
# sequences came from the release, so a proteome staged by hand still has to survive being
# regenerated.
#
# `proteome` is a filename component, not a UniProt accession: main.nf builds the path as
# ${qfo_dir}/${subdir}/${proteome}_${taxon}.fasta, so BOTSCH2026 is a local id chosen to be
# obviously not a UP-accession. `make stage-botryllus` writes that file.
#
# botryllus is a TARGET-ONLY species. It has no Pfam annotations, so build_domain_truth.py
# writes no domain map for it and main.nf's score_in inner-joins it out of scoring: it is
# searched by every sequence arm and by ProstT5, and produces no metrics. mya 550 puts it at
# the same node as ciona on purpose -- two tunicates at one divergence is the comparison the
# species was added for.
EXTRA_ROWS = [
    {
        "label": "botryllus",
        "taxon": "30301",
        "proteome": "BOTSCH2026",
        "subdir": "Eukaryota",
        "mya": 550,
        "n_proteins": 45339,
        "scientific_name": "Botryllus schlosseri",
    },
]

# Where the generated genus-initial + epithet scheme produces something useless. Only
# strain-designator names land here: "Synechocystis sp. (strain PCC 6803)" has no epithet
# to abbreviate, so the scheme yields "ssp".
LABEL_OVERRIDES = {
    "UP000001425": "synechocystis",
}

KINGDOMS = ("Archaea", "Bacteria", "Eukaryota")


def first_header(fasta: Path) -> str:
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                return line
    raise ValueError(f"no FASTA records in {fasta}")


def count_records(fasta: Path) -> int:
    n = 0
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                n += 1
    return n


def binomial(header: str) -> str:
    """Scientific name from a UniProt header's OS= field, minus parenthesised synonyms."""
    m = re.search(r"OS=(.*?)(?: OX=| GN=| PE=| SV=|$)", header)
    if not m:
        raise ValueError(f"no OS= field in header: {header[:120]}")
    return re.sub(r"\s*\(.*", "", m.group(1)).strip()


def taxon_from_header(header: str) -> str | None:
    m = re.search(r"OX=(\d+)", header)
    return m.group(1) if m else None


def make_label(proteome: str, name: str) -> str:
    """Genus initial + species epithet, e.g. Mycobacterium tuberculosis -> mtuberculosis."""
    if proteome in LABEL_OVERRIDES:
        return LABEL_OVERRIDES[proteome]
    parts = name.split()
    epithet = re.sub(r"[^a-z]", "", parts[1].lower()) if len(parts) > 1 else ""
    if epithet in ("", "sp", "spp"):
        epithet = re.sub(r"[^a-z]", "", parts[-1].lower())
    return parts[0][0].lower() + epithet


def canonical_fastas(release: Path):
    """The canonical proteome FASTAs, excluding the isoform and DNA companions.

    Also excluding the EXTRA_ROWS proteomes. `make stage-botryllus` writes its FASTA INTO
    the release directory, because ${qfo_dir}/${subdir}/${proteome}_${taxon}.fasta is the
    only path main.nf can resolve. That puts a file here whose headers are `>FUN000001_...`
    with no OS= field, and binomial() raises on it -- so a staged local proteome would make
    this generator crash rather than regenerate. It is described by EXTRA_ROWS instead.
    """
    skip = {f"{r['proteome']}_{r['taxon']}.fasta" for r in EXTRA_ROWS}
    for kingdom in KINGDOMS:
        d = release / kingdom
        if not d.is_dir():
            continue
        for fasta in sorted(d.glob("*.fasta")):
            if fasta.name.endswith(("_additional.fasta", "_DNA.fasta")):
                continue
            if fasta.name in skip:
                continue
            yield kingdom, fasta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release", type=Path, required=True,
                   help="QfO release directory holding Archaea/ Bacteria/ Eukaryota/")
    p.add_argument("--out", type=Path, required=True, help="TSV to write")
    args = p.parse_args()

    rows = []
    for kingdom, fasta in canonical_fastas(args.release):
        proteome, taxon_from_name = fasta.stem.rsplit("_", 1)
        header = first_header(fasta)
        name = binomial(header)
        taxon = taxon_from_header(header) or taxon_from_name
        # The two disagree only if a file was renamed by hand, which is worth stopping for:
        # taxon is what build_pfam_architectures.py queries UniProt with.
        if taxon != taxon_from_name:
            print(f"WARNING: {fasta.name} names taxon {taxon_from_name} but its header "
                  f"says OX={taxon}; using the header", file=sys.stderr)
        label = LOCKED_LABELS_INV.get(proteome) or make_label(proteome, name)
        rows.append({
            "label": label,
            "taxon": taxon,
            "proteome": proteome,
            "subdir": kingdom,
            "mya": LOCKED_MYA.get(label, ""),
            "n_proteins": count_records(fasta),
            "scientific_name": name,
        })

    # Locally staged proteomes, appended before the duplicate-label check so they are
    # covered by it -- an EXTRA_ROWS label that collided with a release label would
    # otherwise point two species at one cache. n_proteins is counted from the staged file
    # when it is there, so the recorded number cannot drift from what is actually indexed;
    # the recorded value is the fallback for a machine where the file has not been staged.
    for extra in EXTRA_ROWS:
        row = dict(extra)
        staged = args.release / row["subdir"] / f"{row['proteome']}_{row['taxon']}.fasta"
        if staged.exists():
            row["n_proteins"] = count_records(staged)
        else:
            print(f"NOTE: {staged.name} is not staged in {args.release}; writing the "
                  f"recorded n_proteins={row['n_proteins']} for '{row['label']}'. "
                  f"Stage it with `make stage-botryllus`.", file=sys.stderr)
        rows.append(row)

    labels = [r["label"] for r in rows]
    dups = sorted({l for l in labels if labels.count(l) > 1})
    if dups:
        raise SystemExit(f"repeated labels {dups}: labels name cache entries and must be "
                         f"unique. Add an entry to LABEL_OVERRIDES.")

    # The locked labels have to survive verbatim, and has to be the species they were
    # locked to. A release that dropped or renumbered one of them would otherwise write a
    # registry that quietly points an existing cache at different sequences.
    by_label = {r["label"]: r for r in rows}
    for label, proteome in LOCKED_LABELS.items():
        got = by_label.get(label)
        if got is None:
            raise SystemExit(f"locked label '{label}' ({proteome}) is not in {args.release}")
        if got["proteome"] != proteome:
            raise SystemExit(f"locked label '{label}' is {got['proteome']} in this release, "
                             f"not {proteome}; every cache under that label is stale")

    if "human" not in by_label:
        raise SystemExit("no human proteome in the release; human is the query")

    rows.sort(key=lambda r: (KINGDOMS.index(r["subdir"]), r["label"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, delimiter="\t", lineterminator="\n",
                           fieldnames=["label", "taxon", "proteome", "subdir", "mya",
                                       "n_proteins", "scientific_name"])
        w.writeheader()
        w.writerows(rows)

    n_mya = sum(1 for r in rows if r["mya"] != "")
    print(f"{args.out}: {len(rows)} species ({len(rows) - 1} targets, human is the query), "
          f"{n_mya} with a divergence time")
    return 0


LOCKED_LABELS_INV = {v: k for k, v in LOCKED_LABELS.items()}

if __name__ == "__main__":
    sys.exit(main())
