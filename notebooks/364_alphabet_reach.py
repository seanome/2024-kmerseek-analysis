"""Notebook 364: how far up the tree of life each sequence alphabet still finds shared k-mers.

For each anchor in PATHS (Mus musculus, the mussel Mytilus edulis, the oyster Magallana
gigas, the bacterium Escherichia coli K-12) and a relative at each taxonomic rank, sketch every proteome and coding-sequence (CDS) set with the sourmash
branchwater plugin and measure containment (the fraction of one species' sampled k-mers
that also occur in the other).

k-sizes are matched on bits per k-mer, k * log2(alphabet size), at 30 and 42 bits:
    30 bits: DNA k=15, protein k=7,  dayhoff k=12, hp k=30
    42 bits: DNA k=21, protein k=10, dayhoff k=16, hp k=42
Protein-space alphabets are sketched from the proteome; DNA from the CDS of the same
genes (UniProt's *_DNA.fasta.gz next to each reference proteome).

Subcommands, in run order:

    resolve   (laptop, small API calls)  -> taxonomic_path.tsv
    fetch     (Sherlock job)             -> downloads, fills download_date
    entropy   (Sherlock job)             -> bits_per_kmer.tsv (measured letter entropy)
    sketch    (Sherlock job)             -> sigs/proteome.sig.zip, sigs/cds.sig.zip
                                            (`sourmash scripts manysketch`)
    compare   (Sherlock job)             -> containment.tsv (`sourmash scripts multisearch`)
    reach     (laptop)                   -> reach.tsv
    figure    (laptop)                   -> reach_by_alphabet_<input>.png / .svg / .pdf

Reach rule, fixed before the run (2026-11-08, from the task text):
    floor(alphabet, k, input type, anchor) = max(c_yeast, c_arabidopsis)
                                             + 3 * |c_yeast - c_arabidopsis|
    where c is the max of the two containment directions. For the E. coli anchor the
    two other-kingdom species are Bacillus subtilis and Thermotoga maritima (set
    2026-11-08, before its run).
    reach =the most distant rank (excluding the two different-kingdom species) whose
    max containment is strictly above the floor. Closer ranks may still be below the
    floor; such a gap is reported in reach.tsv, not hidden.
"""

import argparse
import collections
import csv
import datetime
import gzip
import json
import math
import os
import re
import shutil
import string
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("NB364_OUT", HERE / "364_alphabet_reach"))

ALPHABET_SIZE = {"DNA": 4, "protein": 20, "dayhoff": 6, "hp": 2}
# (alphabet, k, bits level); bits_uniform = k * log2(alphabet size).
SETTINGS = [
    ("DNA", 15, 30),
    ("DNA", 21, 42),
    ("protein", 7, 30),
    ("protein", 10, 42),
    ("dayhoff", 12, 30),
    ("dayhoff", 16, 42),
    ("hp", 30, 30),
    ("hp", 42, 42),
]
# k sweeps for dayhoff and hp (olgabot, 2026-11-08), reported next to the 30/42-bit
# settings above; bits level "sweep". With the 30/42-bit k they give dayhoff k = 12, 15,
# 16, 17, 19, 21 and hp k = 24, 27, 30, 33, 36, 39, 42.
SWEEP = [("dayhoff", k, "sweep") for k in (15, 17, 19, 21)] + [
    ("hp", k, "sweep") for k in (24, 27, 33, 36, 39)
]
# Codon ladder (olgabot, 2026-11-09): DNA k = 3 x protein k, so both cover the same codons;
# dayhoff and hp k matched to protein k in bits, rounded to the nearest whole k.
LADDER_PROTEIN_K = (7, 8, 9, 10, 11, 12)


def bit_matched_k(alphabet, protein_k):
    return round(protein_k * math.log2(20) / math.log2(ALPHABET_SIZE[alphabet]))


def ladder(protein_k):
    return [
        ("DNA", 3 * protein_k),
        ("protein", protein_k),
        ("dayhoff", bit_matched_k("dayhoff", protein_k)),
        ("hp", bit_matched_k("hp", protein_k)),
    ]


_have = {(a, k) for a, k, _ in SETTINGS + SWEEP}
LADDER = [
    (a, k, "ladder")
    for pk in LADDER_PROTEIN_K
    for a, k in ladder(pk)
    if (a, k) not in _have
]
ALL_SETTINGS = SETTINGS + SWEEP + LADDER
SCALED = {"DNA": 1000, "protein": 200, "dayhoff": 200, "hp": 200}
# NB364_SCALED=<n> sets one scaled value for every alphabet. Its sketches, comparisons
# and results go in folders named scaled<n>, so the default run's files are kept.
ONE_SCALED = os.environ.get("NB364_SCALED")
if ONE_SCALED:
    SCALED = dict.fromkeys(SCALED, int(ONE_SCALED))
RES = OUT / f"scaled{ONE_SCALED}" if ONE_SCALED else OUT


def run_dir(work, name):
    return Path(work) / (f"{name}_scaled{ONE_SCALED}" if ONE_SCALED else name)


INPUT_TYPES = ("proteome", "cds")


def bits_uniform(alphabet, k):
    return k * math.log2(ALPHABET_SIZE[alphabet])


def settings_for(input_type):
    return [s for s in ALL_SETTINGS if (s[0] == "DNA") == (input_type == "cds")]


def input_type_of(alphabet):
    return "cds" if alphabet == "DNA" else "proteome"


# sourmash 4.9.4 src/core/src/encodings.rs, DAYHOFFTABLE and HPTABLE.
DAYHOFF = {
    **dict.fromkeys("C", "a"),
    **dict.fromkeys("AGPST", "b"),
    **dict.fromkeys("DENQ", "c"),
    **dict.fromkeys("HKR", "d"),
    **dict.fromkeys("ILMV", "e"),
    **dict.fromkeys("FWY", "f"),
}
HP = {**dict.fromkeys("AFGILMPVWY", "h"), **dict.fromkeys("NCSTDERHKQ", "p")}


def floor_species(anchor):
    """The anchor's two other-kingdom species, whose containment sets its noise floor."""
    return [e[1] for e in PATHS[anchor] if e[3] == "other kingdom"]


# requested_species, used_species, taxid, requested_rank, UniProt proteome (None: NCBI
# protein set), NCBI assembly, substitution note.
# The assembly is the one UniProt built the proteome from (UniProt genomeAssembly field).
# CDS come from the same place as the proteins: UniProt's *_DNA.fasta.gz for a reference
# proteome, NCBI's cds_from_genomic for an NCBI protein set.
PATHS = {
    "Mus musculus": [
        (
            "Mus musculus",
            "Mus musculus",
            10090,
            "anchor",
            "UP000000589",
            "GCA_000001635.9",
            "",
        ),
        (
            "Mus spretus",
            "Mus caroli",
            10089,
            "genus",
            "UP000515126",
            "GCF_900094665.2",
            "Mus spretus has no UniProt proteome (searched organism_id:10096, 2026-11-08); "
            "Mus caroli is the only other Mus species with a reference proteome",
        ),
        (
            "Rattus norvegicus",
            "Rattus norvegicus",
            10116,
            "family",
            "UP000002494",
            "GCA_036323735.1",
            "",
        ),
        (
            "Cavia porcellus",
            "Cavia porcellus",
            10141,
            "order",
            "UP000005447",
            "GCA_000151735.1",
            "",
        ),
        (
            "Rhinolophus sinicus",
            "Rhinolophus ferrumequinum",
            59479,
            "class",
            "UP000472240",
            "GCA_004115265.2",
            "Rhinolophus sinicus has no UniProt proteome (only viruses match, 2026-11-08); "
            "R. ferrumequinum is the same genus with a reference proteome. Homo sapiens not used: "
            "it shares superorder Euarchontoglires with mouse, closer than class",
        ),
        (
            "Danio rerio",
            "Danio rerio",
            7955,
            "phylum",
            "UP000000437",
            "GCF_049306965.1",
            "",
        ),
        (
            "Drosophila melanogaster",
            "Drosophila melanogaster",
            7227,
            "kingdom",
            "UP000000803",
            "GCA_000001215.4",
            "",
        ),
        (
            "Saccharomyces cerevisiae",
            "Saccharomyces cerevisiae",
            559292,
            "other kingdom",
            "UP000002311",
            "GCA_000146045.2",
            "strain S288c (taxid 559292), the UniProt reference proteome",
        ),
        (
            "Arabidopsis thaliana",
            "Arabidopsis thaliana",
            3702,
            "other kingdom",
            "UP000006548",
            "GCA_000001735.1",
            "",
        ),
    ],
    # Bivalve anchor (added 2026-11-08 in place of Botryllus, at olgabot's request). One
    # relative per rank that has a UniProt reference proteome; no other Mytilidae or
    # Mytilida species has one, so family and order are missing.
    "Mytilus edulis": [
        (
            "Mytilus edulis",
            "Mytilus edulis",
            6550,
            "anchor",
            "UP000683360",
            "GCA_905397895.1",
            "",
        ),
        (
            "Mytilus coruscus",
            "Mytilus coruscus",
            42192,
            "genus",
            "UP000507470",
            "GCA_011752425.2",
            "",
        ),
        (
            "Magallana gigas",
            "Magallana gigas",
            29159,
            "subclass",
            "UP000005408",
            "GCA_902806645.1",
            "",
        ),
        (
            "Solemya velum",
            "Solemya velum",
            13268,
            "class",
            "UP001724931",
            "GCA_048127485.1",
            "",
        ),
        (
            "Octopus sinensis",
            "Octopus sinensis",
            2607531,
            "phylum",
            "UP000515154",
            "GCF_006345805.1",
            "",
        ),
        (
            "Capitella teleta",
            "Capitella teleta",
            283909,
            "kingdom",
            "UP000014760",
            "GCA_000328365.1",
            "",
        ),
        (
            "Saccharomyces cerevisiae",
            "Saccharomyces cerevisiae",
            559292,
            "other kingdom",
            "UP000002311",
            "GCA_000146045.2",
            "strain S288c (taxid 559292), the UniProt reference proteome",
        ),
        (
            "Arabidopsis thaliana",
            "Arabidopsis thaliana",
            3702,
            "other kingdom",
            "UP000006548",
            "GCA_000001735.1",
            "",
        ),
    ],
    # Bacterial anchor (added 2026-11-08). Two strains of one species give a species rank,
    # which no animal ladder here has. The floor is two bacteria from other NCBI kingdoms
    # (Bacillati, Thermotogati), as yeast and Arabidopsis are for the animals.
    # Pacific oyster (olgabot, 2026-11-09): the bivalve anchor once only proteomes at least
    # 90% complete by UniProt BUSCO are used (Mytilus edulis is 80.1%). No reference
    # proteome in its genus or order reaches 90%, so those ranks have no relative.
    "Magallana gigas": [
        (
            "Magallana gigas",
            "Magallana gigas",
            29159,
            "anchor",
            "UP000005408",
            "GCA_902806645.1",
            "",
        ),
        (
            "Magallana angulata",
            "Magallana angulata",
            2784310,
            "genus",
            None,
            "GCF_025612915.1",
            "no UniProt proteome in genus Magallana besides M. gigas (2026-11-09); NCBI RefSeq annotation, BUSCO 98.8%",
        ),
        (
            "Crassostrea virginica",
            "Crassostrea virginica",
            6565,
            "family",
            None,
            "GCF_002022765.2",
            "UniProt UP000694844 ships a CDS file of 7 sequences (3 kb, 2026-11-09), so proteins "
            "and CDS come from the same NCBI RefSeq annotation it was built from (BUSCO 98.1%)",
        ),
        (
            "Mizuhopecten yessoensis",
            "Mizuhopecten yessoensis",
            6573,
            "order",
            "UP000242188",
            "GCA_002113885.2",
            "NCBI order Ostreida holds only oysters, so no genome shares the order but not the family; "
            "the scallop shares the clade Pteriomorphia, the nearest group above order (BUSCO 88.9%)",
        ),
        (
            "Mactra antiquata",
            "Mactra antiquata",
            2302425,
            "class",
            "UP001697639",
            "GCA_047288015.1",
            "",
        ),
        (
            "Sinanodonta woodiana",
            "Sinanodonta woodiana",
            1069815,
            "class",
            "UP001634394",
            "GCA_046254885.1",
            "",
        ),
        (
            "Solemya velum",
            "Solemya velum",
            13268,
            "class",
            "UP001724931",
            "GCA_048127485.1",
            "the only class-level bivalve (outside subclass Autobranchia) with a reference proteome (BUSCO 82.5%)",
        ),
        (
            "Aplysia californica",
            "Aplysia californica",
            6500,
            "phylum",
            "UP000694888",
            "GCF_000002075.1",
            "",
        ),
        (
            "Capitella teleta",
            "Capitella teleta",
            283909,
            "kingdom",
            "UP000014760",
            "GCA_000328365.1",
            "",
        ),
        (
            "Saccharomyces cerevisiae",
            "Saccharomyces cerevisiae",
            559292,
            "other kingdom",
            "UP000002311",
            "GCA_000146045.2",
            "strain S288c (taxid 559292), the UniProt reference proteome",
        ),
        (
            "Arabidopsis thaliana",
            "Arabidopsis thaliana",
            3702,
            "other kingdom",
            "UP000006548",
            "GCA_000001735.1",
            "",
        ),
    ],
    "Escherichia coli K-12": [
        (
            "Escherichia coli K-12",
            "Escherichia coli K-12",
            83333,
            "anchor",
            "UP000000625",
            "GCA_000005845.2",
            "",
        ),
        (
            "Escherichia coli O157:H7",
            "Escherichia coli O157:H7",
            83334,
            "species",
            "UP000000558",
            "GCA_000008865.2",
            "",
        ),
        (
            "Escherichia fergusonii",
            "Escherichia fergusonii",
            585054,
            "genus",
            "UP000000745",
            "GCA_000026225.1",
            "",
        ),
        (
            "Salmonella enterica",
            "Salmonella enterica",
            99287,
            "family",
            "UP000001014",
            "GCA_000006945.2",
            "serovar Typhimurium strain LT2 (taxid 99287), the UniProt reference proteome",
        ),
        (
            "Yersinia pestis",
            "Yersinia pestis",
            632,
            "order",
            "UP000000815",
            "GCA_000009065.1",
            "",
        ),
        (
            "Pseudomonas aeruginosa",
            "Pseudomonas aeruginosa",
            208964,
            "class",
            "UP000002438",
            "GCA_000006765.1",
            "strain PAO1 (taxid 208964)",
        ),
        (
            "Caulobacter vibrioides",
            "Caulobacter vibrioides",
            190650,
            "phylum",
            "UP000001816",
            "GCA_000006905.1",
            "strain CB15 (taxid 190650)",
        ),
        (
            "Bacteroides thetaiotaomicron",
            "Bacteroides thetaiotaomicron",
            226186,
            "kingdom",
            "UP000001414",
            "GCA_000011065.1",
            "strain VPI-5482 (taxid 226186)",
        ),
        (
            "Bacillus subtilis",
            "Bacillus subtilis",
            224308,
            "other kingdom",
            "UP000001570",
            "GCA_000009045.1",
            "strain 168 (taxid 224308)",
        ),
        (
            "Thermotoga maritima",
            "Thermotoga maritima",
            243274,
            "other kingdom",
            "UP000008183",
            "GCA_000008545.1",
            "strain MSB8 (taxid 243274)",
        ),
    ],
}

# Ranks from nearest to most distant, for ordering the shared ranks NCBI reports.
RANK_ORDER = [
    "species",
    "subgenus",
    "genus",
    "subfamily",
    "family",
    "superfamily",
    "infraorder",
    "suborder",
    "order",
    "superorder",
    "cohort",
    "infraclass",
    "subclass",
    "class",
    "superclass",
    "subphylum",
    "phylum",
    "superphylum",
    "subkingdom",
    "kingdom",
    "other kingdom",
]

COLUMNS = [
    "anchor",
    "relative",
    "requested_species",
    "requested_rank",
    "rank",
    "shared_taxon",
    "lowest_shared_clade",
    "lineage_depth",
    "taxid",
    "uniprot_proteome_id",
    "uniprot_release",
    "proteome_url",
    "ncbi_assembly",
    "cds_url",
    "substitution",
    "download_date",
]


def get(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read(), dict(r.headers)
        except Exception:
            if attempt == retries - 1:
                raise


def quality(args):
    """UniProt's completeness report for every proteome used: BUSCO (share of a lineage's
    expected single-copy genes found) and CPD (protein count against close relatives).

    Writes proteome_quality.tsv. A gappy proteome shares fewer k-mers with everything,
    so it can pull a relative's containment down; this table is the check for that.
    """
    rows, seen = [], set()
    for t in read_tsv(OUT / "taxonomic_path.tsv"):
        up = t["uniprot_proteome_id"]
        if not up and t["ncbi_assembly"] not in seen:
            seen.add(t["ncbi_assembly"])
            body, _ = get(
                f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{t['ncbi_assembly']}/dataset_report"
            )
            ann = json.loads(body)["reports"][0].get("annotation_info", {})
            busco = ann.get("busco", {})
            rows.append(
                {
                    "species": t["relative"],
                    "uniprot_proteome_id": "",
                    "busco_lineage": busco.get("busco_lineage", "")
                    + " (NCBI annotation)",
                    "busco_complete_pct": (
                        f"{100 * busco['complete']:.1f}" if busco else ""
                    ),
                    "busco_fragmented_pct": (
                        f"{100 * busco['fragmented']:.1f}" if busco else ""
                    ),
                    "busco_missing_pct": (
                        f"{100 * busco['missing']:.1f}" if busco else ""
                    ),
                    "cpd_status": "",
                    "assembly_level": t["ncbi_assembly"],
                    "retrieved": datetime.date.today().isoformat(),
                }
            )
            print(
                t["ncbi_assembly"],
                t["relative"],
                rows[-1]["busco_complete_pct"],
                flush=True,
            )
            continue
        if not up or up in seen:
            continue
        seen.add(up)
        body, _ = get(f"https://rest.uniprot.org/proteomes/{up}.json")
        d = json.loads(body)
        rep = d.get("proteomeCompletenessReport") or {}
        busco = rep.get("buscoReport") or {}
        cpd = rep.get("cpdReport") or {}
        total = busco.get("total")
        rows.append(
            {
                "species": t["relative"],
                "uniprot_proteome_id": up,
                "busco_lineage": busco.get("lineageDb", ""),
                "busco_complete_pct": (
                    f"{100 * busco['complete'] / total:.1f}" if total else ""
                ),
                "busco_fragmented_pct": (
                    f"{100 * busco['fragmented'] / total:.1f}" if total else ""
                ),
                "busco_missing_pct": (
                    f"{100 * busco['missing'] / total:.1f}" if total else ""
                ),
                "cpd_status": cpd.get("status", ""),
                "assembly_level": (d.get("genomeAssembly") or {}).get("level", ""),
                "retrieved": datetime.date.today().isoformat(),
            }
        )
        print(up, t["relative"], rows[-1]["busco_complete_pct"], flush=True)
    write_tsv(OUT / "proteome_quality.tsv", rows, list(rows[0]))


def lineage(taxid):
    """Ranked lineage [(rank, name)] from NCBI Taxonomy, root first, the taxon itself last."""
    body, _ = get(
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=taxonomy&id={taxid}&retmode=xml"
    )
    taxon = ET.fromstring(body).find("Taxon")
    out = [
        (t.findtext("Rank"), t.findtext("ScientificName"))
        for t in taxon.find("LineageEx")
    ]
    return out + [(taxon.findtext("Rank"), taxon.findtext("ScientificName"))]


def shared_rank(anchor_lin, rel_lin):
    """Lowest shared ranked taxon, lowest shared taxon of any kind, and that taxon's depth.

    Ranked: unranked clades ("clade", "no rank") skipped; two kingdoms share only the
    domain (Eukaryota, Bacteria), called "other kingdom". Depth is the position of the lowest shared
    taxon in the anchor's full lineage, so a larger depth means a closer relative even when
    two relatives share the same rank. Species counts as a shared rank, for two strains of
    one species.
    """
    rel = set(rel_lin)
    depth = max(i for i, t in enumerate(anchor_lin) if t in rel)
    clade = anchor_lin[depth][1]
    for rank, name in reversed(anchor_lin):
        if rank in RANK_ORDER and (rank, name) in rel:
            return rank, name, clade, depth
    domain = next(n for r, n in anchor_lin if r in ("superkingdom", "domain"))
    return "other kingdom", domain, clade, depth


def genome_url(acc):
    body, _ = get(
        f"https://api.ncbi.nlm.nih.gov/datasets/v2/genome/accession/{acc}/dataset_report?filters.assembly_version=all_assemblies"
    )
    rep = json.loads(body)["reports"][0]
    if rep["accession"] != acc:
        sys.exit(f"{acc}: NCBI returned {rep['accession']}")
    name = rep["assembly_info"]["assembly_name"].replace(" ", "_")
    d = acc[4:13]
    return f"https://ftp.ncbi.nlm.nih.gov/genomes/all/{acc[:3]}/{d[:3]}/{d[3:6]}/{d[6:9]}/{acc}_{name}/{acc}_{name}_genomic.fna.gz"


def resolve(args):
    rows = []
    for anchor, entries in PATHS.items():
        anchor_lin = lineage(entries[0][2])
        for req, used, taxid, req_rank, up, acc, note in entries:
            lin = lineage(taxid)
            if lin[-1][1].split(" (")[0] != used and not lin[-1][1].startswith(used):
                sys.exit(f"taxid {taxid} is {lin[-1][1]}, expected {used}")
            if req_rank == "anchor":
                rank, taxon, clade, depth = "anchor", used, used, len(anchor_lin)
            else:
                rank, taxon, clade, depth = shared_rank(anchor_lin, lin)
            if up:
                body, hdr = get(f"https://rest.uniprot.org/proteomes/{up}?format=json")
                p = json.loads(body)
                if (
                    p["taxonomy"]["taxonId"] != taxid
                    or p["proteomeType"] != "Reference proteome"
                ):
                    sys.exit(
                        f"{up}: taxid {p['taxonomy']['taxonId']}, type {p['proteomeType']}"
                    )
                built = p.get("genomeAssembly", {}).get("assemblyId", "")
                if built != acc and not (
                    built.split(".")[0] == acc.split(".")[0] and note
                ):
                    sys.exit(
                        f"{up} was built from {built}, table says {acc}, and no note says why"
                    )
                release = hdr.get("X-UniProt-Release", "")
                domain = p["superkingdom"].capitalize()
                purl = (
                    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
                    f"reference_proteomes/{domain}/{up}/{up}_{taxid}.fasta.gz"
                )
                curl = purl.replace(".fasta.gz", "_DNA.fasta.gz")
            else:
                release = ""
                purl = genome_url(acc).replace("_genomic.fna.gz", "_protein.faa.gz")
                curl = genome_url(acc).replace(
                    "_genomic.fna.gz", "_cds_from_genomic.fna.gz"
                )
            rows.append(
                dict(
                    anchor=anchor,
                    relative=used,
                    requested_species=req,
                    requested_rank=req_rank,
                    rank=rank,
                    shared_taxon=taxon,
                    lowest_shared_clade=clade,
                    lineage_depth=depth,
                    taxid=taxid,
                    uniprot_proteome_id=up or "",
                    uniprot_release=release,
                    proteome_url=purl,
                    ncbi_assembly=acc,
                    cds_url=curl,
                    substitution=note,
                    download_date="",
                )
            )
            print(
                anchor,
                "|",
                used,
                "|",
                req_rank,
                "->",
                rank,
                taxon,
                "| clade",
                clade,
                depth,
                flush=True,
            )
    write_tsv(OUT / "taxonomic_path.tsv", rows, COLUMNS)


def write_tsv(path, rows, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def read_tsv(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def slug(species):
    return re.sub(r"[^A-Za-z0-9]+", "_", species)


def species_files():
    """{species: (proteome_url, cds_url)} with each species once."""
    return {
        r["relative"]: (r["proteome_url"], r["cds_url"])
        for r in read_tsv(OUT / "taxonomic_path.tsv")
    }


def raw_path(work, sp, seq):
    return Path(work) / "raw" / f"{slug(sp)}.{seq}.gz"


def download(url, dest):
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            with gzip.open(tmp) as g:  # full read: a truncated download fails here
                while g.read(1 << 24):
                    pass
            tmp.rename(dest)
            return
        except Exception as e:
            print(f"{url}: attempt {attempt + 1} failed: {e}", flush=True)
    sys.exit(f"could not download {url}")


def fetch(args):
    """Download every proteome and CDS set once; record the UTC date of each download."""
    (Path(args.work) / "raw").mkdir(parents=True, exist_ok=True)
    dates = {}
    for sp, urls in species_files().items():
        for seq, url in zip(INPUT_TYPES, urls):
            dest = raw_path(args.work, sp, seq)
            if not dest.exists():
                download(url, dest)
            print(sp, seq, dest.stat().st_size, flush=True)
        mtime = raw_path(args.work, sp, "proteome").stat().st_mtime
        dates[sp] = (
            datetime.datetime.fromtimestamp(mtime, datetime.timezone.utc)
            .date()
            .isoformat()
        )
    # A CDS set should hold about 3 bases per residue of its proteome; UniProt's
    # Crassostrea virginica file held 7 sequences (2026-11-09), so check every one.
    for sp in species_files():
        aa = sum(letter_counts(raw_path(args.work, sp, "proteome"), dna=False).values())
        nt = sum(letter_counts(raw_path(args.work, sp, "cds"), dna=True).values())
        if nt < 1.5 * aa:
            sys.exit(
                f"{sp}: CDS has {nt} bases for {aa} residues; expected about {3 * aa}"
            )
    path = OUT / "taxonomic_path.tsv"
    rows = read_tsv(path)
    for r in rows:
        r["download_date"] = dates[r["relative"]]
    write_tsv(path, rows, COLUMNS)


def letter_counts(path, dna):
    """Letter counts of a gzipped FASTA: A/C/G/T for DNA, the 20 amino acids otherwise."""
    keep = b"ACGT" if dna else b"ACDEFGHIKLMNPQRSTVWY"
    counts = collections.Counter()
    with gzip.open(path) as f:
        for line in f:
            if line.startswith(b">"):
                continue
            line = line.strip().upper()
            for c in keep:
                counts[chr(c)] += line.count(bytes([c]))
    return counts


def shannon(counts):
    n = sum(counts.values())
    return -sum(c / n * math.log2(c / n) for c in counts.values() if c), n


def entropy(args):
    """Measured Shannon entropy per letter of each anchor's proteome (per alphabet) and CDS."""
    rows = []
    for anchor in PATHS:
        aa = letter_counts(raw_path(args.work, anchor, "proteome"), dna=False)
        nt = letter_counts(raw_path(args.work, anchor, "cds"), dna=True)
        by_alphabet = {"DNA": nt, "protein": aa}
        for name, table in (("dayhoff", DAYHOFF), ("hp", HP)):
            mapped = collections.Counter()
            for letter, c in aa.items():
                mapped[table[letter]] += c
            by_alphabet[name] = mapped
        for alphabet, k, level in ALL_SETTINGS:
            h, n = shannon(by_alphabet[alphabet])
            rows.append(
                dict(
                    anchor=anchor,
                    alphabet=alphabet,
                    k=k,
                    bits_level=level,
                    source=input_type_of(alphabet),
                    letters_counted=n,
                    entropy_per_letter=f"{h:.4f}",
                    bits_uniform=f"{bits_uniform(alphabet, k):.2f}",
                    bits_measured=f"{k * h:.2f}",
                )
            )
        print(anchor, "entropy done", flush=True)
    write_tsv(OUT / "bits_per_kmer.tsv", rows, list(rows[0]))


def sig_path(work, input_type):
    """One zip per input type holding every species' sketches, named by species."""
    return run_dir(work, "sigs") / f"{input_type}.sig.zip"


def param_string(alphabet, k):
    moltype = "dna" if alphabet == "DNA" else alphabet
    return f"k={k},scaled={SCALED[alphabet]},{moltype}"


def sketch(args):
    """`sourmash scripts manysketch` (branchwater): proteomes in protein space, CDS as DNA.

    manysketch reads the FASTA column that matches each moltype: protein_filename for
    protein, dayhoff and hp; genome_filename for DNA. A file in the other column is
    skipped, so each input type gets its own CSV.
    """
    work = Path(args.work)
    run_dir(work, "sigs").mkdir(parents=True, exist_ok=True)
    for input_type in INPUT_TYPES:
        out = sig_path(work, input_type)
        if out.exists():
            import zipfile

            # sourmash cannot open a zip over 4 GiB (see load_sigs): read the manifest.
            with zipfile.ZipFile(out) as z:
                lines = z.read("SOURMASH-MANIFEST.csv").decode().splitlines()
            have = {
                (row["name"], row["moltype"], int(row["ksize"]))
                for row in csv.DictReader(
                    line for line in lines if not line.startswith("#")
                )
            }
            want = {
                (sp, alphabet, k)
                for sp in species_files()
                for alphabet, k, _ in settings_for(input_type)
            }
            if have == want:
                print(out, "exists", flush=True)
                continue
            print(out, "holds other species or settings; rebuilding", flush=True)
            out.unlink()
        listing = run_dir(work, "sigs") / f"{input_type}.manysketch.csv"
        with open(listing, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["name", "genome_filename", "protein_filename"])
            for sp in sorted(species_files()):
                fasta = str(raw_path(work, sp, input_type))
                w.writerow([sp, fasta, ""] if input_type == "cds" else [sp, "", fasta])
        part = out.with_name(out.name.replace(".sig.zip", ".part.sig.zip"))
        cmd = ["sourmash", "scripts", "manysketch", str(listing), "-o", str(part)]
        for alphabet, k, _ in settings_for(input_type):
            cmd += ["-p", param_string(alphabet, k)]
        cmd += ["-c", str(args.cores)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        os.rename(part, out)


def load_sigs(work, input_type, alphabet, k):
    """{species: SourmashSignature} for one alphabet and k, each species exactly once."""
    import gzip
    import zipfile

    import sourmash

    # sourmash 4.9.4 and branchwater 0.9.15 both fail on a zip over 4 GiB ("Invalid Zip
    # archive: Extra data field contains disk number"); the scaled-10 proteome zip is
    # 5.2 GB (2026-11-09). Python's zipfile reads it, so pick sketches by the manifest.
    moltype = "DNA" if alphabet == "DNA" else alphabet
    out = {}
    with zipfile.ZipFile(sig_path(work, input_type)) as z:
        lines = z.read("SOURMASH-MANIFEST.csv").decode().splitlines()
        manifest = csv.DictReader(line for line in lines if not line.startswith("#"))
        sigs = []
        for row in manifest:
            if int(row["ksize"]) == k and row["moltype"] == moltype:
                data = z.read(row["internal_location"])
                if row["internal_location"].endswith(".gz"):
                    data = gzip.decompress(data)
                # The manifest row already picks the sketch; each file holds one.
                # (load_signatures' ksize= filter would compare 3k for protein.)
                (sig,) = sourmash.load_signatures(data)
                if sig.minhash.ksize != k or sig.minhash.moltype != moltype:
                    sys.exit(f"{row['internal_location']}: not {alphabet} k={k}")
                sigs.append(sig)
    for sig in sigs:
        if sig.name in out:
            sys.exit(f"{sig.name} {alphabet} k={k}: more than one sketch")
        out[sig.name] = sig
    want = set(species_files())
    if set(out) != want:
        sys.exit(f"{input_type} {alphabet} k={k}: missing {sorted(want - set(out))}")
    return out


CONTAINMENT_COLUMNS = [
    "anchor",
    "relative",
    "rank",
    "alphabet",
    "k",
    "bits_uniform",
    "input_type",
    "scaled",
    "containment_anchor_in_relative",
    "containment_relative_in_anchor",
    "n_hashes_anchor",
    "n_hashes_relative",
]


def compare(args):
    """Containment both ways from `sourmash scripts multisearch` (branchwater).

    One all-against-all multisearch per alphabet and k, every threshold off (-t 0 -A), so
    each ordered pair (query, match) has a row whose `containment` is the fraction of the
    query's hashes found in the match (checked on a toy set with branchwater 0.9.14,
    2026-11-08). Each value is cross-checked against the sourmash API, where
    a.contained_by(b) is |a & b| / |a|; any difference stops the run.
    """
    work = Path(args.work)
    path = read_tsv(OUT / "taxonomic_path.tsv")
    rows = []
    for input_type in INPUT_TYPES:
        for alphabet, k, _ in settings_for(input_type):
            csv_out = run_dir(work, "compare") / f"{input_type}.{alphabet}.k{k}.csv"
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            # branchwater 0.9.15 cannot read a sketch zip over 4 GiB ("could not load
            # sketches from path" on every entry; Python's zipfile reads the same file),
            # which the scaled-10 proteome zip passed on 2026-11-09. So multisearch reads
            # a zip holding only this alphabet and k, written from the full one.
            import sourmash

            by_species = load_sigs(work, input_type, alphabet, k)
            subset = csv_out.with_suffix(".sig.zip")
            subset.unlink(missing_ok=True)
            with sourmash.save_load.SaveSignatures_ZipFile(str(subset)) as save:
                for sig in by_species.values():
                    save.add(sig)
            sigs = str(subset)
            subprocess.run(
                [
                    "sourmash",
                    "scripts",
                    "multisearch",
                    sigs,
                    sigs,
                    "-k",
                    str(k),
                    "-s",
                    str(SCALED[alphabet]),
                    "-m",
                    alphabet,
                    "-t",
                    "0",
                    "-A",
                    "-c",
                    str(args.cores),
                    "-o",
                    str(csv_out),
                ],
                check=True,
            )
            ms = {
                (r["query_name"], r["match_name"]): float(r["containment"])
                for r in read_csv(csv_out)
            }
            subset.unlink()
            mhs = {name: sig.minhash for name, sig in by_species.items()}
            for anchor in PATHS:
                a_mh = mhs[anchor]
                if a_mh.scaled != SCALED[alphabet]:
                    sys.exit(f"{anchor} scaled {a_mh.scaled} != {SCALED[alphabet]}")
                for r in path:
                    if r["anchor"] != anchor or r["relative"] == anchor:
                        continue
                    rel = r["relative"]
                    r_mh = mhs[rel]
                    a_in_r, r_in_a = ms[(anchor, rel)], ms[(rel, anchor)]
                    for got, want in (
                        (a_in_r, a_mh.contained_by(r_mh)),
                        (r_in_a, r_mh.contained_by(a_mh)),
                    ):
                        if abs(got - want) > 1e-9:
                            sys.exit(
                                f"multisearch {got} != API {want}: {anchor} / {rel} "
                                f"{alphabet} k={k}"
                            )
                    rows.append(
                        dict(
                            anchor=anchor,
                            relative=rel,
                            rank=r["rank"],
                            alphabet=alphabet,
                            k=k,
                            bits_uniform=f"{bits_uniform(alphabet, k):.2f}",
                            input_type=input_type,
                            scaled=SCALED[alphabet],
                            containment_anchor_in_relative=f"{a_in_r:.6g}",
                            containment_relative_in_anchor=f"{r_in_a:.6g}",
                            n_hashes_anchor=len(a_mh),
                            n_hashes_relative=len(r_mh),
                        )
                    )
            print(input_type, alphabet, k, "done", flush=True)
    write_tsv(RES / "containment.tsv", rows, CONTAINMENT_COLUMNS)


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def max_c(r):
    return max(
        float(r["containment_anchor_in_relative"]),
        float(r["containment_relative_in_anchor"]),
    )


DEPTH = {}


def rank_key(r):
    """Closer relatives first: deeper shared taxon in the anchor's NCBI lineage."""
    if not DEPTH:
        for t in read_tsv(OUT / "taxonomic_path.tsv"):
            DEPTH[(t["anchor"], t["relative"])] = int(t["lineage_depth"])
    return -DEPTH[(r["anchor"], r["relative"])]


def reach_table():
    """The reach rule. Returns reach.tsv rows and {(anchor, input, alphabet, k): (floor, [c_yeast, c_ath])}."""
    rows, floors = [], {}
    cont = kept(read_tsv(RES / "containment.tsv"))
    for input_type in INPUT_TYPES:
        for alphabet, k, _ in settings_for(input_type):
            for anchor in PATHS:
                sub = [
                    r
                    for r in cont
                    if r["anchor"] == anchor
                    and r["input_type"] == input_type
                    and r["alphabet"] == alphabet
                    and int(r["k"]) == k
                ]
                fl = [
                    max_c(r)
                    for sp in floor_species(anchor)
                    for r in sub
                    if r["relative"] == sp
                ]
                assert len(fl) == 2, (anchor, input_type, alphabet, k, fl)
                floor = max(fl) + 3 * abs(fl[0] - fl[1])
                floors[(anchor, input_type, alphabet, k)] = (floor, fl)
                ranked = sorted(
                    (r for r in sub if r["relative"] not in floor_species(anchor)),
                    key=rank_key,
                )
                above = [i for i, r in enumerate(ranked) if max_c(r) > floor]
                base = dict(
                    alphabet=alphabet,
                    k=k,
                    bits_uniform=f"{bits_uniform(alphabet, k):.2f}",
                    anchor=anchor,
                    input_type=input_type,
                )
                if not above:
                    rows.append(
                        dict(
                            base,
                            reach_rank="none",
                            reach_relative="",
                            containment_at_reach="",
                            containment_next_rank=f"{max_c(ranked[0]):.6g}",
                            floor=f"{floor:.6g}",
                            ranks_below_floor_inside_reach="",
                        )
                    )
                    continue
                i = above[-1]
                nxt = max_c(ranked[i + 1]) if i + 1 < len(ranked) else max(fl)
                gaps = [ranked[j]["rank"] for j in range(i) if j not in above]
                rows.append(
                    dict(
                        base,
                        reach_rank=ranked[i]["rank"],
                        reach_relative=ranked[i]["relative"],
                        containment_at_reach=f"{max_c(ranked[i]):.6g}",
                        containment_next_rank=f"{nxt:.6g}",
                        floor=f"{floor:.6g}",
                        ranks_below_floor_inside_reach=",".join(gaps),
                    )
                )
    return rows, floors


MIN_BUSCO = 90  # olgabot, 2026-11-09: only proteomes with UniProt BUSCO >= 90% complete
DROPPED = []  # anchors left out by busco_filter, for the figure headers
LOW = set()  # relatives kept although under MIN_BUSCO, marked with a dagger


def busco_filter():
    """Use proteome_quality.tsv: drop an anchor whose proteome is under MIN_BUSCO % complete
    (with its whole panel), and flag, not drop, a relative under it (olgabot, 2026-11-09:
    "don't leave gaps ... just make a note of it"). Changes PATHS in place.
    """
    low = {
        q["species"]
        for q in read_tsv(OUT / "proteome_quality.tsv")
        if q["busco_complete_pct"] and float(q["busco_complete_pct"]) < MIN_BUSCO
    }
    DROPPED[:] = sorted(a for a in PATHS if a in low)
    for anchor in DROPPED:
        del PATHS[anchor]
    LOW.clear()
    LOW.update(e[1] for entries in PATHS.values() for e in entries if e[1] in low)


def kept(rows):
    """Rows whose anchor passes busco_filter."""
    return [r for r in rows if r["anchor"] in PATHS]


def dropped_line():
    out = ""
    if DROPPED:
        out += (
            f"Anchors left out, under {MIN_BUSCO}% complete by BUSCO: "
            + ", ".join(DROPPED)
            + ".\n"
        )
    if LOW:
        out += (
            f"\u2020 Proteome under {MIN_BUSCO}% complete by BUSCO, kept so no rank is empty: "
            + ", ".join(sorted(LOW))
            + ".\n"
        )
    return out


def reach(args):
    busco_filter()
    rows, _ = reach_table()
    write_tsv(RES / "reach.tsv", rows, list(rows[0]))
    for r in rows:
        print("\t".join(str(v) for v in r.values()))


COLOURS = {
    "DNA": "#0072B2",
    "protein": "#E69F00",
    "dayhoff": "#009E73",
    "hp": "#CC79A7",
}


def figure(args):
    sys.path.insert(0, str(HERE))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter, NullFormatter
    import pubfig as pf

    pf.use_style()
    busco_filter()
    cont = kept(read_tsv(RES / "containment.tsv"))
    path = kept(read_tsv(OUT / "taxonomic_path.tsv"))
    _, floors = reach_table()
    y_min = 1e-4
    for input_type in ("proteome",):
        fig, axs = pf.figure(pf.TWO_COLUMN_MM, 85, ncols=len(PATHS))
        for ax, anchor in zip(axs, PATHS):
            rels = sorted(
                (r for r in path if r["anchor"] == anchor and r["rank"] != "anchor"),
                key=lambda r: (rank_key(r), r["relative"]),
            )
            xs = {r["relative"]: i for i, r in enumerate(rels)}
            # Filled point: above that line's own noise floor; hollow: at or below it.
            panel_floors = []
            for alphabet, k, level in SETTINGS:
                it = input_type_of(alphabet)
                sub = sorted(
                    (
                        r
                        for r in cont
                        if r["anchor"] == anchor
                        and r["input_type"] == it
                        and r["alphabet"] == alphabet
                        and int(r["k"]) == k
                    ),
                    key=lambda r: xs[r["relative"]],
                )
                floor, _ = floors[(anchor, it, alphabet, k)]
                panel_floors.append(floor)
                x = [xs[r["relative"]] for r in sub]
                y = [max(max_c(r), y_min) for r in sub]
                colour = COLOURS[alphabet]
                ax.plot(x, y, color=colour, ls="--" if level == 30 else "-", zorder=2)
                above = [max_c(r) > floor for r in sub]
                ax.scatter(
                    [xx for xx, a in zip(x, above) if a],
                    [yy for yy, a in zip(y, above) if a],
                    s=9,
                    color=colour,
                    zorder=3,
                    lw=0,
                )
                ax.scatter(
                    [xx for xx, a in zip(x, above) if not a],
                    [yy for yy, a in zip(y, above) if not a],
                    s=9,
                    facecolor="white",
                    edgecolor=colour,
                    lw=0.6,
                    zorder=3,
                )
            ax.axhspan(
                max(min(panel_floors), y_min),
                max(max(panel_floors), y_min),
                color=pf.GREY,
                alpha=0.25,
                lw=0,
                zorder=1,
            )
            ax.set_yscale("log")
            ax.set_ylim(y_min, 1.5)
            ax.set_xticks(range(len(rels)))
            ax.set_xticklabels(
                [
                    f"{r['rank']}: {r['relative'][0]}. {r['relative'].split(' ', 1)[1]}"
                    for r in rels
                ],
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", labelsize=5)
            if ax is axs[0]:
                ax.set_ylabel("Max containment, both directions")
            ax.set_title(f"Anchor: {anchor}", loc="left")
        handles = [Patch(color=c, label=a) for a, c in COLOURS.items()]
        handles += [
            Line2D([], [], color="k", ls="--", label="30 bits per k-mer"),
            Line2D([], [], color="k", ls="-", label="42 bits per k-mer"),
            Line2D(
                [],
                [],
                color="k",
                marker="o",
                ls="",
                ms=3,
                label="above its line's noise floor",
            ),
            Line2D(
                [],
                [],
                color="k",
                marker="o",
                ls="",
                ms=3,
                mfc="white",
                label="at or below it",
            ),
            Patch(color=pf.GREY, alpha=0.5, label="range of the 8 lines' noise floors"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 1.0)
        )
        fig.suptitle(
            f"Protein, dayhoff and hp from the proteome (scaled {SCALED['protein']}); "
            f"DNA from the CDS of the same genes (scaled {SCALED['DNA']})\n"
            "Points on the bottom axis: containment below 0.0001, most of them 0",
            y=1.16,
        )
        fig.supxlabel(
            "Lowest rank shared with the anchor (NCBI Taxonomy): relative species"
        )
        for ax, letter in zip(axs, string.ascii_lowercase):
            pf.panel_label(ax, letter)
        pf.save(fig, str(RES / f"reach_by_alphabet_{input_type}"))
    figure_sweep(cont, path, floors)
    figure_pyramid(cont, path, floors)
    figure_reach_bars(cont, path)


PYRAMID = [("DNA", 21), ("protein", 10), ("dayhoff", 16), ("hp", 42)]


def percent(c):
    """Containment as a percentage, without the % sign (the colour bar says it)."""
    if c == 0:
        return "0"
    if c < 1e-4:
        return "<0.01"
    p = 100 * c
    return f"{p:.0f}" if p >= 10 else f"{p:.2g}" if p >= 1 else f"{p:.1g}"


MAIN_RANKS = ["species", "genus", "family", "order", "class", "phylum", "kingdom"]
SUB_TO_MAIN = {
    "subgenus": "genus",
    "subfamily": "family",
    "subclass": "class",
    "subphylum": "phylum",
}
PATHS_BACTERIUM = "Escherichia coli K-12"
# The settings drawn in the reach-by-rank figures, one figure each: (settings, how k was matched).
REACH_SETS = {
    "42bits": (
        PYRAMID,
        "DNA k=21 gives 42 bits per k-mer; protein k=10, dayhoff k=16 and hp k=42 are the "
        "whole k nearest to 42 bits",
    ),
    **{
        f"dna{3 * pk}_protein{pk}": (
            ladder(pk),
            f"DNA k={3 * pk} spans the same {pk} codons as protein k={pk}; "
            f"dayhoff k={bit_matched_k('dayhoff', pk)} and hp k={bit_matched_k('hp', pk)} "
            f"match protein k={pk} in bits (about {bits_uniform('protein', pk):.0f} per k-mer)",
        )
        for pk in LADDER_PROTEIN_K
    },
}


def column_of(t):
    """The main-rank column a relative is drawn in: the rank it was chosen to stand for
    (taxonomic_path.tsv requested_rank), else the main rank above the rank it shares."""
    if t["requested_rank"] in MAIN_RANKS:
        return t["requested_rank"]
    return SUB_TO_MAIN.get(t["rank"], t["rank"])


def shares_of(t):
    """What a relative shares with its anchor, in words: 'subgenus Mus', or 'clade
    Pteriomorphia, inside subclass Autobranchia' when the lowest shared group has no rank.
    """
    if t["lowest_shared_clade"] != t["shared_taxon"]:
        return (
            f"clade {t['lowest_shared_clade']}, inside {t['rank']} {t['shared_taxon']}"
        )
    return f"{t['rank']} {t['shared_taxon']}"


FOLD = (
    10  # a relative must share at least 10 times more than the other-kingdom background
)
MIN_SHARED = 100  # and at least this many shared sampled k-mers (hashes), so a ratio of tiny counts cannot pass


def reach_fold_table(cont, path):
    """Reach by fold over background: per anchor and alphabet at 42 and 30 bits, each relative's
    containment divided by the higher of the two other-kingdom species' containments.

    Writes RES/reach_fold.tsv. Reach is the farthest rank reached without a failing rank
    nearer the anchor.
    """
    rows = []
    for anchor in PATHS:
        rels = sorted(
            (
                r
                for r in path
                if r["anchor"] == anchor
                and r["rank"] not in ("anchor", "other kingdom")
            ),
            key=lambda r: (rank_key(r), r["relative"]),
        )
        for set_name, alphabet, k in (
            (n, a, k) for n, (settings, _) in REACH_SETS.items() for a, k in settings
        ):
            sel = {
                c["relative"]: c
                for c in cont
                if c["anchor"] == anchor
                and c["alphabet"] == alphabet
                and int(c["k"]) == k
            }
            background = max(max_c(sel[sp]) for sp in floor_species(anchor))
            still_reaching = True
            for r in rels:
                c = sel[r["relative"]]
                shared = round(
                    float(c["containment_anchor_in_relative"])
                    * int(c["n_hashes_anchor"])
                )
                fold = max_c(c) / background if background else math.inf
                passed = fold >= FOLD and shared >= MIN_SHARED
                still_reaching = still_reaching and passed
                rows.append(
                    {
                        "anchor": anchor,
                        "setting_set": set_name,
                        "alphabet": alphabet,
                        "k": k,
                        "relative": r["relative"],
                        "rank": r["rank"],
                        "column": column_of(r),
                        "shares": shares_of(r),
                        "containment": f"{max_c(c):.6g}",
                        "background": f"{background:.6g}",
                        "fold_over_background": f"{fold:.4g}",
                        "shared_hashes": shared,
                        "passes": passed,
                        "reached": still_reaching,
                    }
                )
    write_tsv(RES / "reach_fold.tsv", rows, list(rows[0]))
    return rows


def figure_reach_bars(cont, path):
    """One reach-by-rank figure per set of settings in REACH_SETS."""
    rows = reach_fold_table(cont, path)
    for set_name, (settings, matched) in REACH_SETS.items():
        draw_reach_bars(
            [r for r in rows if r["setting_set"] == set_name],
            settings,
            matched,
            "reach_by_rank_uniprot" + ("" if set_name == "42bits" else f"_{set_name}"),
        )
    draw_ladder_summary(rows)


def draw_ladder_summary(rows):
    """Farthest main rank reached against codons per k-mer: DNA k = 3 x codons, protein
    k = codons, dayhoff and hp k matched to protein in bits. One panel per anchor."""
    import pubfig as pf

    series = [
        ("DNA", "DNA, k = 3 \u00d7 codons", pf.OKABE_ITO["black"], "s"),
        ("protein", "protein, k = codons", pf.OKABE_ITO["blue"], "o"),
        (
            "dayhoff",
            "dayhoff, k matched to protein in bits",
            pf.OKABE_ITO["orange"],
            "^",
        ),
        ("hp", "hp, k matched to protein in bits", pf.OKABE_ITO["vermillion"], "D"),
    ]
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 62, ncols=len(PATHS), sharey=True)
    level = {}  # (anchor, alphabet, codons) -> farthest main rank reached, 0 = none
    for ax, anchor in zip(axs, PATHS):
        for i, (alphabet, label, colour, marker) in enumerate(series):
            xs, ys = [], []
            for pk in LADDER_PROTEIN_K:
                k = dict(ladder(pk))[alphabet]
                reached = [
                    r["column"]
                    for r in rows
                    if r["setting_set"] == f"dna{3 * pk}_protein{pk}"
                    and r["anchor"] == anchor
                    and r["alphabet"] == alphabet
                    and r["k"] == k
                    and r["reached"]
                ]
                xs.append(pk)
                level[(anchor, alphabet, pk)] = (
                    MAIN_RANKS.index(reached[-1]) + 1 if reached else 0
                )
                # Small vertical offsets keep four series on the same rank all visible.
                ys.append(
                    (MAIN_RANKS.index(reached[-1]) + 1 if reached else 0)
                    + (i - 1.5) * 0.09
                )
            ax.plot(xs, ys, color=colour, marker=marker, ms=3, lw=1, label=label)
        ax.set_title(f"anchor: {anchor}", fontsize=7, weight="bold")
        ax.set_xticks(LADDER_PROTEIN_K)
        ax.set_xlabel("Codons per k-mer")
        ax.set_yticks(range(len(MAIN_RANKS) + 1))
        ax.set_yticklabels(["no rank"] + MAIN_RANKS)
        ax.set_ylim(-0.5, len(MAIN_RANKS) + 0.5)
        for y in range(len(MAIN_RANKS) + 1):
            ax.axhline(y, color="#e6e6e6", lw=0.3, zorder=0)
    axs[0].set_ylabel("Farthest rank reached")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.suptitle(
        f"Reached: at least {FOLD}\u00d7 more k-mers shared than with another kingdom, at least "
        f"{MIN_SHARED} sampled k-mers, and every nearer rank reached too.\n"
        f"1 in {SCALED['hp']} k-mers sampled"
        + ("" if SCALED["hp"] == SCALED["DNA"] else f" (DNA: 1 in {SCALED['DNA']})")
        + ".\n"
        + dropped_line()
        + "Ranks are main ranks; a relative sharing a sub-rank counts under the rank above it. "
        "Points on one rank are nudged apart so all four show.\n"
        + ladder_conclusion(level),
        y=1.13,
        va="bottom",
    )
    pf.save(fig, str(RES / "reach_by_codons_uniprot"))


def ladder_conclusion(level):
    """Count, over every anchor and k-mer length, where each method reaches compared with protein."""
    n = len(PATHS) * len(LADDER_PROTEIN_K)
    parts = []
    for alphabet in ("hp", "dayhoff", "DNA"):
        d = [
            level[(a, alphabet, pk)] - level[(a, "protein", pk)]
            for a in PATHS
            for pk in LADDER_PROTEIN_K
        ]
        parts.append(
            f"{alphabet} farther in {sum(x > 0 for x in d)}, same in {sum(x == 0 for x in d)}, "
            f"nearer in {sum(x < 0 for x in d)}"
        )
    return (
        f"Conclusion, against protein over {n} anchor-length pairs:\n"
        + "; ".join(parts)
        + "."
    )


def reach_conclusion(rows, settings):
    """One line saying where dayhoff or hp reach a different main rank than protein."""

    def far(anchor, alphabet):
        ranks = [
            r["column"]
            for r in rows
            if r["anchor"] == anchor and r["alphabet"] == alphabet and r["reached"]
        ]
        return MAIN_RANKS.index(ranks[-1]) if ranks else -1

    names = {a: a.split()[0][0] + ". " + a.split()[1] for a in PATHS}
    further, nearer = {}, {}
    for anchor in PATHS:
        for alphabet in ("dayhoff", "hp"):
            d = far(anchor, alphabet) - far(anchor, "protein")
            if d:
                (further if d > 0 else nearer).setdefault(alphabet, []).append(
                    names[anchor]
                )

    def listing(found):
        return "; ".join(f"{al} for {', '.join(an)}" for al, an in found.items())

    line = "Conclusion: compared with protein, " + (
        f"a reduced alphabet reaches a farther rank only as {listing(further)}."
        if further
        else "no reduced alphabet reaches a farther rank for any anchor."
    )
    if nearer:
        line += f"\nIt stops at a nearer rank as {listing(nearer)}."
    return line + " Every other row stops at the same rank as protein."


def cell_label(fold, shared):
    """Fold over background; the shared count when the background is zero."""
    if math.isinf(fold):
        return f"{shared} shared"
    if fold < 1:
        return "<1\u00d7"
    if 9 <= fold < 11:  # near the 10x line, so a rounded "10x" never hides a miss
        return f"{fold:.1f}\u00d7"
    return f"{fold:.0f}\u00d7" if fold < 1e4 else "\u226510000\u00d7"


def draw_reach_bars(rows, settings, matched, name):
    """Which ranks each alphabet reaches: one panel per anchor, one row per alphabet,
    one column per main rank. Filled cell: passes the fold rule.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    import pubfig as pf

    reached_colour, passed_colour = pf.OKABE_ITO["blue"], pf.OKABE_ITO["sky_blue"]
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 42 * len(PATHS), nrows=len(PATHS))
    # One column per main rank, the same in every panel. A relative that shares a sub-rank
    # (subgenus, subfamily, ...) goes under the main rank above it; where two relatives land
    # in one column, the farther one is drawn, since reaching it means reaching the rank.
    columns = MAIN_RANKS + ["other kingdom"]
    dropped, shared_sub = [], []
    for ax_i, (ax, anchor) in enumerate(zip(axs, PATHS)):
        sub = [r for r in rows if r["anchor"] == anchor]
        rels = list(dict.fromkeys(r["relative"] for r in sub))  # nearest first
        rank_of = {r["relative"]: r["rank"] for r in sub}
        col_of = {r["relative"]: r["column"] for r in sub}
        shares = {r["relative"]: r["shares"] for r in sub}
        in_col = {}
        for rel in rels:
            main = col_of[rel]
            if main in in_col:
                gone = in_col[main]
                dropped.append(f"{gone} ({shares[gone]})")
                shared_sub[:] = [x for x in shared_sub if not x.startswith(gone + " (")]
            if not shares[rel].startswith(main + " "):
                shared_sub.append(f"{rel} ({shares[rel]})")
            in_col[main] = rel  # later = farther
        for i, (alphabet, k) in enumerate(settings):
            y = len(settings) - 1 - i
            line = {r["relative"]: r for r in sub if r["alphabet"] == alphabet}
            for j, col in enumerate(MAIN_RANKS):
                if col not in in_col:
                    ax.plot(j + 0.5, y + 0.5, ".", color=pf.GREY, ms=3)
                    continue
                r = line[in_col[col]]
                fc = (
                    reached_colour
                    if r["reached"]
                    else passed_colour if r["passes"] else "white"
                )
                ax.add_patch(
                    Rectangle(
                        (j + 0.04, y + 0.1), 0.92, 0.8, fc=fc, ec="#999999", lw=0.5
                    )
                )
                fold = float(r["fold_over_background"])
                ax.text(
                    j + 0.5,
                    y + 0.5,
                    cell_label(fold, r["shared_hashes"]),
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if r["reached"] else "k",
                )
            # The other-kingdom background, 1x by definition.
            j = len(MAIN_RANKS)
            ax.add_patch(
                Rectangle(
                    (j + 0.04, y + 0.1), 0.92, 0.8, fc="#e6e6e6", ec="#999999", lw=0.5
                )
            )
            ax.text(j + 0.5, y + 0.5, "1\u00d7", ha="center", va="center", fontsize=5.5)
            last = [
                col
                for col in MAIN_RANKS
                if col in in_col and line[in_col[col]]["reached"]
            ]
            ax.text(
                len(columns) + 0.15,
                y + 0.5,
                f"reaches {last[-1]}" if last else "reaches no rank here",
                va="center",
                fontsize=6,
                weight="bold",
            )
        ax.set_yticks([len(settings) - 1 - i + 0.5 for i in range(len(settings))])
        ax.set_yticklabels(
            [f"{a} k={k} ({bits_uniform(a, k):.0f} bits)" for a, k in settings]
        )
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([j + 0.5 for j in range(len(columns))])
        ax.set_xticklabels(
            [
                (
                    (in_col[col] + (" \u2020" if in_col[col] in LOW else "")).replace(
                        " ", "\n", 1
                    )
                    if col in in_col
                    else ""
                )
                for col in MAIN_RANKS
            ]
            + [
                (
                    "yeast or\nArabidopsis"
                    if anchor != PATHS_BACTERIUM
                    else "Bacillus or\nThermotoga"
                )
            ],
            fontsize=5.5,
            style="italic",
        )
        ax.tick_params(axis="x", length=0)
        if ax_i == 0:  # rank names once, above the first panel, shared by all panels
            for j, col in enumerate(columns):
                ax.text(
                    j + 0.5,
                    len(settings) + 0.15,
                    "another kingdom\n(background)" if col == "other kingdom" else col,
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    weight="bold",
                )
        ax.set_xlim(0, len(columns) + 2.2)
        ax.set_ylim(0, len(settings))
        for side in ("left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.set_title(
            f"anchor: {anchor}",
            loc="left",
            fontsize=7,
            weight="bold",
            pad=22 if ax_i == 0 else 6,  # clear the rank names over the first panel
        )
    fig.legend(
        handles=[
            Patch(
                fc=reached_colour,
                ec="#999999",
                lw=0.5,
                label="reached: passes, and so does every rank nearer the anchor",
            ),
        ]
        # Listed only when some cell has it: a key entry with no mark is noise.
        + [
            Patch(
                fc=passed_colour,
                ec="#999999",
                lw=0.5,
                label="passes, but a nearer relative (drawn, or listed as not drawn) does not",
            )
        ]
        * any(r["passes"] and not r["reached"] for r in rows)
        + [
            Patch(fc="white", ec="#999999", lw=0.5, label="does not pass"),
            Patch(
                fc="#e6e6e6",
                ec="#999999",
                lw=0.5,
                label="another kingdom: the background, 1\u00d7 by definition",
            ),
            Line2D(
                [],
                [],
                marker=".",
                ls="",
                color=pf.GREY,
                ms=3,
                label="no relative at this rank in the species set",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    sampled = (
        f"1 in {SCALED['hp']}"
        if SCALED["hp"] == SCALED["DNA"]
        else f"1 in {SCALED['hp']} (DNA: 1 in {SCALED['DNA']})"
    )
    header = (
        f"A relative is reached when it shares at least {FOLD}\u00d7 more k-mers with the anchor than an organism "
        f"from another kingdom does, and at least {MIN_SHARED} sampled k-mers.\n"
        f"Number in each cell: how many times more; {sampled} k-mers sampled.\n"
        + (
            "Where the other-kingdom species share no sampled k-mer, the cell gives "
            "how many the relative shares instead.\n"
            if any(math.isinf(float(r["fold_over_background"])) for r in rows)
            else ""
        )
        + f"{matched}.\n"
        + dropped_line()
        + "Protein, dayhoff and hp from UniProt reference proteomes; DNA from the coding sequences of the same genes.\n"
        "dayhoff letters: AGPST C DENQ FWY HKR ILMV; hp letters: AFGILMPVWY (hydrophobic), CDEHKNQRST (polar).\n"
        "Each column is one main rank, the same in every panel; each relative sits under the rank it was chosen for.\n"
        + (
            "Where what a relative shares with its anchor is not its column's rank: "
            + ", ".join(shared_sub)
            + ".\n"
            if shared_sub
            else ""
        )
        + (
            "Not drawn, because a farther relative stands for the same rank: "
            + ", ".join(dropped)
            + ".\n"
            if dropped
            else ""
        )
        # Built from the data, so it cannot go stale.
        + reach_conclusion(rows, settings)
    )
    # A line naming many relatives is wider than the page; wrap every line to fit 183 mm.
    import textwrap

    header = "\n".join(textwrap.fill(line, 150) for line in header.split("\n"))
    # Anchor the header's bottom just above the legend, however many lines it has.
    fig.suptitle(
        header, y=1 + 10 / (42 * len(PATHS)), va="bottom"
    )  # 10 mm above the panels
    pf.save(fig, str(RES / name))


def figure_pyramid(cont, path, floors):
    """The taxonomic-rank pyramid of each anchor, with the containment of every relative.

    One row per relative, farthest at the top, the anchor at the tip. Cells: the larger of
    the two containments at 42 bits per k-mer. Black outline: above that setting's floor.
    """
    import matplotlib as mpl
    from matplotlib.patches import Patch, Polygon, Rectangle
    import pubfig as pf

    cmap = mpl.colormaps["mako_r"]
    norm = mpl.colors.LogNorm(1e-4, 1)
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 105, ncols=len(PATHS))
    n_max = max(sum(r["anchor"] == a for r in path) for a in PATHS)
    for ax, anchor in zip(axs, PATHS):
        rels = sorted(
            (r for r in path if r["anchor"] == anchor and r["rank"] != "anchor"),
            key=lambda r: (rank_key(r), r["relative"]),
        )[
            ::-1
        ]  # farthest first, drawn at the top
        groups = (
            []
        )  # (rank, shared taxon, [relatives]); the two different-kingdom species share a band
        for r in rels:
            if groups and groups[-1][0] == r["rank"]:
                groups[-1][2].append(r)
            else:
                groups.append((r["rank"], r["shared_taxon"], [r]))
        n_rows = len(rels) + 1
        top = n_max  # rows stacked down from the same top line in every panel
        width = (
            lambda y: 0.9 * (y - (top - n_rows)) / n_rows
        )  # pyramid half-width at height y
        y = top
        for rank, taxon, members in groups:
            y0 = y - len(members)
            band = [
                (0.9 - width(y), y - 0.06),
                (0.9 + width(y), y - 0.06),
                (0.9 + width(y0 + 0.06), y0 + 0.06),
                (0.9 - width(y0 + 0.06), y0 + 0.06),
            ]
            ax.add_patch(Polygon(band, closed=True, fc="#e6e6e6", ec="k", lw=0.5))
            label = "domain only" if rank == "other kingdom" else rank
            ax.text(
                2.0,
                y - 0.3,
                f"{label}: {taxon}",
                fontsize=5.5,
                weight="bold",
                va="center",
            )
            for i, r in enumerate(members):
                yy = y - i - 0.5
                ax.text(
                    2.0,
                    yy - (0.22 if i == 0 else 0),
                    r["relative"],
                    fontsize=5.5,
                    style="italic",
                    va="center",
                )
                for j, (alphabet, k) in enumerate(PYRAMID):
                    row = next(
                        c
                        for c in cont
                        if c["anchor"] == anchor
                        and c["relative"] == r["relative"]
                        and c["alphabet"] == alphabet
                        and int(c["k"]) == k
                    )
                    c = max_c(row)
                    floor, _ = floors[(anchor, input_type_of(alphabet), alphabet, k)]
                    above = rank != "other kingdom" and c > floor
                    x = 9.2 + 1.5 * j
                    fc = cmap(norm(max(c, 1e-4))) if c > 0 else "white"
                    ax.add_patch(
                        Rectangle(
                            (x, yy - 0.42),
                            1.3,
                            0.84,
                            fc=fc,
                            ec="k" if above else "none",
                            lw=1.2 if above else 0,
                        )
                    )
                    dark = c > 0 and norm(max(c, 1e-4)) > 0.45
                    ax.text(
                        x + 0.65,
                        yy,
                        percent(c),
                        fontsize=5,
                        ha="center",
                        va="center",
                        color="white" if dark else "k",
                    )
            y = y0
        # The anchor at the tip.
        tip = [
            (0.9 - width(y), y - 0.06),
            (0.9 + width(y), y - 0.06),
            (0.9, y - 1 + 0.06),
        ]
        ax.add_patch(Polygon(tip, closed=True, fc="#b3b3b3", ec="k", lw=0.5))
        ax.text(
            2.0, y - 0.5, f"anchor: {anchor}", fontsize=5.5, weight="bold", va="center"
        )
        for j, (alphabet, k) in enumerate(PYRAMID):
            ax.text(
                9.2 + 1.5 * j + 0.65,
                top + 0.25,
                f"{alphabet}\nk={k}",
                fontsize=5,
                ha="center",
                va="bottom",
            )
        ax.set_xlim(0, 15.2)
        ax.set_ylim(top - n_max - 0.1, top + 1.2)
        ax.axis("off")
    cax = fig.add_axes([0.30, 1.03, 0.25, 0.018])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm, cmap), cax=cax, orientation="horizontal"
    )
    cb.set_ticks([1e-4, 1e-3, 1e-2, 0.1, 1])
    cb.set_ticklabels(["0.01%", "0.1%", "1%", "10%", "100%"])
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.set_label(
        "Share of k-mers in common, % (larger of the two directions)", fontsize=6
    )
    fig.legend(
        handles=[
            Patch(
                fc="white",
                ec="k",
                lw=1.2,
                label="above the noise floor set by the two\nspecies that share only the domain\n(the rule fixed before the run)",
            ),
        ],
        loc="center left",
        bbox_to_anchor=(0.6, 1.04),
        frameon=False,
    )
    fig.suptitle(
        "Each anchor's relatives by the lowest taxonomic rank they share with it (NCBI Taxonomy). "
        "42 bits per k-mer; scaled "
        + (
            str(SCALED["hp"])
            if SCALED["hp"] == SCALED["DNA"]
            else f"{SCALED['hp']} (DNA: {SCALED['DNA']})"
        ),
        y=1.12,
    )
    pf.save(fig, str(RES / "reach_pyramid"))


def figure_sweep(cont, path, floors):
    """Dayhoff and hp at every k swept: one row per alphabet, one column per anchor.

    Colour runs light to dark with k (seaborn crest); filled point = above that line's own
    noise floor, hollow = at or below it, as in the main figure.
    """
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, NullFormatter
    import pubfig as pf

    y_min = 1e-4
    alphabets = ("dayhoff", "hp")
    fig, axs = pf.figure(
        pf.TWO_COLUMN_MM, 150, nrows=len(alphabets), ncols=len(PATHS), sharey=True
    )
    for row, alphabet in zip(axs, alphabets):
        ks = sorted(k for a, k, _ in SETTINGS + SWEEP if a == alphabet)
        cmap = mpl.colormaps["crest"]
        colour = {k: cmap(0.15 + 0.85 * i / (len(ks) - 1)) for i, k in enumerate(ks)}
        for ax, anchor in zip(row, PATHS):
            rels = sorted(
                (r for r in path if r["anchor"] == anchor and r["rank"] != "anchor"),
                key=lambda r: (rank_key(r), r["relative"]),
            )
            xs = {r["relative"]: i for i, r in enumerate(rels)}
            for k in ks:
                sub = sorted(
                    (
                        r
                        for r in cont
                        if r["anchor"] == anchor
                        and r["alphabet"] == alphabet
                        and int(r["k"]) == k
                    ),
                    key=lambda r: xs[r["relative"]],
                )
                floor, _ = floors[(anchor, "proteome", alphabet, k)]
                x = [xs[r["relative"]] for r in sub]
                y = [max(max_c(r), y_min) for r in sub]
                above = [max_c(r) > floor for r in sub]
                ax.plot(x, y, color=colour[k], zorder=2)
                ax.scatter(
                    [xx for xx, a in zip(x, above) if a],
                    [yy for yy, a in zip(y, above) if a],
                    s=9,
                    color=colour[k],
                    lw=0,
                    zorder=3,
                )
                ax.scatter(
                    [xx for xx, a in zip(x, above) if not a],
                    [yy for yy, a in zip(y, above) if not a],
                    s=9,
                    facecolor="white",
                    edgecolor=colour[k],
                    lw=0.6,
                    zorder=3,
                )
            ax.set_yscale("log")
            ax.set_ylim(y_min, 1.5)
            ax.set_xticks(range(len(rels)))
            if alphabet == alphabets[-1]:
                ax.set_xticklabels(
                    [
                        f"{r['rank']}: {r['relative'][0]}. {r['relative'].split(' ', 1)[1]}"
                        for r in rels
                    ],
                    rotation=30,
                    ha="right",
                    rotation_mode="anchor",
                )
                ax.tick_params(axis="x", labelsize=5)
            else:
                ax.set_xticklabels([])
                ax.set_title(f"Anchor: {anchor}", loc="left")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if ax is row[0]:
                ax.set_ylabel(f"{alphabet}: max containment")
        row[-1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    color=colour[k],
                    marker="o",
                    ms=3,
                    label=f"k={k} ({bits_uniform(alphabet, k):.1f} bits)",
                )
                for k in ks
            ],
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            title=alphabet,
        )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="k",
                marker="o",
                ls="",
                ms=3,
                label="above its line's noise floor",
            ),
            Line2D(
                [],
                [],
                color="k",
                marker="o",
                ls="",
                ms=3,
                mfc="white",
                label="at or below it",
            ),
        ],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"Dayhoff and hp at each k, from the proteome (scaled {SCALED['hp']})\n"
        "Points on the bottom axis: containment below 0.0001, most of them 0",
        y=1.08,
    )
    fig.supxlabel(
        "Lowest rank shared with the anchor (NCBI Taxonomy): relative species"
    )
    for ax, letter in zip(axs.flat, "abcdef"):
        pf.panel_label(ax, letter)
    pf.save(fig, str(RES / "reach_by_k_dayhoff_hp"))


CORES = 4


def submit(args):
    """Submit fetch+entropy, then sketch, then compare, chained with afterok.

    Sizes (2026-11-08): the inputs are 13 proteomes (2-12 M residues each) and 13 CDS sets
    (9-60 M bases, UniProt _DNA files). At scaled 200 a 12 M-residue proteome keeps about
    60 000 hashes per setting, so every sketch together is well under 1 GB in memory.
    The time limits are about 10x what that size needs; fetch also builds the pinned
    branchwater environment the first time. NB364_SCALED reaches the jobs through
    --export=ALL. At scaled 10 the largest sketch (57 M CDS bases) holds about 5.7 M
    hashes per setting, 20x the default run's; the scaled-200/1000 run peaked at 0.6 GB,
    so 32 GB leaves room for a 20x increase and more.
    """
    sb = str(HERE / "364_alphabet_reach.sbatch")

    def sbatch(*extra):
        out = subprocess.run(
            ["sbatch", "--parsable", *extra, sb],
            check=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        return out.stdout.strip().split(";")[0]

    after = [f"--dependency=afterok:{args.after}"] if args.after else []
    j1 = sbatch(
        "--job-name=364-fetch",
        *after,
        "--export=ALL,STEP=fetch",
        "--mem=8G",
        "--time=3:00:00",
    )
    mem = "32G" if min(SCALED.values()) < 100 else "16G"
    common = [f"--cpus-per-task={CORES}", f"--mem={mem}", "--time=2:00:00"]
    j2 = sbatch(
        "--job-name=364-sketch",
        f"--dependency=afterok:{j1}",
        "--export=ALL,STEP=sketch",
        *common,
    )
    j3 = sbatch(
        "--job-name=364-compare",
        f"--dependency=afterok:{j2}",
        "--export=ALL,STEP=compare",
        *common,
    )
    print(f"fetch {j1}, sketch {j2}, compare {j3}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd")  # no required=: Sherlock python3 is 3.6
    for name in ("resolve", "reach", "figure", "quality"):
        sub.add_parser(name)
    sub.add_parser("submit").add_argument(
        "--after", help="job id the fetch job waits for (afterok)"
    )
    for name in ("fetch", "entropy"):
        sub.add_parser(name).add_argument("--work", required=True)
    for name in ("sketch", "compare"):
        s = sub.add_parser(name)
        s.add_argument("--work", required=True)
        s.add_argument("--cores", type=int, default=CORES)
    args = p.parse_args()
    if args.cmd is None:
        p.error("a subcommand is required")
    globals()[args.cmd](args)


if __name__ == "__main__":
    main()
