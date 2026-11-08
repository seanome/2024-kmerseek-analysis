"""Notebook 364: how far up the tree of life each sequence alphabet still finds shared k-mers.

For two anchor species (Mus musculus, Botryllus schlosseri) and a relative at each
taxonomic rank, sketch every proteome and genome with sourmash and measure containment
(the fraction of one species' sampled k-mers that also occur in the other).

k-sizes are matched on bits per k-mer, k * log2(alphabet size), at 30 and 42 bits:
    30 bits: DNA k=15, protein k=7,  dayhoff k=12, hp k=30
    42 bits: DNA k=21, protein k=10, dayhoff k=16, hp k=42
Protein-space alphabets are sketched from the proteome and from the six-frame
translated genome; DNA from the genome.

Subcommands, in run order:

    resolve   (laptop, small API calls)  -> taxonomic_path.tsv
    fetch     (Sherlock job)             -> downloads, fills download_date
    entropy   (Sherlock job)             -> bits_per_kmer.tsv (measured letter entropy)
    sketch    (Sherlock array job)       -> one .sig.zip per species and input type
    compare   (Sherlock job)             -> containment.tsv
    reach     (laptop)                   -> reach.tsv
    figure    (laptop)                   -> reach_by_alphabet_<input>.png / .svg / .pdf

Reach rule, fixed before the run (2026-11-08, from the task text):
    floor(alphabet, k, input type, anchor) = max(c_yeast, c_arabidopsis)
                                             + 3 * |c_yeast - c_arabidopsis|
    where c is the max of the two containment directions.
    reach = the most distant rank (excluding the two different-kingdom species) whose
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
import shutil
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
SCALED = {"DNA": 1000, "protein": 200, "dayhoff": 200, "hp": 200}
INPUT_TYPES = ("proteome", "translated_genome", "genome")


def bits_uniform(alphabet, k):
    return k * math.log2(ALPHABET_SIZE[alphabet])


def settings_for(input_type):
    return [s for s in SETTINGS if (s[0] == "DNA") == (input_type == "genome")]


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

FLOOR_SPECIES = ("Saccharomyces cerevisiae", "Arabidopsis thaliana")

# requested_species, used_species, taxid, requested_rank, UniProt proteome (None: NCBI
# protein set), NCBI assembly, substitution note.
# The assembly is the one UniProt built the proteome from (UniProt genomeAssembly field),
# so genome and proteome describe the same genome build.
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
    "Botryllus schlosseri": [
        (
            "Botryllus schlosseri",
            "Botryllus schlosseri",
            30301,
            "anchor",
            None,
            "GCF_051294905.1",
            "no UniProt proteome for taxid 30301 (2026-11-08); proteins are the NCBI RefSeq "
            "annotation GCF_051294905.1-RS_2026_02 of the same assembly, all isoforms",
        ),
        (
            "Ciona intestinalis",
            "Ciona intestinalis",
            7719,
            "subphylum",
            "UP000008144",
            "GCA_000224145.2",
            "UniProt built UP000008144 from GCA_000224145.1, which has no FASTA on the NCBI FTP site "
            "(2026-11-08); GCA_000224145.2 is the same KH assembly with chromosome MT changed",
        ),
        (
            "Branchiostoma floridae",
            "Branchiostoma floridae",
            7739,
            "phylum",
            "UP000001554",
            "GCF_000003815.2",
            "",
        ),
        (
            "Strongylocentrotus purpuratus",
            "Strongylocentrotus purpuratus",
            7668,
            "superphylum",
            "UP000007110",
            "GCA_000002235.4",
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
}

# Ranks from nearest to most distant, for ordering the shared ranks NCBI reports.
RANK_ORDER = [
    "species",
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
    "genome_url",
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
    domain Eukaryota, called "other kingdom". Depth is the position of the lowest shared
    taxon in the anchor's full lineage, so a larger depth means a closer relative even when
    two relatives share the same rank (sea urchin and fly both share only kingdom Metazoa
    with Botryllus, but sea urchin also shares the unranked clade Deuterostomia).
    """
    rel = set(rel_lin)
    depth = max(i for i, t in enumerate(anchor_lin) if t in rel)
    clade = anchor_lin[depth][1]
    for rank, name in reversed(anchor_lin):
        if rank in RANK_ORDER and rank not in ("species",) and (rank, name) in rel:
            if rank in ("superkingdom", "domain"):
                break
            return rank, name, clade, depth
    return "other kingdom", "Eukaryota", clade, depth


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
                purl = (
                    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/"
                    f"reference_proteomes/Eukaryota/{up}/{up}_{taxid}.fasta.gz"
                )
            else:
                release = ""
                purl = genome_url(acc).replace("_genomic.fna.gz", "_protein.faa.gz")
            gurl = genome_url(acc)
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
                    genome_url=gurl,
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
    return species.replace(" ", "_")


def species_files():
    """{species: (proteome_url, genome_url)} with each species once."""
    return {
        r["relative"]: (r["proteome_url"], r["genome_url"])
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
    """Download every proteome and genome once; record the UTC date of each download."""
    (Path(args.work) / "raw").mkdir(parents=True, exist_ok=True)
    dates = {}
    for sp, urls in species_files().items():
        for seq, url in zip(("proteome", "genome"), urls):
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
    path = OUT / "taxonomic_path.tsv"
    rows = read_tsv(path)
    for r in rows:
        r["download_date"] = dates[r["relative"]]
    write_tsv(path, rows, COLUMNS)


def letter_counts(path, genome):
    """Letter counts of a gzipped FASTA: A/C/G/T for a genome, the 20 amino acids otherwise."""
    keep = b"ACGT" if genome else b"ACDEFGHIKLMNPQRSTVWY"
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
    """Measured Shannon entropy per letter of each anchor's proteome (per alphabet) and genome."""
    rows = []
    for anchor in PATHS:
        aa = letter_counts(raw_path(args.work, anchor, "proteome"), genome=False)
        nt = letter_counts(raw_path(args.work, anchor, "genome"), genome=True)
        by_alphabet = {"DNA": nt, "protein": aa}
        for name, table in (("dayhoff", DAYHOFF), ("hp", HP)):
            mapped = collections.Counter()
            for letter, c in aa.items():
                mapped[table[letter]] += c
            by_alphabet[name] = mapped
        for alphabet, k, level in SETTINGS:
            h, n = shannon(by_alphabet[alphabet])
            rows.append(
                dict(
                    anchor=anchor,
                    alphabet=alphabet,
                    k=k,
                    bits_level=level,
                    source="genome" if alphabet == "DNA" else "proteome",
                    letters_counted=n,
                    entropy_per_letter=f"{h:.4f}",
                    bits_uniform=f"{bits_uniform(alphabet, k):.2f}",
                    bits_measured=f"{k * h:.2f}",
                )
            )
        print(anchor, "entropy done", flush=True)
    write_tsv(OUT / "bits_per_kmer.tsv", rows, list(rows[0]))


def sig_path(work, sp, input_type):
    return Path(work) / "sigs" / f"{slug(sp)}.{input_type}.sig.zip"


def sketch_tasks():
    return [(sp, it) for sp in sorted(species_files()) for it in INPUT_TYPES]


def sketch(args):
    """One (species, input type) per array task; run `sketch-tasks` to see the numbering."""
    sp, input_type = sketch_tasks()[args.index]
    out = sig_path(args.work, sp, input_type)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(out, "exists", flush=True)
        return
    sub = {"proteome": "protein", "translated_genome": "translate", "genome": "dna"}[
        input_type
    ]
    seq = "proteome" if input_type == "proteome" else "genome"
    cmd = ["sourmash", "sketch", sub]
    for alphabet, k, _ in settings_for(input_type):
        cmd += [
            "-p",
            (
                f"k={k},scaled={SCALED[alphabet]}"
                if alphabet == "DNA"
                else f"{alphabet},k={k},scaled={SCALED[alphabet]}"
            ),
        ]
    part = out.with_name(out.name.replace(".sig.zip", ".part.sig.zip"))
    cmd += [str(raw_path(args.work, sp, seq)), "--name", sp, "-o", str(part)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    os.rename(part, out)


def load_mh(work, sp, input_type, alphabet, k):
    import sourmash

    moltype = "DNA" if alphabet == "DNA" else alphabet
    sigs = list(
        sourmash.load_file_as_signatures(
            str(sig_path(work, sp, input_type)), ksize=k, select_moltype=moltype
        )
    )
    if len(sigs) != 1:
        sys.exit(f"{sp} {input_type} {alphabet} k={k}: {len(sigs)} sketches")
    return sigs[0].minhash


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
    """Containment both ways from the sourmash API, cross-checked against `sourmash compare`.

    In the `sourmash compare --containment` CSV, row i column j is the fraction of j's
    hashes found in i (checked on a toy pair with sourmash 4.9.4, 2026-11-08). The API's
    a.contained_by(b) is |a & b| / |a|.
    """
    work = Path(args.work)
    path = read_tsv(OUT / "taxonomic_path.tsv")
    rows = []
    for input_type in INPUT_TYPES:
        for alphabet, k, _ in settings_for(input_type):
            for anchor in PATHS:
                members = [r for r in path if r["anchor"] == anchor]
                names = [r["relative"] for r in members]
                csv_out = (
                    work
                    / "compare"
                    / f"{slug(anchor)}.{input_type}.{alphabet}.k{k}.csv"
                )
                csv_out.parent.mkdir(parents=True, exist_ok=True)
                flag = "--dna" if alphabet == "DNA" else f"--{alphabet}"
                subprocess.run(
                    [
                        "sourmash",
                        "-q",
                        "compare",
                        *[str(sig_path(work, n, input_type)) for n in names],
                        "--containment",
                        flag,
                        "-k",
                        str(k),
                        "--csv",
                        str(csv_out),
                    ],
                    check=True,
                )
                with open(csv_out) as f:
                    rd = csv.reader(f)
                    header = next(rd)
                    matrix = {
                        h: dict(zip(header, map(float, row)))
                        for h, row in zip(header, rd)
                    }
                a_mh = load_mh(work, anchor, input_type, alphabet, k)
                if a_mh.scaled != SCALED[alphabet]:
                    sys.exit(f"{anchor} scaled {a_mh.scaled} != {SCALED[alphabet]}")
                for r in members[1:]:
                    r_mh = load_mh(work, r["relative"], input_type, alphabet, k)
                    a_in_r, r_in_a = a_mh.contained_by(r_mh), r_mh.contained_by(a_mh)
                    for got, want in (
                        (matrix[r["relative"]][anchor], a_in_r),
                        (matrix[anchor][r["relative"]], r_in_a),
                    ):
                        if abs(got - want) > 1e-9:
                            sys.exit(
                                f"compare CSV {got} != API {want}: {anchor} / {r['relative']} "
                                f"{input_type} {alphabet} k={k}"
                            )
                    rows.append(
                        dict(
                            anchor=anchor,
                            relative=r["relative"],
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
                print(input_type, alphabet, k, anchor, "done", flush=True)
    write_tsv(OUT / "containment.tsv", rows, CONTAINMENT_COLUMNS)


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
    cont = read_tsv(OUT / "containment.tsv")
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
                    max_c(r) for sp in FLOOR_SPECIES for r in sub if r["relative"] == sp
                ]
                assert len(fl) == 2, (anchor, input_type, alphabet, k, fl)
                floor = max(fl) + 3 * abs(fl[0] - fl[1])
                floors[(anchor, input_type, alphabet, k)] = (floor, fl)
                ranked = sorted(
                    (r for r in sub if r["relative"] not in FLOOR_SPECIES), key=rank_key
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


def reach(args):
    rows, _ = reach_table()
    write_tsv(OUT / "reach.tsv", rows, list(rows[0]))
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
    cont = read_tsv(OUT / "containment.tsv")
    path = read_tsv(OUT / "taxonomic_path.tsv")
    _, floors = reach_table()
    y_min = 1e-4
    for input_type in ("proteome", "translated_genome"):
        fig, axs = pf.figure(pf.TWO_COLUMN_MM, 85, ncols=2)
        for ax, anchor in zip(axs, PATHS):
            rels = sorted(
                (r for r in path if r["anchor"] == anchor and r["rank"] != "anchor"),
                key=lambda r: (rank_key(r), r["relative"]),
            )
            xs = {r["relative"]: i for i, r in enumerate(rels)}
            # DNA from the genome is drawn on both figures, so protein-space lines have a DNA reference.
            # Filled point: above that line's own noise floor; hollow: at or below it.
            panel_floors = []
            for alphabet, k, level in SETTINGS:
                it = "genome" if alphabet == "DNA" else input_type
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
                    f"{r['rank']}: {r['relative'][0]}. {r['relative'].split()[1]}"
                    for r in rels
                ],
                rotation=30,
                ha="right",
                rotation_mode="anchor",
            )
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.tick_params(axis="x", labelsize=5)
            ax.set_xlabel(
                "Lowest rank shared with the anchor (NCBI Taxonomy), relative species"
            )
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
        label = input_type.replace("_", " ")
        fig.suptitle(
            f"Protein, dayhoff and hp from {label} (scaled 200); DNA from genome (scaled 1000)\n"
            "Points on the bottom axis: containment below 0.0001, most of them 0",
            y=1.16,
        )
        for ax, letter in zip(axs, "ab"):
            pf.panel_label(ax, letter)
        pf.save(fig, str(OUT / f"reach_by_alphabet_{input_type}"))


IMAGE = "docker://quay.io/biocontainers/sourmash:4.9.4--hdfd78af_0"


def submit(args):
    """Submit fetch+entropy, then the sketch array, then compare, chained with afterok."""
    sb = str(HERE / "364_alphabet_reach.sbatch")
    n = len(sketch_tasks())

    def sbatch(*extra):
        out = subprocess.run(
            ["sbatch", "--parsable", *extra, sb],
            check=True,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
        return out.stdout.strip().split(";")[0]

    j1 = sbatch(
        "--job-name=364-fetch", "--export=ALL,STEP=fetch", "--mem=8G", "--time=4:00:00"
    )
    j2 = sbatch(
        "--job-name=364-sketch",
        f"--dependency=afterok:{j1}",
        f"--array=0-{n - 1}",
        "--export=ALL,STEP=sketch",
        "--mem=32G",
        "--time=6:00:00",
    )
    j3 = sbatch(
        "--job-name=364-compare",
        f"--dependency=afterok:{j2}",
        "--export=ALL,STEP=compare",
        "--mem=64G",
        "--time=4:00:00",
    )
    print(f"fetch {j1}, sketch array {j2} ({n} tasks), compare {j3}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd")  # no required=: Sherlock python3 is 3.6
    for name in ("resolve", "reach", "figure", "sketch-tasks", "submit"):
        sub.add_parser(name)
    for name in ("fetch", "entropy", "compare"):
        sub.add_parser(name).add_argument("--work", required=True)
    s = sub.add_parser("sketch")
    s.add_argument("--work", required=True)
    s.add_argument("--index", type=int, required=True)
    args = p.parse_args()
    if args.cmd is None:
        p.error("a subcommand is required")
    if args.cmd == "sketch-tasks":
        for i, t in enumerate(sketch_tasks()):
            print(i, *t)
        return
    globals()[args.cmd](args)


if __name__ == "__main__":
    main()
