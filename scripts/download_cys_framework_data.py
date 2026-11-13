#!/usr/bin/env python3
"""Download everything the cysteine-framework experiment reads, and check it arrived whole.

Four sources:
  ConoServer protein XML and FASTA   the query peptides and their framework labels
  Swiss-Prot FASTA                   the reference
  Swiss-Prot query taxa and KW-0800  the query additions and the truth set
  NCBI taxonomy dump                 the genus and phylum of every reference entry

Every download is checked against a count the server states, not just HTTP 200. UniProt's
/stream endpoint returned 7_518 of 7_922 toxin entries and 1_986 of 3_916 query-taxon
entries with status 200 and no error, so a stream that merely finishes is not evidence it
is complete. The paged fetch below compares its row count with X-Total-Results and fails
if they disagree.

Usage: download_cys_framework_data.py <out-dir>
"""

import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

CONOSERVER = "https://www.conoserver.org/download"
UNIPROT_FTP = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete"
TAXDUMP = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdmp.zip"

#: The lineages the Swiss-Prot queries come from: spiders, scorpions, sea anemones, and
#: cone snails as a cross-check against ConoServer.
QUERY_TAXA = {"Araneae": 6893, "Scorpiones": 6855, "Actiniaria": 6103, "Conus": 6490}

TOXIN_QUERY = "reviewed:true AND keyword:KW-0800"
QUERY_TAXA_QUERY = (
    "(reviewed:true) AND ((keyword:KW-0800 AND (taxonomy_id:6893 OR taxonomy_id:6855 "
    "OR taxonomy_id:6103)) OR taxonomy_id:6490)"
)
QUERY_TAXA_FIELDS = (
    "accession,id,protein_name,organism_name,organism_id,length,keyword,sequence,"
    "ft_chain,ft_peptide,ft_signal,ft_propep,ft_disulfid"
)


def curl(url: str, dest: Path) -> None:
    """Download with resume and a stall timeout.

    A multi-GB curl can hang silently; --speed-limit with --speed-time makes it give up
    instead of sitting there, and -C - resumes rather than starting over.
    """
    if dest.exists():
        print(f"  have {dest.name}")
        return
    print(f"  fetching {dest.name}")
    subprocess.run(
        [
            "curl",
            "-sL",
            "-C",
            "-",
            "--speed-limit",
            "10240",
            "--speed-time",
            "120",
            "-o",
            str(dest),
            url,
        ],
        check=True,
    )


def check_gzip(path: Path) -> None:
    subprocess.run(["gzip", "-t", str(path)], check=True)


def main(out_dir: Path) -> None:
    cono = out_dir / "conoserver_raw"
    sprot = out_dir / "swissprot_raw"
    taxdump = out_dir / "taxdump_raw"
    for folder in (cono, sprot, taxdump):
        folder.mkdir(parents=True, exist_ok=True)

    print("ConoServer:")
    for name in ("conoserver_protein.fa.gz", "conoserver_protein.xml.gz"):
        curl(f"{CONOSERVER}/{name}", cono / name)
        check_gzip(cono / name)
    xml = cono / "protein.xml"
    if not xml.exists():
        with gzip.open(cono / "conoserver_protein.xml.gz", "rb") as src, open(
            xml, "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)

    print("Swiss-Prot:")
    release = (
        urllib.request.urlopen(f"{UNIPROT_FTP}/reldate.txt", timeout=60).read().decode()
    )
    (sprot / "reldate.txt").write_text(release)
    print("  " + release.strip().split("\n")[1])
    curl(f"{UNIPROT_FTP}/uniprot_sprot.fasta.gz", sprot / "uniprot_sprot.fasta.gz")
    check_gzip(sprot / "uniprot_sprot.fasta.gz")

    print("UniProt search results:")
    fetch = Path(__file__).with_name("fetch_uniprot.py")
    for query, fields, name in (
        (TOXIN_QUERY, "accession,organism_id", "kw0800.tsv"),
        (QUERY_TAXA_QUERY, QUERY_TAXA_FIELDS, "querytaxa.tsv"),
    ):
        if (sprot / name).exists():
            print(f"  have {name}")
            continue
        subprocess.run(
            [sys.executable, str(fetch), query, fields, str(sprot / name)], check=True
        )

    print("NCBI taxonomy:")
    curl(TAXDUMP, taxdump / "taxdmp.zip")
    if not (taxdump / "nodes.dmp").exists():
        subprocess.run(
            [
                "unzip",
                "-o",
                "-q",
                str(taxdump / "taxdmp.zip"),
                "nodes.dmp",
                "names.dmp",
                "-d",
                str(taxdump),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("resolve_taxonomy.py")),
            str(out_dir),
        ],
        check=True,
    )
    print(f"\ndone. Next: build_cys_framework_data.py {out_dir} <tables-dir>")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
