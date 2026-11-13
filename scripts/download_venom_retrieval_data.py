#!/usr/bin/env python3
"""Download the raw inputs for the venom-peptide retrieval benchmark.

Three sources, each written to its own folder under the data directory so that nothing
here overwrites the cys-frameworks download (`download_cys_framework_data.py`, which
fetches a different, smaller field set):

  conoserver_raw/   ConoServer's protein dump, XML and FASTA. ConoServer is the only
                    source here that states a cysteine framework label and a
                    pharmacological family for a cone-snail peptide.
  uniprot/          Two Swiss-Prot searches, with the feature columns the manifest needs:
                    disulfide bonds, modified residues, mature-peptide boundaries, and the
                    function text whose ECO codes say whether a claim was measured.
  release/          What each source said its version was on the day it was downloaded, so
                    a later rebuild can tell whether the inputs moved.

The two Swiss-Prot searches are the positives and the controls:

  toxins.tsv              reviewed, Toxin keyword (KW-0800), 10 to 300 residues. The
                          upper bound keeps precursors of short peptides and drops the
                          large toxins (latrotoxins and the like), which no part of this
                          benchmark is about.
  metazoa_nontoxin.tsv    reviewed, NOT the Toxin keyword, 10 to 300 residues, Metazoa.
                          This is where the hard controls come from: defensins, protease
                          inhibitors, EGF-like modules and other cysteine-rich animal
                          peptides that share a framework with a toxin without being one.

Both go through scripts/fetch_uniprot.py, which pages the REST API and fails if the row
count does not match the total the server states. UniProt's /stream endpoint has returned
short results with HTTP 200 on this exact query shape before, so a download that finishes
is not evidence that it is complete.

Usage: download_venom_retrieval_data.py <data-dir>
"""

import gzip
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

CONOSERVER = "https://www.conoserver.org/download"
UNIPROT_REST = "https://rest.uniprot.org/uniprotkb"

#: Every column the manifest reads. One list, used for both searches, so the two tables
#: have identical columns and can be concatenated without aligning anything by hand.
#:
#: Why each one is here:
#:   sequence, length                the precursor as Swiss-Prot holds it
#:   ft_peptide, ft_chain            where a mature peptide is annotated, with its evidence
#:   ft_signal, ft_propep            the parts cut off during processing
#:   ft_disulfid                     the measured disulfide pairs: the connectivity label
#:   ft_mod_res, ft_lipid,
#:   ft_carbohyd, ft_crosslnk        modifications a sequence-only method cannot represent
#:   cc_ptm                          the free-text processing note, for terminal amidation
#:   keyword, keywordid              the activity and target labels
#:   cc_function                     the measured activity, with ECO evidence codes
#:   xref_pdb                        whether a structure of this entry was solved
#:   protein_families, cc_similarity the family statement, used for grouping the splits
#:   lineage                         phylum and class, for the leakage checks
FIELDS = ",".join(
    [
        "accession",
        "id",
        "protein_name",
        "organism_name",
        "organism_id",
        "lineage",
        "length",
        "sequence",
        "keyword",
        "keywordid",
        "ft_peptide",
        "ft_chain",
        "ft_signal",
        "ft_propep",
        "ft_disulfid",
        "ft_mod_res",
        "ft_lipid",
        "ft_carbohyd",
        "ft_crosslnk",
        "cc_ptm",
        "cc_function",
        "cc_similarity",
        "xref_pdb",
        "protein_families",
    ]
)

#: The length window both searches use. 10 is the shortest thing worth calling a peptide;
#: 300 holds the precursor of a 60-residue peptide with room to spare.
LENGTH = "length:[10 TO 300]"

SEARCHES = {
    "toxins.tsv": f"(reviewed:true) AND (keyword:KW-0800) AND ({LENGTH})",
    # taxonomy_id:33208 is Metazoa.
    "metazoa_nontoxin.tsv": (
        f"(reviewed:true) NOT (keyword:KW-0800) AND ({LENGTH}) AND (taxonomy_id:33208)"
    ),
}


def curl(url: str, dest: Path) -> None:
    """Download `url` to `dest`, giving up rather than hanging on a stalled transfer.

    A plain curl of ConoServer has sat at 0 bytes/s without returning; --speed-limit with
    --speed-time makes it fail instead, which a caller can see.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    partial = dest.with_suffix(dest.suffix + ".partial")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--retry",
            "3",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.rename(dest)
    print(f"  {dest.name}: {dest.stat().st_size / 1e6:.1f} MB")


def main(data_dir: Path) -> None:
    cono = data_dir / "conoserver_raw"
    uniprot = data_dir / "uniprot"
    release = data_dir / "release"
    for folder in (cono, uniprot, release):
        folder.mkdir(parents=True, exist_ok=True)

    print("ConoServer:")
    # ConoServer publishes no release number, so the download date is the provenance.
    # The dump is served gzipped only; the plain .xml path is a 404.
    for name in ("conoserver_protein.xml.gz", "conoserver_protein.fa.gz"):
        curl(f"{CONOSERVER}/{name}", cono / name)
    xml = cono / "protein.xml"
    if not xml.exists():
        with gzip.open(cono / "conoserver_protein.xml.gz", "rb") as src, open(
            xml, "wb"
        ) as dst:
            shutil.copyfileobj(src, dst)
        print(f"  protein.xml: {xml.stat().st_size / 1e6:.1f} MB")

    print("Swiss-Prot release:")
    reldate = urllib.request.urlopen(
        f"{UNIPROT_REST}/search?query=accession:P01513&format=tsv&fields=accession&size=1",
        timeout=60,
    )
    stated = reldate.headers.get("X-UniProt-Release", "unknown")
    (release / "uniprot_release.txt").write_text(stated + "\n")
    print(f"  {stated}")

    print("Swiss-Prot searches:")
    fetch = Path(__file__).with_name("fetch_uniprot.py")
    for name, query in SEARCHES.items():
        dest = uniprot / name
        if dest.exists():
            print(f"  have {name}")
            continue
        print(f"  {name}")
        subprocess.run(
            [sys.executable, str(fetch), query, FIELDS, str(dest)], check=True
        )
        (release / f"{name}.query.txt").write_text(query + "\n" + FIELDS + "\n")

    print(f"\ndone. Next: python -m venom_retrieval manifest --data {data_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser())
