"""Pinned Branchwater tests separate X hashing from stop-only translation."""

import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import sourmash

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peptide_protocol import export, PROTOCOL
from sketch_corrected_peptides import build, SETTINGS
from artifacts import write_new, sha


def verify(root, executable):
    root.mkdir()
    rng = random.Random(20261025)
    codons = [
        "GCT",
        "TGT",
        "GAT",
        "GAA",
        "TTT",
        "GGT",
        "CAT",
        "ATT",
        "AAA",
        "CTG",
        "ATG",
        "AAT",
        "CCT",
        "CAA",
        "CGT",
        "TCT",
        "ACT",
        "GTT",
        "TGG",
        "TAT",
    ]
    long_dna = (
        "".join(rng.choices(codons, k=500))
        + "NNN"
        + "".join(rng.choices(codons, k=500))
    )
    source = root / "fixture.fa"
    source.write_text(
        ">short\nACGT\n>positive\n"
        + long_dna
        + "TAA"
        + "GCT" * 30
        + "\n>same_hash_other_id\n"
        + long_dna
        + "TAA"
        + "GCT" * 30
        + "\n"
    )
    export(source, root / "translation")
    build(root / "translation", root / "sketches", executable, 4)
    expected_ids = {"short", "positive", "same_hash_other_id"}
    for mol, _, _ in SETTINGS:
        sigs = {
            s.name: s.minhash
            for s in sourmash.load_file_as_signatures(
                str(root / "sketches" / f"{mol}-10.sig.gz")
            )
        }
        assert set(sigs) == expected_ids
        assert not len(sigs["short"]) and len(sigs["positive"]) > 0
        assert sigs["positive"] == sigs["same_hash_other_id"]
    peptide = (
        "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=60))
        + "X"
        + "".join(rng.choices("ACDEFGHIKLMNPQRSTVWY", k=60))
    )
    (root / "x.fa").write_text(">internalX\n" + peptide + "\n")
    with (root / "x.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "genome_filename", "protein_filename"])
        writer.writerow(["x", "", str(root / "x.fa")])
    cmd = [
        str(executable),
        "scripts",
        "manysketch",
        str(root / "x.csv"),
        "--singleton",
        "-c",
        "4",
        "-o",
        str(root / "x.zip"),
    ]
    for mol, k, _ in SETTINGS:
        cmd += ["-p", f"{mol},k={k},scaled=1"]
    with (root / "x.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    probes = []
    for sig in sourmash.load_file_as_signatures(str(root / "x.zip")):
        mol = sig.minhash.moltype.lower()
        k, flags = next((k, f) for m, k, f in SETTINGS if m == mol)
        api = sourmash.MinHash(n=0, ksize=k, scaled=1, **flags)
        api.add_protein(peptide)
        assert api == sig.minhash
        all_words = set()
        x_words = set()
        for i in range(len(peptide) - k + 1):
            word = peptide[i : i + k]
            mh = sourmash.MinHash(n=0, ksize=k, scaled=1, **flags)
            mh.add_protein(word)
            all_words.update(mh.hashes)
            if "X" in word:
                x_words.update(mh.hashes)
        assert all_words == set(sig.minhash.hashes)
        probes.append(
            dict(
                moltype=mol,
                k=k,
                scaled=1,
                peptide=peptide,
                total_hashes=len(api),
                x_spanning_distinct_hashes=len(x_words),
                x_spanning_hashes_retained=len(x_words & set(api.hashes)),
            )
        )
    assert len(probes) == 3
    write_new(
        root / "protocol-audit.json",
        dict(
            protocol=PROTOCOL,
            source_sha256=sha(source),
            original_ids=sorted(expected_ids),
            duplicate_hash_ids_preserved=True,
            nonempty_controls=True,
            x_hash_probes=probes,
        ),
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
