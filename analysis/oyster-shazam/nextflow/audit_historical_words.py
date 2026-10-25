"""Rebuild historical preflight hashes from individual words in original DNA.

This deliberately checks the old stop/X-splitting rule, not the corrected protocol.
"""

import json
import os
from pathlib import Path
import sys
import time

from Bio.Seq import Seq
import sourmash
from audit_completed import dna_records, load, require
from artifacts import write_new, sha


def word_sketch(dna, mol, k, flags):
    words = 0
    mh = sourmash.MinHash(n=0, ksize=k, scaled=10, **flags)
    if mol == "dna":
        mh.add_sequence(dna, force=True)
    else:
        for strand in (Seq(dna), Seq(dna).reverse_complement()):
            for offset in range(3):
                end = offset + max(0, (len(strand) - offset) // 3) * 3
                peptide = str(strand[offset:end].translate())
                # Enumerating each word independently avoids reusing the historical splitter.
                for i in range(len(peptide) - k + 1):
                    word = peptide[i : i + k]
                    if "*" not in word and "X" not in word:
                        mh.add_protein(word)
                        words += 1
    return mh, words


def audit(output):
    group = Path(os.environ["GROUP_SCRATCH"])
    replay = group / "oyster-first-public-linear-20261025"
    records = dna_records(replay, load(replay / "manifest.json"))
    root = group / "oyster-stopfree-public-pilot-linear-20261025"
    summaries = []
    for mol, k, flags in [
        ("dna", 31, {}),
        ("protein", 7, {"is_protein": True}),
        ("dayhoff", 12, {"dayhoff": True}),
        ("hp", 30, {"hp": True}),
    ]:
        start = time.monotonic()
        coarse = {}
        for sig in sourmash.load_file_as_signatures(
            str(root / f"{mol}.query100.sig.gz")
        ):
            require(sig.name not in coarse, "Duplicate coarse ID")
            require(
                sig.minhash.scaled == 100
                and sig.minhash.ksize == k
                and sig.minhash.moltype.lower() == mol
                and sig.minhash.seed == 42
                and not sig.minhash.track_abundance,
                "Wrong coarse sketch settings",
            )
            coarse[sig.name] = sig.md5sum()
        require(set(coarse) == set(records), "Coarse membership mismatch")
        count = words = 0
        for sig in sourmash.load_file_as_signatures(
            str(root / f"{mol}.query10.sig.gz")
        ):
            require(sig.name in coarse, "Missing or duplicate fine ID")
            dna = records[sig.name]
            mh, counted_words = word_sketch(dna, mol, k, flags)
            words += counted_words
            require(
                mh == sig.minhash, f"Historical fine hash mismatch: {mol} {sig.name}"
            )
            require(
                sourmash.SourmashSignature(mh.downsample(scaled=100)).md5sum()
                == coarse.pop(sig.name),
                "Historical coarse hash mismatch",
            )
            count += 1
        require(
            not coarse and count == len(records),
            "Historical original membership incomplete",
        )
        summaries.append(
            dict(
                moltype=mol,
                queries=count,
                independent_words=words,
                seconds=time.monotonic() - start,
                query10_sha256=sha(root / f"{mol}.query10.sig.gz"),
                query100_sha256=sha(root / f"{mol}.query100.sig.gz"),
            )
        )
        print("WORD AUDIT", summaries[-1], flush=True)
    write_new(
        output,
        dict(
            protocol="HISTORICAL stop/X-splitting, minimum stretch equal to k",
            validation="All original query hashes reconstructed; translated hashes built one non-stop non-X word at a time",
            settings=summaries,
        ),
    )


if __name__ == "__main__":
    audit(Path(sys.argv[1]))
