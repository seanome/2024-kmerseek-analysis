#!/usr/bin/env python3
"""Count the residues of a protein FASTA: sequences, residues, and each letter.

Usage: db_composition.py <fasta[.gz]> <name> <out.json>

Two readers run side by side, one per CPU: `grep -c` counts the headers, and the other drops
them and counts letters with numpy in 256 MB chunks, so a 90 GB UniRef90 fits in a few GB of memory. `n_residues` counts every letter of
the sequence lines; `n_canonical` only the 20 canonical residues, which is what the class
frequencies are computed from.
"""

import json
import shutil
import subprocess
import sys
from datetime import date

import numpy as np

CHUNK = 1 << 28
CANONICAL = "ACDEFGHIKLMNPQRSTVWY"


def count(path: str) -> dict:
    """Letter counts and the number of sequences in one FASTA file."""
    unzip = "pigz -dc" if shutil.which("pigz") else "gzip -dc"
    reader = f"{unzip} '{path}'" if path.endswith(".gz") else f"cat '{path}'"
    headers = subprocess.Popen(
        f"set -o pipefail; {reader} | grep -c '^>'",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        text=True,
    )
    proc = subprocess.Popen(
        f"set -o pipefail; {reader} | grep -v '^>' | tr -d '\\n\\r'",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
    )
    counts = np.zeros(256, dtype=np.int64)
    while chunk := proc.stdout.read(CHUNK):
        counts += np.bincount(np.frombuffer(chunk, dtype=np.uint8), minlength=256)
    if proc.wait() != 0:
        raise RuntimeError(f"reading {path} failed with exit {proc.returncode}")
    n_sequences = int(headers.communicate()[0])
    if headers.returncode != 0:
        raise RuntimeError(
            f"counting headers of {path} failed with exit {headers.returncode}"
        )
    letters = {chr(i): int(c) for i, c in enumerate(counts) if c}
    return {
        "n_sequences": n_sequences,
        "n_residues": int(counts.sum()),
        "n_canonical": sum(letters.get(aa, 0) for aa in CANONICAL),
        "letters": letters,
    }


def main() -> None:
    path, name, out = sys.argv[1:4]
    result = {"database": name, "path": path, "date_counted": date.today().isoformat()}
    result.update(count(path))
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(
        f"{name}: {result['n_sequences']:_} sequences, {result['n_residues']:_} residues"
    )


if __name__ == "__main__":
    main()
