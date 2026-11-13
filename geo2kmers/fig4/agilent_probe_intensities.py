"""Write one intensity per probe and sample from Agilent feature extraction files.

Usage: agilent_probe_intensities.py <folder of raw files> <platform table> <out.tsv.gz>
Takes gProcessedSignal, background already removed, and scales each sample to the same 75th percentile,
as Agilent suggests. The platform table links each feature number to its probe sequence.
"""

import gzip
import re
import sys
from pathlib import Path

import numpy as np

from geo2kmers.geo import read_table
from geo2kmers.sequences import find_column

folder, platform, out = sys.argv[1:]
_, columns, rows = read_table(platform, "platform")
sequence_column = find_column(columns, ["SEQUENCE"])
sequence = {r[columns[0]]: r[sequence_column].strip() for r in rows if r.get(sequence_column, "").strip()}


def read_signal(path):
    """Return {feature number: gProcessedSignal} from one raw file."""
    signal = {}
    number = value = None
    with gzip.open(path, "rt") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "FEATURES":
                number, value = fields.index("FeatureNum"), fields.index("gProcessedSignal")
            elif fields[0] == "DATA" and value is not None:
                signal[fields[number]] = float(fields[value])
    return signal


paths = sorted(p for p in Path(folder).glob("GSM*.txt.gz"))
samples = [re.match(r"GSM\d+", p.name).group() for p in paths]
features = sorted(sequence, key=int)
matrix = np.array([[s.get(f, np.nan) for f in features] for s in map(read_signal, paths)]).T
matrix = matrix / np.nanpercentile(matrix, 75, axis=0) * np.nanmean(np.nanpercentile(matrix, 75, axis=0))
with gzip.open(out, "wt") as f:
    f.write("sequence\t" + "\t".join(samples) + "\n")
    for feature, row in zip(features, matrix, strict=True):
        if not np.isnan(row).any():
            f.write(sequence[feature] + "\t" + "\t".join(f"{v:.2f}" for v in row) + "\n")
