"""Restore only line endings when Git normalized a frozen sample table."""

import hashlib
import json
from pathlib import Path


def restore_frozen_samples(samples, design, output):
    data = Path(samples).read_bytes()
    expected = json.loads(Path(design).read_text())["samples_sha256"]
    lf = data.replace(b"\r\n", b"\n")
    for candidate in (data, lf, lf.replace(b"\n", b"\r\n")):
        if hashlib.sha256(candidate).hexdigest() == expected:
            with Path(output).open("xb") as stream:
                stream.write(candidate)
            return
    raise ValueError("Frozen sample metadata changed beyond line endings")
