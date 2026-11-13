"""Run the production counting script on known counts around counter boundaries."""

import gzip
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import unittest


class CountTests(unittest.TestCase):
    def test_native_counter_boundaries(self):
        workflow = (
            Path(__file__).resolve().parents[2] / "nextflow-runs/botryllus-shazam/main.nf"
        )
        process = (
            workflow.read_text()
            .split("process COUNT_KMERS {")[1]
            .split("process JOIN_KMERS")[0]
        )
        script = re.search(r'"""(.*?)"""', process, flags=re.S).group(1)
        dekupl = Path(os.environ["GROUP_HOME"]) / "opt/botryllus-dekupl-1.3.3-20261024"
        rng = random.Random(992)
        expected = {}
        records = []
        for i, repeats in enumerate((1, 2, 63, 64, 65, 67, 127, 128, 129, 1024)):
            seq = "".join(rng.choice("ACGT") for _ in range(31))
            reverse = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            key = min(seq, reverse)
            assert key not in expected
            expected[key] = 2 * repeats
            records.extend(f"@{i}_{j}\n{seq}\n+\n{'I' * 31}\n" for j in range(repeats))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mate in (1, 2):
                with gzip.open(root / f"reads{mate}.fq.gz", "wt") as f:
                    f.write("".join(records))
            for token, value in {
                "${params.dekupl}": str(dekupl),
                "${task.cpus}": "1",
                "${reads[0]}": "reads1.fq.gz",
                "${reads[1]}": "reads2.fq.gz",
                "${run}": "fixture",
            }.items():
                script = script.replace(token, value)
            subprocess.run(
                ["bash", "-euo", "pipefail", "-c", script], cwd=root, check=True
            )
            with gzip.open(root / "fixture.counts.txt.gz", "rt") as f:
                actual = {key: int(n) for key, n in (line.split() for line in f)}
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
