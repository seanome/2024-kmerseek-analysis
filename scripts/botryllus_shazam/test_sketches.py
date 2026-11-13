"""Compare sourmash API k sizes and six-frame translation against its actual CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import sourmash


class SketchTests(unittest.TestCase):
    def test_cli_api_translation(self):
        dna = "ATGGCTGATGAATTTGGTCATATTAAACTGATGAATCCTCAACGTTCTACTGTTTGGTAT" * 4
        for moltype, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
            with self.subTest(moltype=moltype), tempfile.TemporaryDirectory() as d:
                inp = Path(d) / "dna.fa"
                out = Path(d) / "out.sig"
                inp.write_text(">example\n" + dna + "\n")
                command = [
                    sys.executable,
                    "-m",
                    "sourmash",
                    "sketch",
                    "translate",
                    str(inp),
                    "-p",
                    f"{moltype},k={k},scaled=1",
                    "-o",
                    str(out),
                ]
                subprocess.run(
                    command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                cli = list(sourmash.load_file_as_signatures(str(out)))[0].minhash
                api = sourmash.MinHash(
                    n=0,
                    ksize=k,
                    scaled=1,
                    **{
                        dict(protein="is_protein", dayhoff="dayhoff", hp="hp")[
                            moltype
                        ]: True
                    },
                )
                api.add_sequence(dna, force=True)
                self.assertEqual(dict(cli.hashes), dict(api.hashes))
                self.assertEqual(api.ksize, k)
                self.assertGreater(len(api), 0)
                # On-disk sourmash protein sketches encode the nucleotide-equivalent k.
                stored = json.loads(out.read_text())[0]["signatures"][0]["ksize"]
                self.assertEqual(stored, 3 * k)


if __name__ == "__main__":
    unittest.main()
