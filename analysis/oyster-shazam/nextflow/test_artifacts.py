import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import sourmash
from sourmash.save_load import SaveSignaturesToLocation
from artifacts import SETTINGS, PREPROCESSING, adopt, audit, metadata, sha


class AdoptionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.make("query", {"hit": [1, 2], "miss": [3], "empty": []})
        self.make("reference", {"RUN": [1, 2, 4]})
        self.matches = self.root / "matches.csv"
        self.matches.write_text("query_name,match_name,intersect_hashes\nhit,RUN,2\n")
        self.log = self.root / "search.log"
        self.log.write_text("DONE. Processed 3 search sigs\n")

    def make(self, role, sketches):
        path = self.root / f"{role}.sig.gz"
        with SaveSignaturesToLocation(str(path)) as save:
            for name, hashes in sketches.items():
                mh = sourmash.MinHash(n=0, ksize=31, scaled=100)
                mh.add_many(hashes)
                save.add(sourmash.SourmashSignature(mh, name=name))
        manifest = dict(
            role=role,
            settings=SETTINGS,
            preprocessing=PREPROCESSING,
            ownership="completed-read-only",
            artifact_id=role,
            artifact_path=str(path),
            artifact_sha256=sha(path),
            source_sha256="a" * 64,
            source_code_sha256="b" * 64,
            source_tool_versions={"fixture": "1"},
            expected_ids=list(sketches),
        )
        (self.root / f"{role}.json").write_text(json.dumps(manifest))
        adopt(
            self.root / f"{role}.json", path, role, self.root / f"{role}-receipt.json"
        )

    def check(self):
        audit(
            self.root / "query.sig.gz",
            self.root / "reference.sig.gz",
            self.root / "query-receipt.json",
            self.root / "reference-receipt.json",
            self.matches,
            self.log,
            self.root / "validated.json",
        )

    def test_exact_matches_and_all_original_ids(self):
        self.check()
        result = json.loads((self.root / "validated.json").read_text())
        self.assertEqual(result["match_pairs"], 1)
        self.assertEqual(result["unmatched_ids"], ["empty", "miss"])
        self.assertEqual(result["empty_ids"], ["empty"])
        with self.assertRaises(FileExistsError):
            self.check()

    def test_wrong_protocol_ownership_and_missing_provenance(self):
        original = json.loads((self.root / "query.json").read_text())
        for key, value in (
            ("settings", {**SETTINGS, "stop_policy": "split-X"}),
            ("ownership", "running-worker"),
            ("source_sha256", ""),
            ("expected_ids", ["q", "q"]),
            ("preprocessing", {**PREPROCESSING, "quality": 10}),
        ):
            changed = copy.deepcopy(original)
            changed[key] = value
            with self.assertRaises(ValueError):
                metadata(changed, "query")

    def test_missing_build_provenance_requires_explicit_test_scope(self):
        manifest = json.loads((self.root / "query.json").read_text())
        manifest["source_code_sha256"] = None
        with self.assertRaisesRegex(ValueError, "bounded tests"):
            metadata(manifest, "query")
        manifest["provenance_scope"] = "historical-build-code-unrecorded-test-only"
        self.assertIs(metadata(manifest, "query"), manifest)

    def test_corruption_rejected(self):
        path = self.root / "query.sig.gz"
        path.write_bytes(path.read_bytes()[:20])
        with self.assertRaisesRegex(ValueError, "checksum"):
            adopt(self.root / "query.json", path, "query", self.root / "bad.json")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.check()
        self.assertFalse((self.root / "validated.json").exists())

    def test_unknown_missing_duplicate_wrong_count_pairs_rejected(self):
        header = "query_name,match_name,intersect_hashes\n"
        for rows in (
            "",
            "other,RUN,2\n",
            "hit,RUN,1\n",
            "hit,RUN,2\nhit,RUN,2\n",
            "hit,OTHER,2\n",
        ):
            self.matches.write_text(header + rows)
            with self.assertRaises(ValueError):
                self.check()
            self.assertFalse((self.root / "validated.json").exists())

    def test_incomplete_log_rejected(self):
        for text in (
            "",
            "DONE. Processed 2 search sigs",
            "DONE. Processed 3 search sigs\nDONE. Processed 3 search sigs",
        ):
            self.log.write_text(text)
            with self.assertRaises(ValueError):
                self.check()

    def test_query_change_and_missing_membership(self):
        path = self.root / "query.json"
        manifest = json.loads(path.read_text())
        manifest["expected_ids"].append("absent")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "IDs"):
            adopt(path, self.root / "query.sig.gz", "query", self.root / "bad.json")

    def test_zero_matches_requires_independent_zero(self):
        # Remove overlap from reference, and produce a fresh receipt.
        (self.root / "reference-receipt.json").unlink()
        self.make("reference", {"RUN": [4]})
        self.matches.write_text("")
        self.check()
        result = json.loads((self.root / "validated.json").read_text())
        self.assertEqual(result["match_pairs"], 0)
        self.assertEqual(result["unmatched_ids"], ["empty", "hit", "miss"])


if __name__ == "__main__":
    unittest.main()
