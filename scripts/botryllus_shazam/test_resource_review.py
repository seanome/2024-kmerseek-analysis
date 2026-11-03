import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

from measure_nextflow_resources import measure


class ResourceReviewTests(unittest.TestCase):
    def test_only_exact_diagnosed_oom_is_released(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            trace = root / "trace.tsv"
            row = dict(
                task_id="246",
                name="COMPACT (splash)",
                status="FAILED",
                exit="1",
                memory="96 GB",
                time="2d",
                peak_rss="-",
                realtime="7m 50s",
                workdir=str(root),
            )
            with trace.open("w") as f:
                writer = csv.DictWriter(f, row.keys(), delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            review = dict(
                trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
                task_id="246",
                name=row["name"],
                cause="oom",
            )
            (root / ".command.log").write_text("Detected 1 oom_kill event")
            self.assertEqual(len(measure([trace])["COMPACT"]["failures"]), 1)
            result = measure([trace], [review])["COMPACT"]
            self.assertEqual(result["memory_gb"], 192)
            self.assertEqual(result["time_hours"], 48)
            self.assertEqual(result["failures"], [])
            self.assertEqual(len(result["reviewed_failures"]), 1)
            (root / ".command.log").write_text("Unrelated exit 1")
            with self.assertRaisesRegex(ValueError, "OOM evidence"):
                measure([trace], [review])
            with self.assertRaisesRegex(ValueError, "does not match"):
                measure([trace], [dict(review, task_id="247")])
