"""Resource prediction must cover the failed first asks before a resume."""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
from measure_resources import predict, quantity


def test_memory_failures_raise_first_ask_and_runtime_has_headroom(tmp_path):
    path = tmp_path / "trace.txt"
    fields = ["process", "status", "exit", "memory", "time", "realtime"]
    with path.open("w") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(fields)
        w.writerows(
            [
                ["alignInput", "FAILED", "137", "48 GB", "12h", "7m"],
                ["targets", "FAILED", "137", "16 GB", "3h", "2m"],
                ["targets", "COMPLETED", "0", "32 GB", "6h", "5h 30m"],
            ]
        )
    result = predict([path])
    assert result["alignInput"][:2] == (128, 36)
    assert result["targets"][:2] == (32, 9)
    assert quantity("1d 2h 30m", {"d": 86400, "h": 3600, "m": 60}) == 95400


def test_refuses_memory_above_available_ceiling(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text(
        "process\tstatus\texit\tmemory\ttime\trealtime\nlarge\tFAILED\t137\t300 GB\t1h\t1m\n"
    )
    with pytest.raises(ValueError, match="ceiling"):
        predict([path])
