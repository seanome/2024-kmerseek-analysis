"""A species with no reviewed Swiss-Prot entries must lose its swissprot arm, not the run.

On the 77-species run of 2026-09-10, pramorum and loculatus had zero reviewed entries.
build_swissprot_truth.py wrote each a <species>_domain_map.parquet with the right columns
and no rows, and recorded n_features 0 for them in swissprot_summary.json. Scoring refuses
an empty map on purpose, because for the Pfam arm it means the target annotation was never
built, so scoreDomainCalls failed on both species, exhausted its retries, and its `finish`
stopped every scoring task that had not started. Nothing after scoring ran.

The fix is a channel filter in main.nf, not a change to either script: the scripts are
hashed into the cached scoring tasks. Two things are guarded here:

  * main.nf gates the swissprot map channel on the summary's n_features, so the scorer's
    guard stays and is never reached by a map that is empty for an honest reason
  * the script's summary still reports n_features 0 for such a species, and its map is
    the thing the scorer refuses. If either changes, the gate reads the wrong key or
    protects against nothing.

Whether the whole workflow then completes needs a real Nextflow run over every stage and
is not tested here.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import BIN, FIXTURES, run, write_perfect_regions

MAIN_NF = Path(__file__).resolve().parents[1] / "main.nf"
SOURCE = MAIN_NF.read_text()


def _swissprot_block() -> str:
    """The workflow lines that wire the swissprot truth arm, comments dropped."""
    start = SOURCE.index("sprot = buildSwissprotTruth(")
    end = SOURCE.index('add_prebuilt("pfamn"', start)
    return "\n".join(l for l in SOURCE[start:end].splitlines()
                     if not l.lstrip().startswith("//"))


def test_main_nf_gates_the_swissprot_maps_on_summary_n_features():
    block = _swissprot_block()
    assert "sprot.summary" in block, (
        "the swissprot map channel must read buildSwissprotTruth's summary; without it "
        "an empty map for an uncurated proteome reaches scoreDomainCalls and fails the run"
    )
    assert "n_features" in block, "the gate must key on the summary's n_features"
    assert re.search(r"map_of\(sprot\.maps\)[\s\S]*?\.filter\s*\{", block), (
        "map_of(sprot.maps) must be filtered before it is mixed into map_ch"
    )
    assert "log.warn" in block, "a dropped species must be named in the log"


def _annotations_with_uncurated_species(tmp_path: Path) -> Path:
    """The fixture annotations plus a species whose accessions Swiss-Prot never reviewed."""
    ann = tmp_path / "annotations"
    shutil.copytree(FIXTURES / "annotations", ann)
    yeast = pl.read_parquet(ann / "yeast_pfam_domains.parquet")
    fake = yeast.with_columns(
        (pl.lit("X9") + pl.col("accession").str.slice(2)).alias("accession")
    )
    fake.write_parquet(ann / "nocuration_pfam_domains.parquet")
    return ann


def test_uncurated_species_gets_n_features_zero_and_an_empty_map(tmp_path):
    ann = _annotations_with_uncurated_species(tmp_path)
    out = tmp_path / "sprot"
    out.mkdir()
    run("build_swissprot_truth.py",
        "--sprot-dat", FIXTURES / "uniprot_sprot_fixture.dat",
        "--annotations", ann,
        "--truth-out", out / "human_swissprot_truth.parquet",
        "--map-outdir", out,
        "--summary-out", out / "swissprot_summary.json")

    summary = json.loads((out / "swissprot_summary.json").read_text())
    assert summary["nocuration"]["n_features"] == 0
    assert summary["yeast"]["n_features"] > 0
    assert pl.read_parquet(out / "nocuration_domain_map.parquet").height == 0, (
        "the script still writes a schema-only map for an uncurated species; the gate in "
        "main.nf exists because of that file"
    )

    # The selection main.nf makes from the summary, written out in Python: every key whose
    # value is a mapping with n_features, kept when n_features > 0. `human` carries
    # n_features too but never has a map, so it is harmless either way.
    kept = {k for k, v in summary.items()
            if isinstance(v, dict) and "n_features" in v and v["n_features"] > 0}
    assert "yeast" in kept
    assert "nocuration" not in kept

    # And the reason the gate is needed: the scorer refuses the empty map, and must keep
    # doing so, because for the Pfam arm an empty map is a build error.
    regions = tmp_path / "regions.tsv"
    write_perfect_regions(out / "human_swissprot_truth.parquet",
                          out / "yeast_domain_map.parquet", regions)
    workdir = tmp_path / "score"
    workdir.mkdir()
    (workdir / "manifest.tsv").write_text(f"perfect\tdefault\t{regions}\n")
    proc = subprocess.run(
        [sys.executable, str(BIN / "evaluate_domain_calls.py"),
         "--manifest", str(workdir / "manifest.tsv"),
         "--species", "nocuration", "--species-mya", "1000",
         "--truth", str(out / "human_swissprot_truth.parquet"),
         "--domain-map", str(out / "nocuration_domain_map.parquet"),
         "--truth-set", "swissprot"],
        cwd=workdir, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "empty domain map" in proc.stderr
