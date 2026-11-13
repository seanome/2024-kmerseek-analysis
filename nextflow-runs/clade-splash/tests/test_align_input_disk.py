"""Compare disk-backed alignment preparation with the original on bounded fixtures."""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))
import clade_splash_align_input as disk
import clade_splash_pfam as original


@pytest.mark.parametrize(
    "empty,duplicate_bucket", [(False, False), (True, False), (False, True)]
)
def test_matches_original_across_files_and_domain_boundaries(
    tmp_path, empty, duplicate_bucket
):
    anchors = pl.DataFrame(
        {
            "setting": ["dayhoff:13"] * 3,
            "gap": [0] * 3,
            "anchor": [2**63 + 1, 7, 8],
            "main_pfam": ["PF00001"] * 3,
            "outside_pfam": [False, False, True],
            "n_occurrences": [4, 4, 4],
            "n_occ_in_pfam": [2, 2, 0],
        },
        schema_overrides={"anchor": pl.UInt64},
    )
    if empty:
        anchors = anchors.head(0)
    anchors.write_parquet(tmp_path / "pfam.parquet")
    # Duplicate occurrences within a bucket are collapsed. One target
    # lies outside its domain; its anchor still has an empty FASTA and manifest row.
    occ = pl.DataFrame(
        {
            "setting": ["dayhoff:13"] * 4,
            "gap": [0] * 4,
            "anchor": [2**63 + 1, 2**63 + 1, 7, 8],
            "species_idx": [0, 1, 0, 0],
            "protein_idx": [0] * 4,
            "pos0": [0, 1, 30, 0],
        },
        schema_overrides={"anchor": pl.UInt64},
    )
    occ = pl.concat([occ, occ.head(1)])
    paths = []
    for i, part in enumerate(
        [
            occ.filter(pl.col("anchor") > 2**63),
            occ.filter(pl.col("anchor") < 2**63),
            occ.head(0),
        ]
    ):
        path = tmp_path / f"occ{i}.parquet"
        part.write_parquet(path)
        paths.append(str(path))
    if duplicate_bucket:
        paths.append(paths[0])
    dom = tmp_path / "domains.txt"
    with dom.open("w") as fh:
        fh.write("# comment\n\n")
        for query in ["s0_p0", "s1_p0"]:
            for lo, hi in [(1, 30), (2, 31)]:
                fields = (
                    [query, "-", "60", "FAKE", "PF00001.1", "30"]
                    + ["0"] * 13
                    + [str(lo), str(hi), "0.9", "-"]
                )
                fh.write(" ".join(fields) + "\n")
    fasta = tmp_path / "proteins.fa"
    fasta.write_text(
        ">s0_p0\nACDEFGHIKLMNPQRSTVWY\n" + "A" * 40 + "\n>s1_p0\n" + "C" * 60 + "\n"
    )
    # Duplicate protein in another FASTA must keep the first sequence.
    duplicate = tmp_path / "duplicate.fa"
    duplicate.write_text(">s0_p0\n" + "D" * 60 + "\n")
    results = []
    for name, module in [("old", original), ("new", disk)]:
        root = tmp_path / name
        root.mkdir()
        args = [
            "--pfam",
            str(tmp_path / "pfam.parquet"),
            "--occurrences",
            *paths,
            "--domtblout",
            str(dom),
            "--fastas",
            str(fasta),
            str(duplicate),
            "--outdir",
            str(root / "fasta"),
            "--manifest",
            str(root / "manifest.tsv"),
            "--parts-dir",
            str(root / "parts"),
            "--part-size",
            "1",
        ]
        if name == "new" and duplicate_bucket:
            with pytest.raises(ValueError, match="more than one hash bucket"):
                module.main(args)
            return
        module.main((["align-input"] if name == "old" else []) + args)
        results.append(
            {
                str(p.relative_to(root)): p.read_bytes()
                for p in root.rglob("*")
                if p.is_file()
            }
        )
    assert results[0] == results[1]
    if not empty:
        assert len(list((tmp_path / "new" / "fasta").glob("*.fa"))) == 2
        assert (
            tmp_path / "new" / "fasta" / "dayhoff_13_g0_7_PF00001.fa"
        ).read_text() == ""
