#!/usr/bin/env python3
"""Check score_transfers.py on a made-up case whose answer is worked out by hand.

    python3 test_score_transfers.py

Four real queries and their decoys. R1 and R2 are in the tune half, R3 and R4 in the test
half. P4 instances: R1 (Motif), R3 (Repeat), R4 (36 aa); R2 is a 191 aa Domain.
- kmerseek hp_pbotc_1st_ed2 k19 extend finds R1, R3 and R4. Its tune decoy (DECOY_R1, E 10)
  sets the cutoff just above that score; its test decoy DECOY_R3 transfers at E 1e-6.
- kmerseek protein20 k5 exact finds R1; its tune decoy DECOY_R2 scores 2.
- hmmscan finds R4 only. MMseqs2 and phmmer find nothing.
Tune: both kmerseek settings have 1 unique find (R1) and 0 expected, so the tie goes to the
lower decoy rate (both 0), then to the name: hp_pbotc_1st_ed2 k19 extend.
Test: R3 is a unique find, R4 is not (hmmscan has it); decoy rate 1 transfer / 2 decoys,
times 2 P4 queries = 1 expected, so unique minus expected = 0 and P4 does not hold.
In pilot mode (no --read-test-half) no test-half query appears in any output.

Seed reference (Pfam 33.1 seed members as targets, notebook 280's second reference): each
entry's one domain row spans the whole entry, so a hit anywhere on it transfers that
family, through compact_kmerseek.awk (kmerseek) and transfer_by_overlap (MMseqs2, phmmer).
make_search_inputs.py reference-pfam-seed is checked on a two-family Stockholm file.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KM_HEADER = (
    "query\tqstart\tqend\ttarget\ttstart\ttend\tpfam_acc\tdstart\tdend\tregion_length\t"
    "region_n_shared_kmers\tregion_poisson_score\tregion_tfidf\tregion_mean_idf\t"
    "region_n_mismatches\tregion_ka_bits\tregion_evalue\tquery_poisson_pvalue\n"
)


def km_row(q, s, e, fam, poisson, evalue):
    return f"{q}\t{s}\t{e}\tP9\t1\t20\t{fam}\t1\t50\t20\t3\t{poisson}\t1\t1\t0\t10\t{evalue}\t0.01\n"


def write_gz(path: Path, text: str) -> None:
    with gzip.open(path, "wt") as fh:
        fh.write(text)


def build(d: Path) -> None:
    for sub in (
        "inputs",
        "reference",
        "kmerseek",
        "stats",
        "mmseqs2",
        "phmmer",
        "hmmscan",
    ):
        (d / sub).mkdir()
    q = ["query\tmd5\tis_decoy\tsource\thalf\trole\tlength"]
    for name, half in (("R1", "tune"), ("R2", "tune"), ("R3", "test"), ("R4", "test")):
        q += [
            f"{name}\tx\tfalse\t{name}\t{half}\tsample\t300",
            f"DECOY_{name}\ty\ttrue\t{name}\t{half}\tsample\t300",
        ]
    (d / "inputs/queries.tsv").write_text("\n".join(q) + "\n")
    (d / "inputs/instances.tsv").write_text(
        "query\trole\tpfam_acc\tpfam_id\tpfam_type\tclan_t1\tt1_start\tt1_end\tlength\tlength_bin\tkingdom\tuncharacterized_t0\n"
        "R1\tsample\tPF1\tf1\tMotif\tNA\t10\t30\t21\t<60 aa\tBacteria\ttrue\n"
        "R2\tsample\tPF2\tf2\tDomain\tNA\t10\t200\t191\t>=60 aa\tBacteria\ttrue\n"
        "R3\tsample\tPF1\tf1\tRepeat\tNA\t10\t50\t41\t<60 aa\tBacteria\ttrue\n"
        "R4\tsample\tPF3\tf3\tDomain\tNA\t5\t40\t36\t<60 aa\tBacteria\ttrue\n"
    )
    (d / "reference/sprot_domains.tsv").write_text(
        "sp_acc\tpfam_acc\tsp_start\tsp_end\n"
    )
    write_gz(
        d / "kmerseek/hp_pbotc_1st_ed2.k19.extend.chunk.transfers.tsv.gz",
        KM_HEADER
        + km_row("R1", 12, 28, "PF1", 3, "1e-5")
        + km_row("R3", 12, 40, "PF1", 3, "1e-4")
        + km_row("R4", 6, 38, "PF3", 3, "1e-3")
        + km_row("DECOY_R1", 1, 20, "PF9", 3, "10")
        + km_row("DECOY_R3", 1, 20, "PF9", 3, "1e-6"),
    )
    write_gz(
        d / "kmerseek/protein20.k5.exact.chunk.transfers.tsv.gz",
        KM_HEADER
        + km_row("R1", 12, 28, "PF1", 5, "inf")
        + km_row("DECOY_R2", 1, 20, "PF8", 2, "inf"),
    )
    write_gz(d / "mmseqs2/chunk.mmseqs2.tsv.gz", "")
    write_gz(d / "phmmer/chunk.phmmer.domtbl.gz", "")
    write_gz(
        d / "hmmscan/chunk.hmmscan.domtbl.gz",
        "f3 PF3.5 36 R4 - 300 1e-8 30.0 0.0 1 1 1e-9 1e-9 30.0 0.0 1 36 6 38 5 40 0.9 desc\n",
    )


def run(d: Path, out: str, *extra: str) -> dict:
    (d / out).mkdir()
    args = [sys.executable, str(HERE / "score_transfers.py")]
    for a in (
        "kmerseek",
        "stats",
        "mmseqs2",
        "phmmer",
        "hmmscan",
        "inputs",
        "reference",
    ):
        args += [f"--{a}", str(d / a)]
    args += ["--settings", str(d / "inputs/queries.tsv"), "--out", str(d / out), *extra]
    subprocess.run(args, check=True, capture_output=True)
    return json.loads((d / out / "p4.json").read_text())


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        build(d)
        full = run(d, "full", "--read-test-half")
        assert full["chosen"] == "kmerseek hp_pbotc_1st_ed2 k19 extend", full
        assert full["tune_unique_finds"] == 1 and full["tune_expected"] == 0.0, full
        assert full["test_unique_finds"] == 1 and full["test_p4_queries"] == 2, full
        assert (
            full["test_decoy_rate"] == 0.5 and full["test_unique_minus_expected"] == 0.0
        ), full
        assert full["p4_holds"] is False, full
        assert "R3" in (d / "full/unique_finds_test_instances.csv").read_text()

        pilot = run(d, "pilot")
        assert not any(k.startswith("test_") for k in pilot), pilot
        for f in (x for x in (d / "pilot").rglob("*") if x.is_file()):
            text = f.read_bytes()
            for name in (b"R3", b"R4"):
                assert name not in text, f"{name} in {f.name}"
        check_seed_reference(d)
    print("score_transfers.py: all checks passed")


SEED_STOCKHOLM = """# STOCKHOLM 1.0
#=GF ID   fam1
#=GF AC   PF00001.21
#=GS A_HUMAN/11-40 AC P11111.2
#=GS B_MOUSE/5-34 AC P22222.1
#=GS Q_SELF/1-30 AC QSELF.1
A_HUMAN/11-40  MKVLAAGIVG..LLAAGCSSHHkkPPEEDDRRW
B_MOUSE/5-34   MKVLAAGIVG..LLAAGCSSHHKKPPEEDDRRW
Q_SELF/1-30    MSTNPKPQRK..TKRNTNRRPQDVKFPGGGQ-
#=GC seq_cons  MKVLAAGIVG..LLAAGCSSHHKKPPEEDDRRW
//
# STOCKHOLM 1.0
#=GF ID   fam2
#=GF AC   PF00002.9
#=GS A2_HUMAN/1-31 AC P33333.1
#=GS IN_Q/3-24 AC P44444.1
A2_HUMAN/1-31  MKVLAAGIVG--LLAAGCSSHHKKPPEEDDRRW
IN_Q/3-24      ---AAAWWWYYYCCCDDDEEEFFF.........
//
"""


def check_seed_reference(d: Path) -> None:
    sys.path.insert(0, str(HERE))
    import polars as pl
    from score_transfers import transfer_by_overlap

    s = d / "seed"
    s.mkdir()
    write_gz(s / "Pfam-A.seed.gz", SEED_STOCKHOLM)
    # QSELF is a query; IN_Q's member lies inside query Q2; the decoy must not count.
    (s / "queries.fasta").write_text(
        ">QSELF\nMSTNPKPQRKTKRNTNRRPQDVKFPGGGQ\n"
        ">Q2\nMMMAAAWWWYYYCCCDDDEEEFFFGG\n>DECOY_Q2\nGGFFFEEEDDDCCCYYYWWWAAAMMM\n"
    )
    (s / "sprot.fasta").write_text(">P1\n" + "A" * 400 + "\n")
    subprocess.run(
        [
            sys.executable,
            str(HERE / "make_search_inputs.py"),
            "reference-pfam-seed",
            "--seed",
            str(s / "Pfam-A.seed.gz"),
            "--sprot",
            str(s / "sprot.fasta"),
            "--queries",
            str(s / "queries.fasta"),
            "--out",
            str(s / "out"),
        ],
        check=True,
        capture_output=True,
    )
    fasta = (s / "out/reference.fasta").read_text()
    seq = "MKVLAAGIVGLLAAGCSSHHKKPPEEDDRRW"
    # A_HUMAN, B_MOUSE and A2_HUMAN are one sequence (lower case and gaps removed):
    # one entry, named after the first member, with a row for each family.
    assert fasta == f">PF00001__P11111_11_40\n{seq}\n", fasta
    dom = pl.read_csv(s / "out/sprot_domains.tsv", separator="\t")
    assert dom.rows() == [
        ("PF00001__P11111_11_40", "PF00001", 1, 31),
        ("PF00001__P11111_11_40", "PF00002", 1, 31),
    ], dom
    c = json.loads((s / "out/counts.json").read_text())
    assert c["seed_members"] == 5 and c["entries"] == 1, c
    assert c["members_dropped_accession_is_a_query"] == 1, c
    assert c["members_dropped_sequence_inside_a_query"] == 1, c
    assert c["members_sharing_an_entry"] == 2 and c["residues"] == 31, c
    assert abs(c["bits_saved_log2_ratio"] - 3.6897) < 1e-3, c  # log2(400 / 31)

    # A hit on residues 5-20 of the entry transfers both families, through each path.
    hits = pl.DataFrame(
        {
            "query": ["R1"],
            "target": ["PF00001__P11111_11_40"],
            "tstart": [5],
            "tend": [20],
        }
    )
    assert sorted(transfer_by_overlap(hits, dom)["pfam_acc"]) == ["PF00001", "PF00002"]
    km = (
        "query_name,target_name,region_start,region_end,target_start,target_end,"
        "region_length,region_n_shared_kmers,region_poisson_score,region_tfidf,"
        "region_mean_idf,region_n_mismatches,region_ka_bits,region_evalue,"
        "query_poisson_pvalue\n"
        "R1,PF00001__P11111_11_40,9,25,4,20,16,3,2.5,1,1,0,9,0.01,0.001\n"
        "R1,OTHER,9,25,4,20,16,3,2.5,1,1,0,9,0.01,0.001\n"
    )
    out = subprocess.run(
        [
            "awk",
            "-F,",
            "-v",
            f"domains={s / 'out/sprot_domains.tsv'}",
            "-f",
            str(HERE / "compact_kmerseek.awk"),
        ],
        input=km,
        text=True,
        check=True,
        capture_output=True,
    ).stdout.splitlines()
    rows = [r.split("\t") for r in out[1:]]
    # 0-based start 4, end-exclusive 20 -> residues 5-20; one row per family.
    assert [(r[3], r[4], r[5], r[6]) for r in rows] == [
        ("PF00001__P11111_11_40", "5", "20", "PF00001"),
        ("PF00001__P11111_11_40", "5", "20", "PF00002"),
    ], out


if __name__ == "__main__":
    main()
