"""Shared fixtures: build the truth sets once, then let each test read them.

Everything here runs the REAL bin/ scripts through subprocess rather than importing them.
That is deliberate: the pipeline invokes them as commands with a CLI, and an import-level
test would not catch an argparse change, a missing default, or a script that only works
when its sibling module is on sys.path.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

# Standalone scripts under tests/ that are NOT pytest modules: they run at import and call
# sys.exit, which pytest reports as an INTERNALERROR during collection. Listed here rather
# than converted, because each documents its own `python3 tests/<name>.py` invocation and
# has an exit-code contract. Naming one that does not exist on this branch is harmless, and
# keeps collection working the moment it arrives from another.
collect_ignore = ["test_dedup_passes.py"]

BIN = Path(__file__).resolve().parents[1] / "bin"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run(script: str, *args) -> subprocess.CompletedProcess:
    """Run a bin/ script the way Nextflow does, and fail loudly with its stderr."""
    cmd = [sys.executable, str(BIN / script), *map(str, args)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(
            f"{script} exited {proc.returncode}\n"
            f"--- cmd ---\n{' '.join(cmd)}\n--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return proc


@pytest.fixture(scope="session")
def pfam_truth(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("pfam_truth")
    run("build_domain_truth.py",
        "--annotations", FIXTURES / "annotations",
        "--truth-out", d / "human_domain_truth.parquet",
        "--map-outdir", d,
        "--summary-out", d / "summary.json")
    return {"dir": d, "truth": d / "human_domain_truth.parquet",
            "map": d / "yeast_domain_map.parquet"}


@pytest.fixture(scope="session")
def swissprot_truth(tmp_path_factory) -> dict:
    d = tmp_path_factory.mktemp("sprot_truth")
    run("build_swissprot_truth.py",
        "--sprot-dat", FIXTURES / "uniprot_sprot_fixture.dat",
        "--annotations", FIXTURES / "annotations",
        "--truth-out", d / "human_swissprot_truth.parquet",
        "--map-outdir", d,
        "--summary-out", d / "summary.json")
    return {"dir": d, "truth": d / "human_swissprot_truth.parquet",
            "map": d / "yeast_domain_map.parquet"}


def write_perfect_regions(truth_path: Path, target_map: Path, out: Path,
                          pad: int = 0) -> int:
    """A tool that is exactly right, as an 8-column region TSV.

    For every human truth instance, emit one region whose QUERY interval is the instance
    itself and whose TARGET interval is a same-family domain in the target proteome. That
    makes the transfer step's target-side coverage 1.0 and the scoring step's query-side
    IoU 1.0, so a correct scorer must return recall_reachable and precision of exactly 1.0.

    A known answer is the point. Scoring real tool output tells you the number moved; this
    tells you which direction is right.

    `pad` widens the QUERY interval on both sides, which is how a real alignment actually
    looks: it runs past the annotated feature rather than stopping on its boundary. The
    shortest region any tool emitted on the mini set was 3 residues, and the median 169.
    Padding matters for point features specifically -- an unpadded region on a 1-residue
    truth interval is itself 1 residue and scores IoU 1.0, which no real tool could ever
    achieve, so testing the point-semantics rule needs a realistic width or it tests
    nothing.
    """
    truth = pl.read_parquet(truth_path)
    tmap = pl.read_parquet(target_map)
    # One target domain per family is enough, and picking the first by sort keeps the
    # fixture deterministic across polars versions.
    per_fam = (tmap.sort("accession", "domain_start")
                   .group_by("pfam_id")
                   .agg(pl.col("accession").first(),
                        pl.col("domain_start").first(),
                        pl.col("domain_end").first()))
    joined = truth.join(per_fam, on="pfam_id", how="inner")
    lines = []
    for r in joined.iter_rows(named=True):
        qs = max(1, r["domain_start"] - pad)
        qe = r["domain_end"] + pad
        lines.append("\t".join(str(x) for x in (
            r["accession"], r["accession_right"],
            qs, qe,
            r["domain_start_right"], r["domain_end_right"],
            100.0, 1e-40,
        )))
    out.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


@pytest.fixture(scope="session")
def perfect_pfam_calls(pfam_truth, tmp_path_factory) -> pl.DataFrame:
    """Per-call detail for the perfect arm, so a test can assert WHY a call was wrong."""
    d = tmp_path_factory.mktemp("perfect_calls")
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], d / "regions.tsv")
    score(pfam_truth["truth"], pfam_truth["map"], d / "regions.tsv", d / "run", "pfam")
    files = sorted((d / "run").glob("*.calls.parquet"))
    assert files, "no calls parquet written"
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def write_decoy_regions(truth_path: Path, target_map: Path, out: Path,
                        n: int = 12, width: int = 150, score: float = 9_999.0) -> int:
    """High-scoring calls that land in UNANNOTATED stretches of a real protein.

    These become gray, not false positives: classify_scoreable excludes a call whose
    territory the annotation never covered from the precision denominator rather than
    charging it to the tool. Given the top score, they form a leading threshold block with
    zero SCOREABLE calls in it -- tp_calls and fp_calls both 0 -- which is the exact shape
    that made precision come out 0/0 = NaN. polars sorts NaN as the largest float, so
    `sort("f1", descending=True).head(1)` then returned that row as best_f1.

    Without this the perfect-tool fixture has no gray calls at all and the NaN guard is
    untestable: the metric simply never reaches the branch.

    Appended to a regions file rather than replacing it, so the arm still has real
    positives and the curve has somewhere to go.
    """
    truth = pl.read_parquet(truth_path)
    tmap = pl.read_parquet(target_map)
    target = tmap.sort("accession", "domain_start").head(1).to_dicts()
    if not target:
        return 0
    tgt = target[0]

    lines, made = [], 0
    for acc, grp in truth.group_by("accession"):
        acc = acc[0] if isinstance(acc, tuple) else acc
        plen = grp["protein_length"].max()
        if plen is None:
            continue
        # The first gap after the last annotated domain that is wide enough to sit in.
        last_end = int(grp["domain_end"].max())
        if plen - last_end < width + 2:
            continue
        qs, qe = last_end + 1, min(int(plen), last_end + width)
        lines.append("\t".join(str(x) for x in (
            acc, tgt["accession"], qs, qe,
            tgt["domain_start"], tgt["domain_end"], score, 1e-99)))
        made += 1
        if made >= n:
            break

    with open(out, "a") as f:
        for line in lines:
            f.write(line + "\n")
    return made


def score(truth: Path, target_map: Path, regions: Path, workdir: Path,
          truth_set: str, *extra, tool: str = "perfect",
          variant: str = "default") -> pl.DataFrame:
    """Run evaluate_domain_calls on one arm and return its metric rows.

    `tool`/`variant` are settable because several report sections key off them: the
    alphabet and ceiling panels only draw kmerseek rows, and only parse a variant spelled
    <alphabet>_k<k>_lc<True|False>. A test that wants those panels has to present itself
    as that arm.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = workdir / "manifest.tsv"
    manifest.write_text(f"{tool}\t{variant}\t{regions}\n")
    subprocess.run(
        [sys.executable, str(BIN / "evaluate_domain_calls.py"),
         "--manifest", str(manifest), "--species", "yeast", "--species-mya", "1000",
         "--truth", str(truth), "--domain-map", str(target_map),
         # The un-deduplicated arm only. The dedup-transfers pass is NOT deterministic on
         # this branch -- it drops rows by a row index computed inside a lazy plan, the plan
         # is re-executed per collect, and a re-executed scan does not return rows in the
         # same order. Four identical runs here differ on 20 of 40 metric cells, and every
         # one of them is a dedup_transfers=true row while the off arm is stable to the
         # last integer.
         #
         # Sweeping it would make every assertion below flaky for a reason none of them is
         # about. That pass has its own fix and its own test in flight on
         # olgabot/score-domain-calls-oom; widen this to "off,on" once that lands.
         "--dedup-transfer-modes", "off",
         "--truth-set", truth_set, *map(str, extra)],
        cwd=workdir, capture_output=True, text=True, check=True)
    files = sorted(workdir.glob("*.metrics.parquet"))
    assert files, f"no metrics written in {workdir}"
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
