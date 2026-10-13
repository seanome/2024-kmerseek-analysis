"""Every process must survive a task the cluster kills.

Nextflow's default errorStrategy is `terminate`: the first failure cancels every task
still running. On 2026-09-05 one hhblitsBuildDB task on zmays came back with no exit
status -- "terminated for an unknown reason -- Likely it has been terminated by the
external system" -- and took 582 in-flight tasks with it, on a run that had been queueing
and computing for hours. Nothing about that task was wrong; it was killed.

Two regressions are guarded here, and both are the kind that are invisible until a run
dies:

  * a process added without any errorStrategy, which silently inherits `terminate`
  * `task.exitStatus in 128..143` used on its own. Nextflow sets exitStatus to
    Integer.MAX_VALUE when no .exitcode file was written, and 2147483647 is not in that
    range, so the test reads false and the strategy falls through to its non-retry branch
    on exactly the failure it was written for.

A third one is specific to kmerseekSearch. Its script tolerates a nonzero exit from
`kmerseek search` as a no-hit result, and until 2026-09-11 that tolerance included exit
137, the cgroup OOM kill: the shell wrote an empty parquet into storeDir, storeDir served
it as done on every later run, and scoring read it as recall 0. 178 of the 227 empty
region files on the 2026-09-10 run were kills. The script must re-raise a signal exit, and
the process must use the strategy that ignores an exhausted kill rather than stopping the
run, because a combo that cannot fit is an expected outcome of the sweep.
"""

import re
from pathlib import Path

MAIN_NF = Path(__file__).resolve().parents[1] / "main.nf"
SOURCE = MAIN_NF.read_text()

# Directives run from the process's opening brace to whichever body section comes first.
BODY_MARKERS = ("\n    input:", "\n    output:", "\n    script:", "\n    shell:",
                "\n    exec:")


def _processes() -> dict:
    """Map process name -> its directive block."""
    starts = [(m.start(), m.group(1))
              for m in re.finditer(r"^process ([A-Za-z]\w*) \{", SOURCE, re.M)]
    blocks = {}
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(SOURCE)
        chunk = SOURCE[pos:end]
        cuts = [chunk.index(m) for m in BODY_MARKERS if m in chunk]
        blocks[name] = chunk[:min(cuts)] if cuts else chunk
    return blocks


def test_every_process_declares_an_error_strategy():
    missing = sorted(n for n, block in _processes().items()
                     if "errorStrategy" not in block)
    assert not missing, (
        "these processes inherit Nextflow's default `terminate`, so one task killed by "
        "SLURM cancels the whole run: " + ", ".join(missing)
    )


def _code_lines(block: str) -> str:
    """The block with comment lines dropped -- the notes discuss 128..143 on purpose."""
    return "\n".join(l for l in block.splitlines() if not l.lstrip().startswith("//"))


def test_no_process_tests_the_signal_range_on_its_own():
    offenders = sorted(n for n, block in _processes().items()
                       if "128..143" in _code_lines(block))
    assert not offenders, (
        "`task.exitStatus in 128..143` misses the no-exit-code case, where Nextflow sets "
        "exitStatus to Integer.MAX_VALUE. Use retryOnKill: " + ", ".join(offenders)
    )


def _closure(name: str) -> str:
    m = re.search(rf"^def {name} = \{{.*?^\}}", SOURCE, re.M | re.S)
    assert m, f"{name} closure not found in main.nf"
    return m.group(0)


def test_kill_test_covers_the_no_exit_code_sentinel():
    body = _closure("killedByCluster")
    assert "Integer.MAX_VALUE" in body, (
        "killedByCluster must treat Integer.MAX_VALUE as a kill -- that is the value "
        "Nextflow reports when the task wrote no .exitcode at all"
    )


def test_retry_on_kill_ends_at_finish():
    body = _closure("retryOnKill")
    assert "killedByCluster(task)" in body, (
        "retryOnKill must decide kills through killedByCluster rather than restate the test"
    )
    assert "'finish'" in body and "'terminate'" not in body, (
        "every path out of retryOnKill must end at `finish`, so a task that cannot be made "
        "to work costs its own arm rather than every task still running"
    )


def test_retry_on_kill_else_ignore_shares_the_kill_test():
    body = _closure("retryOnKillElseIgnore")
    assert "killedByCluster(task)" in body and "128..143" not in _code_lines(body), (
        "retryOnKillElseIgnore must decide kills through killedByCluster, not a second "
        "copy of the signal range that can drift from the first"
    )
    assert "'retry'" in body and "'ignore'" in body and "'finish'" in body, (
        "retryOnKillElseIgnore has three exits: retry a kill under the cap, ignore a kill "
        "past it, finish on anything that is not a kill"
    )
    assert "'terminate'" not in body
    assert "log.warn" in body, "an ignored kill must be named in the log, not vanish"


def test_kmerseek_search_reraises_a_signal_exit_and_ignores_an_exhausted_kill():
    block = _processes()["kmerseekSearch"]
    assert "retryOnKillElseIgnore(task)" in _code_lines(block), (
        "kmerseekSearch must use retryOnKillElseIgnore: a combo that cannot fit after two "
        "doublings is an expected outcome of the sweep, and `finish` would stop the run on it"
    )
    start = SOURCE.index("process kmerseekSearch {")
    end = SOURCE.index("\nprocess ", start + 1)
    script = SOURCE[start:end]
    # The re-raise has to test kmerseek's own exit (PIPESTATUS[0], held in rc[0]) against
    # the signal range and exit with that status, so errorStrategy sees the kill.
    reraise = re.search(
        r'\[ "\\\$\{rc\[0\]\}" -ge 128 \] && \[ "\\\$\{rc\[0\]\}" -le 143 \]; then'
        r'.*?exit "\\\$\{rc\[0\]\}"',
        script, re.S)
    assert reraise, (
        "kmerseekSearch must re-raise a signal exit of `kmerseek search` (128..143, 137 is "
        "the OOM kill) instead of storing it as an empty no-hit parquet under storeDir"
    )
    # And it must come before the no-hit tolerance, or the tolerance swallows it.
    tolerance = script.index("treating as a no-hit result")
    assert reraise.start() < tolerance
