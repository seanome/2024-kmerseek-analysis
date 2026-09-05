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


def test_retry_on_kill_covers_the_no_exit_code_sentinel():
    closure = re.search(r"^def retryOnKill = \{.*?^\}", SOURCE, re.M | re.S)
    assert closure, "retryOnKill closure not found in main.nf"
    body = closure.group(0)
    assert "Integer.MAX_VALUE" in body, (
        "retryOnKill must treat Integer.MAX_VALUE as a kill -- that is the value Nextflow "
        "reports when the task wrote no .exitcode at all"
    )
    assert "'finish'" in body and "'terminate'" not in body, (
        "every path out of retryOnKill must end at `finish`, so a task that cannot be made "
        "to work costs its own arm rather than every task still running"
    )
