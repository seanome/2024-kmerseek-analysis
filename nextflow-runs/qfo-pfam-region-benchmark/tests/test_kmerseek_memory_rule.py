"""The kmerseek memory rule, checked against the measurements it was fitted to.

main.nf sizes kmerseekIndex and kmerseekSearch from ksize and alphabet cardinality. The
constants live in main.nf and this test reads them back out of it, so the file stays the
single source of truth and a tuning change that breaks the fit fails here rather than in
a queue three days later.

tests/fixtures/kmerseek_peak_rss.csv is the basis: the maximum peak_rss per (process,
species, alphabet, ksize) over 4_299 COMPLETED kmerseek tasks from the 2026-08-25, -26 and
-27 Sherlock traces, 15 alphabets from 2 to 20 classes, ksize 5-30, nine target proteomes.

`censored` marks rows whose peak came within 2% of the request. A peak equal to the request
to two decimals, repeated across nine different species, is a cgroup ceiling rather than a
coincidence -- those tasks were clipped and their true peaks are lower bounds, so the rule
has to clear them by more than the rest.
"""

import csv
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN_NF = ROOT / "main.nf"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kmerseek_peak_rss.csv"


def _param(name: str) -> str:
    """Read a `params.<name> = <value>` literal straight out of main.nf.

    Trailing `//` comments are stripped, since every constant in that block carries one.
    """
    m = re.search(rf"^params\.{name}\s*=\s*(.+?)\s*$", MAIN_NF.read_text(), re.M)
    assert m, f"params.{name} not found in main.nf"
    return m.group(1).split("//")[0].strip().strip("'\"")


def _gb(text: str) -> float:
    m = re.match(r"([\d.]+)\s*([KMGT]?B)$", text.strip())
    assert m, f"not a memory literal: {text}"
    return float(m.group(1)) * {"B": 1 / 1024**3, "KB": 1 / 1024**2,
                                "MB": 1 / 1024, "GB": 1, "TB": 1024}[m.group(2)]


@pytest.fixture(scope="module")
def rule():
    """Python mirror of kmerseekIndexMemory / kmerseekSearchMemory, constants from main.nf."""
    base = float(_param("kmerseek_memory_bits_base"))
    decay = float(_param("kmerseek_memory_bits_decay"))
    head = float(_param("kmerseek_memory_headroom"))
    size_frac = float(_param("kmerseek_memory_size_floor_frac"))
    ref_mb = float(_param("kmerseek_reference_proteome_mb"))
    floor = _gb(_param("kmerseek_memory_floor"))
    cap = _gb(_param("kmerseek_memory_max"))
    idx_cap = _gb(_param("kmerseek_index_memory_max"))
    # 'label:GB:maxKsize', several entries per label, tightest bracket wins. See main.nf.
    unmodelled = []
    for entry in _param("kmerseek_memory_unmodelled").split(","):
        if not entry.strip():
            continue
        parts = entry.strip().split(":")
        unmodelled.append((parts[0], float(parts[1]),
                           int(parts[2]) if len(parts) > 2 else 10**6))

    def classes(alphabet: str) -> int:
        m = re.search(r"(\d+)$", alphabet)
        return int(m.group(1)) if m else 20

    def ask(process: str, alphabet: str, ksize: int, mb: float) -> float:
        if process == "kmerseekIndex":
            cap_i = min(idx_cap, cap)
            return max(min(cap_i, floor), min(cap_i, 2.0 + 0.9 * mb))
        bits = ksize * math.log2(classes(alphabet))
        size_f = size_frac + (1 - size_frac) * min(1.0, mb / ref_mb)
        est = head * base * math.exp(-decay * bits) * size_f
        brackets = [(maxk, gb_v) for label, gb_v, maxk in unmodelled
                    if label == alphabet and ksize <= maxk]
        un_gb = min(brackets)[1] if brackets else 0.0
        floor_a = min(cap, max(floor, un_gb))
        return max(floor_a, min(cap, est))

    return ask


@pytest.fixture(scope="module")
def measured():
    with FIXTURE.open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "peak-RSS fixture is empty"
    for r in rows:
        r["ksize"] = int(r["ksize"])
        r["proteome_mb"] = float(r["proteome_mb"])
        r["max_peak_rss_gb"] = float(r["max_peak_rss_gb"])
        r["max_request_gb"] = float(r["max_request_gb"])
        r["n_tasks"] = int(r["n_tasks"])
        r["censored"] = r["censored"] == "1"
    return rows


def test_no_measured_combo_is_under_sized(rule, measured):
    """The whole point of a sizing rule: nothing it covers may OOM."""
    under = [(r, rule(r["process"], r["alphabet"], r["ksize"], r["proteome_mb"]))
             for r in measured
             if r["max_peak_rss_gb"] > rule(r["process"], r["alphabet"],
                                            r["ksize"], r["proteome_mb"])]
    assert not under, "under-sized combos:\n" + "\n".join(
        f"  {r['species']}_{r['alphabet']}_k{r['ksize']} ({r['process']}): "
        f"peak {r['max_peak_rss_gb']:.2f} GB > ask {ask:.2f} GB"
        for r, ask in sorted(under, key=lambda t: -t[0]["max_peak_rss_gb"])[:15])


def test_every_combo_keeps_at_least_1_25x_headroom(rule, measured):
    """Not OOMing is the floor; a rule with no slack turns a noisy peak into a requeue."""
    thin = []
    for r in measured:
        ask = rule(r["process"], r["alphabet"], r["ksize"], r["proteome_mb"])
        if ask < 1.25 * r["max_peak_rss_gb"]:
            thin.append((r, ask))
    assert not thin, "combos with under 1.25x headroom:\n" + "\n".join(
        f"  {r['species']}_{r['alphabet']}_k{r['ksize']} ({r['process']}): "
        f"peak {r['max_peak_rss_gb']:.2f} GB, ask {ask:.2f} GB"
        for r, ask in sorted(thin, key=lambda t: t[1] / t[0]["max_peak_rss_gb"])[:10])


def test_censored_combos_get_real_headroom(rule, measured):
    """A clipped peak is a lower bound, so it needs more than a hair of clearance."""
    tight = []
    for r in measured:
        if not r["censored"]:
            continue
        ask = rule(r["process"], r["alphabet"], r["ksize"], r["proteome_mb"])
        if ask < 1.5 * r["max_peak_rss_gb"]:
            tight.append((r, ask))
    assert not tight, "censored combos with under 1.5x headroom:\n" + "\n".join(
        f"  {r['species']}_{r['alphabet']}_k{r['ksize']}: clipped at "
        f"{r['max_peak_rss_gb']:.2f} GB, ask {ask:.2f} GB" for r, ask in tight[:10])


def test_the_rule_asks_for_less_than_the_run_that_measured_it(rule, measured):
    """Right-sizing has to actually reduce the reservation, weighted by task count."""
    old = sum(r["max_request_gb"] * r["n_tasks"] for r in measured)
    new = sum(rule(r["process"], r["alphabet"], r["ksize"], r["proteome_mb"])
              * r["n_tasks"] for r in measured)
    assert new < old, f"rule asks {new:,.0f} GB-tasks against the old {old:,.0f}"
    # Measured saving was 37% overall. Guard the direction and rough size, not the digit.
    assert new / old < 0.80, f"expected a clear saving, got {100 * new / old:.0f}%"


def test_index_is_sized_separately_and_far_smaller_than_search(rule, measured):
    """kmerseekIndex peaked at 7.00 GB over 1_500 tasks; it must not carry a search ask."""
    idx = [r for r in measured if r["process"] == "kmerseekIndex"]
    assert idx, "fixture has no index rows"
    assert max(r["max_peak_rss_gb"] for r in idx) <= 8.0, \
        "fixture changed: index peaks were <= 7.00 GB when the rule was written"
    worst_index_ask = max(rule("kmerseekIndex", r["alphabet"], r["ksize"],
                               r["proteome_mb"]) for r in idx)
    worst_search_ask = max(rule("kmerseekSearch", r["alphabet"], r["ksize"],
                                r["proteome_mb"]) for r in measured)
    assert worst_index_ask < worst_search_ask / 4, \
        "index and search are being sized alike again"


def test_memory_falls_as_the_keyspace_grows(rule):
    """The rule's shape: more bits is a bigger keyspace and less memory."""
    asks = [rule("kmerseekSearch", "hp_thomas_dill2", k, 16.0) for k in range(18, 31)]
    assert asks == sorted(asks, reverse=True), "ask must be monotone in ksize"
    # And a coarser alphabet at the same ksize must cost at least as much.
    assert (rule("kmerseekSearch", "hp_thomas_dill2", 12, 16.0)
            >= rule("kmerseekSearch", "protein20", 12, 16.0))


def test_small_proteomes_do_not_scale_to_nothing(rule):
    """yeast at 3 MB peaked at 25.7 GB on gbmr4 k=12; linear scaling under-sized it."""
    big = rule("kmerseekSearch", "gbmr4", 12, 16.0)
    small = rule("kmerseekSearch", "gbmr4", 12, 3.0)
    assert small > 25.7, f"yeast-sized gbmr4 k12 ask {small:.1f} GB is below its measured peak"
    assert small / big > 0.5, "target-size scaling has become effectively linear again"
