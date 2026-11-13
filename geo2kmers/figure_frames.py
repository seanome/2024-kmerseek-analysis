"""Draw the headline figure of the frame tests.

Usage: figure_frames.py <results/frame_test folder>
Writes frame_choice_human_mussel.svg and .png to that folder.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import pubfig as pf

folder = Path(sys.argv[1])


def table(name):
    with open(folder / name) as f:
        return list(csv.DictReader(f, delimiter="\t"))


pf.use_style()
fig, (left, right) = pf.figure(pf.TWO_COLUMN_MM, 50, ncols=2)

rule = {r["min_longest_stretch_aa"]: r for r in table("longest_stretch_GPL1211.tsv")}
picked = next(
    r
    for r in table("frame_truth_by_length.tsv")
    if (r["alphabet"], r["k"], r["min_longest_stretch_aa"]) == ("protein", "10", "100")
)
rows = [
    ("Longest stretch, no proteome", rule["0"]),
    ("Longest stretch, at least 100 aa", rule["100"]),
    ("Human proteome, protein k = 10,\nat least 100 aa", picked),
]
for y, (label, r) in enumerate(rows):
    share = 100 * int(r["longest_in_true_frame"]) / int(r["sequences"])
    left.plot(share, y, "o", color=pf.OKABE_ITO["black"])
    left.text(share - 3, y, f"{r['longest_in_true_frame']} of {r['sequences']}", ha="right", va="center")
left.set_yticks(range(3), [r[0] for r in rows])
left.set_ylim(2.5, -0.5)
left.set_xlim(0, 100)
left.grid(axis="y", color="#dddddd")
left.set_xlabel("Human GPL1211 sequences in the true frame (%)")

settings = [("protein", "10", "protein, k = 10"), ("dayhoff", "19", "dayhoff, k = 19"), ("hp", "30", "hp, k = 30")]
proteomes = [
    ("mussel_vs_same_species.tsv", "Same species", pf.OKABE_ITO["blue"], "o"),
    ("mussel_vs_oyster.tsv", "Pacific oyster", pf.OKABE_ITO["vermillion"], "s"),
    ("mussel_vs_human_mouse.tsv", "Human and mouse", pf.OKABE_ITO["bluish_green"], "^"),
]
for name, label, colour, marker in proteomes:
    found = {(r["alphabet"], r["k"]): r for r in table(name)}
    xs = [int(found[a, k]["tags_stretch_ge90pct"]) for a, k, _ in settings]
    right.plot(xs, range(len(settings)), marker, color=colour, label=label, linestyle="none")
right.set_yticks(range(len(settings)), [s[2] for s in settings])
right.set_ylim(2.5, -0.5)
right.grid(axis="y", color="#dddddd")
right.set_xlim(0, 550)
right.set_xlabel("Mussel GPL1799 sequences that pass (of 1672)")
pf.shared_legend(fig)
for ax, letter in zip((left, right), "ab"):
    pf.panel_label(ax, letter)
pf.save(fig, folder / "frame_choice_human_mussel")
