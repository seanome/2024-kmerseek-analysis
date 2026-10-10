"""How kmerseek's seven H/P alphabets classify each of the 20 amino acids.

One row per alphabet, one square per residue, with the class letter (h, p or c) inside.
The seven alphabets agree on 15 residues (AFILMV hydrophobic, DEHKNQRST polar) and differ
only on C, G, P, W and Y.

The classes are parsed from kmerseek's ``src/rust/alphabets.rs`` at tag v0.4.0 (the
``build_hp`` / ``build_hpc`` calls), not typed here. The file is downloaded from GitHub
and checked against its SHA-256, or read from ``--alphabets-rs``.

Usage:
    python scripts/plot_hp_borderline_residues.py
    python scripts/plot_hp_borderline_residues.py --alphabets-rs ../kmerseek/src/rust/alphabets.rs

Writes figures/hp_borderline_residues.{pdf,svg,png} and tables/hp_borderline_residues.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
sys.path.insert(0, str(REPO / "scripts"))
import pubfig as pf  # noqa: E402
from shrink_png import process as shrink_png  # noqa: E402

KMERSEEK_TAG = "v0.4.0"
KMERSEEK_COMMIT = "0aed5ca4a81d0e1028db82dfed19a1a74eac06fa"
ALPHABETS_RS_URL = f"https://raw.githubusercontent.com/seanome/kmerseek/{KMERSEEK_TAG}/src/rust/alphabets.rs"
ALPHABETS_RS_SHA256 = "5ec2cb9fedfd83718e34ca27f3b15ce5dd4bde037ec2bb4e05105400ae1b5485"

ALWAYS_H = "AFILMV"
BORDERLINE = "CGPWY"
ALWAYS_P = "DEHKNQRST"
GROUPS = [
    (ALWAYS_H, "h in all seven"),
    (BORDERLINE, "differs"),
    (ALWAYS_P, "p in all seven"),
]

# Alphabets with each borderline residue in h, from the request and the v0.4.0 test.
EXPECTED_N_IN_H = {"C": 4, "G": 3, "P": 4, "W": 6, "Y": 6}

# Same amber and blue as Figure 1 (C_H, C_P in scripts/make_nb275.py, PR 95).
C_H = pf.OKABE_ITO["orange"]
C_P = pf.OKABE_ITO["blue"]
C_C = pf.GREY
FILL = {"h": C_H, "p": C_P, "c": C_C}
INK = {"h": "black", "p": "white", "c": "black"}
CLASS_NAME = {"h": "hydrophobic", "p": "polar", "c": "cysteine in its own class"}


def read_alphabets_rs(path: Path | None) -> str:
    if path is None:
        with urllib.request.urlopen(ALPHABETS_RS_URL, timeout=60) as r:
            data = r.read()
        source = ALPHABETS_RS_URL
    else:
        data = path.read_bytes()
        source = str(path)
    digest = hashlib.sha256(data).hexdigest()
    if digest != ALPHABETS_RS_SHA256:
        raise SystemExit(f"{source}: SHA-256 {digest} is not the {KMERSEEK_TAG} file")
    return data.decode()


def parse_hp_alphabets(text: str) -> dict[str, dict[str, str]]:
    """{moltype: {residue: class}} for every alphabet whose moltype starts with hp_.

    Follows the chain in alphabets.rs: ``Self::X => "moltype"`` (to_moltype),
    ``Self::X => &STATIC`` (partition), ``static STATIC ... build_hp(b"..", b"..")``.
    """
    moltype = dict(re.findall(r'Self::(\w+)\s*=>\s*"(hp_\w+)"', text))
    table = dict(re.findall(r"Self::(\w+)\s*=>\s*&([A-Z0-9_]+)", text))
    builds = {
        name: (fn, re.findall(r'b"([A-Z]*)"', args))
        for name, fn, args in re.findall(
            r"static\s+([A-Z0-9_]+)\s*:\s*LazyLock<HashMap<u8,\s*u8>>\s*=\s*"
            r"LazyLock::new\(\|\|\s*(build_hpc?)\(([^)]*)\)\);",
            text,
        )
    }
    alphabets = {}
    for variant, name in moltype.items():
        fn, groups = builds[table[variant]]
        letters = "hpc" if fn == "build_hpc" else "hp"
        assert len(groups) == len(letters), (name, fn, groups)
        classes = {r: c for c, group in zip(letters, groups) for r in group}
        assert sorted(classes) == sorted("ACDEFGHIKLMNPQRSTVWY"), (name, classes)
        assert sum(map(len, groups)) == 20, name
        alphabets[name] = classes
    return alphabets


def check(alphabets: dict[str, dict[str, str]]) -> None:
    assert len(alphabets) == 7, sorted(alphabets)
    for name, classes in alphabets.items():
        for r in ALWAYS_H:
            assert classes[r] == "h", (name, r)
        for r in ALWAYS_P:
            assert classes[r] == "p", (name, r)
    for r, n in EXPECTED_N_IN_H.items():
        got = sum(c[r] == "h" for c in alphabets.values())
        assert got == n, (r, got, n)
    c_classes = sorted(c["C"] for c in alphabets.values())
    assert c_classes == sorted("hhhhppc"), c_classes
    (only_c,) = [n for n, c in alphabets.items() if c["C"] == "c"]
    assert only_c == "hp_lehninger_hpc3", only_c
    assert all(
        set(c.values()) <= {"h", "p"} for n, c in alphabets.items() if n != only_c
    )


def row_order(alphabets: dict[str, dict[str, str]]) -> list[str]:
    """Most borderline residues in h first; ties keep the order alphabets.rs lists them."""
    source_order = list(alphabets)
    return sorted(
        alphabets,
        key=lambda n: (
            -sum(alphabets[n][r] == "h" for r in BORDERLINE),
            source_order.index(n),
        ),
    )


def write_table(alphabets, order, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "row",
                "alphabet",
                "residue",
                "residue_group",
                "class",
                "class_name",
                "n_borderline_in_h",
                "n_residues_in_h",
                "source",
            ]
        )
        for i, name in enumerate(order, start=1):
            classes = alphabets[name]
            n_border = sum(classes[r] == "h" for r in BORDERLINE)
            n_h = sum(c == "h" for c in classes.values())
            for residues, group in GROUPS:
                for r in residues:
                    c = classes[r]
                    w.writerow(
                        [i, name, r, group, c, CLASS_NAME[c], n_border, n_h,
                         f"seanome/kmerseek {KMERSEEK_TAG} ({KMERSEEK_COMMIT[:7]}) "
                         "src/rust/alphabets.rs"]
                    )  # fmt: skip


# ---- layout, in mm on a figure exactly 89 mm wide ------------------------------------
W = pf.ONE_COLUMN_MM
NAME_RIGHT = 30.6  # alphabet names end here
X0 = 31.8  # first square
PITCH = 2.30
SQ = 2.02  # square side
GAP = 1.0  # between residue groups
COUNT_X = 84.4  # centre of the count column
ROW = 2.55  # row pitch
SQ_PT = 6  # class letter inside a square (Arial: Courier New is too thin on colour)
NAME_PT = 6
RES_PT = 6.5
LABEL_PT = 6


def col_x(j: int) -> float:
    """Left edge of residue column j (0-19), with a gap between the three groups."""
    group = (j >= len(ALWAYS_H)) + (j >= len(ALWAYS_H) + len(BORDERLINE))
    return X0 + j * PITCH + group * GAP


def square(ax, x, y, cls):
    ax.add_patch(
        Rectangle((x, y - SQ / 2), SQ, SQ, facecolor=FILL[cls], edgecolor="none")
    )
    ax.text(x + SQ / 2, y, cls, fontsize=SQ_PT, color=INK[cls],
            ha="center", va="center_baseline")  # fmt: skip


def draw(alphabets, order):
    residues = ALWAYS_H + BORDERLINE + ALWAYS_P
    n = len(order)
    legend_y = 2.0
    group_y = legend_y + 5.2
    res_y = group_y + 3.2
    first_row_y = res_y + 2.9
    height = first_row_y + (n - 1) * ROW + SQ / 2 + 0.6

    fig = plt.figure(figsize=(W * pf.MM, height * pf.MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)  # y grows downward, in mm
    ax.set_axis_off()

    # Legend, above the grid.
    x = 0.4
    for cls, label in [("h", "hydrophobic"), ("p", "polar"),
                       ("c", "cysteine in its own class")]:  # fmt: skip
        square(ax, x, legend_y, cls)
        t = ax.text(
            x + SQ + 0.8, legend_y, label, fontsize=LABEL_PT, va="center_baseline"
        )
        x = x + SQ + 0.8 + text_width_mm(fig, t) + 3.0

    # Group labels with a bracket line, then residue letters.
    j = 0
    for group_res, label in GROUPS:
        left, right = col_x(j), col_x(j + len(group_res) - 1) + SQ
        ax.text((left + right) / 2, group_y, label, fontsize=LABEL_PT, ha="center",
                va="baseline")  # fmt: skip
        ax.plot([left, right], [group_y + 0.8] * 2, color="black", lw=0.5,
                solid_capstyle="butt")  # fmt: skip
        j += len(group_res)
    for j, r in enumerate(residues):
        ax.text(col_x(j) + SQ / 2, res_y, r, family="monospace", fontsize=RES_PT,
                ha="center", va="baseline")  # fmt: skip

    ax.text(COUNT_X, res_y, "residues\nin h\n(of 20)", fontsize=LABEL_PT, ha="center",
            va="baseline", linespacing=1.1)  # fmt: skip

    for i, name in enumerate(order):
        y = first_row_y + i * ROW
        ax.text(NAME_RIGHT, y, name, family="monospace", fontsize=NAME_PT, ha="right",
                va="center_baseline")  # fmt: skip
        for j, r in enumerate(residues):
            square(ax, col_x(j), y, alphabets[name][r])
        n_h = sum(c == "h" for c in alphabets[name].values())
        ax.text(
            COUNT_X, y, str(n_h), fontsize=LABEL_PT, ha="center", va="center_baseline"
        )
    return fig, ax


def text_width_mm(fig, t) -> float:
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
    return bb.width / fig.dpi / pf.MM


def check_layout(fig, ax) -> None:
    """Every text and square stays on the canvas; no two text boxes overlap; no text
    overlaps a square other than the one it sits in."""
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts]
    patches = [p.get_window_extent(r) for p in ax.patches]
    lines = [ln.get_window_extent(r) for ln in ax.lines]
    for s, bb in texts + [("square", b) for b in patches]:
        assert (
            fb.x0 <= bb.x0 and bb.x1 <= fb.x1 and fb.y0 <= bb.y0 and bb.y1 <= fb.y1
        ), f"{s!r} leaves the canvas: {bb} vs {fb}"
    for i, (s1, b1) in enumerate(texts):
        for s2, b2 in texts[i + 1 :]:
            assert not b1.overlaps(b2), f"text {s1!r} overlaps text {s2!r}"
        inside = [p for p in patches if p.contains(*b1.get_points().mean(0))]
        for p in patches:
            if p not in inside:
                assert not b1.overlaps(p), f"text {s1!r} overlaps a square"
        for ln in lines:
            assert not b1.overlaps(ln), f"text {s1!r} overlaps a group line"
    print(
        f"layout: {len(texts)} texts, {len(patches)} squares, no overlaps, all on canvas"
    )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alphabets-rs", type=Path, default=None)
    p.add_argument(
        "--out", type=Path, default=REPO / "figures" / "hp_borderline_residues"
    )
    p.add_argument(
        "--table", type=Path, default=REPO / "tables" / "hp_borderline_residues.csv"
    )
    args = p.parse_args(argv)

    alphabets = parse_hp_alphabets(read_alphabets_rs(args.alphabets_rs))
    check(alphabets)
    order = row_order(alphabets)

    print(f"{'alphabet':26s} {' '.join(BORDERLINE)}  borderline_in_h  in_h")
    for name in order:
        c = alphabets[name]
        print(
            f"{name:26s} {' '.join(c[r] for r in BORDERLINE)}  "
            f"{sum(c[r] == 'h' for r in BORDERLINE):15d}  {sum(v == 'h' for v in c.values()):4d}"
        )
    print("alphabets with residue in h:",
          {r: sum(c[r] == "h" for c in alphabets.values()) for r in BORDERLINE})  # fmt: skip

    write_table(alphabets, order, args.table)

    pf.use_style()
    mpl.rcParams["savefig.bbox"] = None  # keep the figure exactly 89 mm wide
    fig, ax = draw(alphabets, order)
    fig.canvas.draw()
    check_layout(fig, ax)
    pf.save(fig, args.out)
    for ext in ("pdf", "svg", "png"):
        print("wrote", args.out.with_suffix(f".{ext}"))
    before, after = shrink_png(args.out.with_suffix(".png"), check=False)
    print(f"png: {before} -> {after} bytes (256 colours)")
    print("wrote", args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
