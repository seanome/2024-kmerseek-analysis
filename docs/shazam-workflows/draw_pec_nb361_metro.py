"""Draw the PEC (notebook 361) metro map by hand, as SVG and PNG.

nf-metro 2.1.0 could not lay out this map once the link from `kamrat` to
`control_truth` was added (2026-11-01), so every line, step and file is placed
here by coordinate.

Render (the PNG needs `pip install resvg-py==0.5.0`):

    python docs/shazam-workflows/draw_pec_nb361_metro.py \
        docs/shazam-workflows/shazam_pec_nb361_metro.svg \
        docs/shazam-workflows/shazam_pec_nb361_metro.png

Check that the map has one step for each process of the pipeline, and no
others (exits non-zero on any difference):

    python docs/shazam-workflows/draw_pec_nb361_metro.py \
        --check-processes nextflow-runs/pec-domen2026/main.nf
"""

import argparse
import math
import re
import sys

# Steps on the map that are processes of nextflow-runs/pec-domen2026/main.nf.
# Steps marked "(script)" or "(planned)" on the map are not listed here.
PROCESSES = [
    "download",
    "fastp",
    "simulate_control",
    "splash",
    "compactors",
    "anchor_counts",
    "compactors_nominal",
    "dekupl",
    "kamrat",
    "search",
    "control_truth",
    "paper_recovery",
]

COLOURS = dict(
    splash="#0570b0",
    counts="#2db572",
    dekupl="#e6550d",
    kamrat="#e7298a",
    protein="#756bb1",
    check="#7f7f7f",
)
W, H = 1200, 540
LINK_GREY = "#999"


def offset(pts, o):
    """Shift a polyline sideways by o pixels (positive = right of travel, y down)."""
    if o == 0:
        return pts
    segs = []
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        length = math.hypot(x2 - x1, y2 - y1)
        nx, ny = -(y2 - y1) / length, (x2 - x1) / length
        segs.append(((x1 + nx * o, y1 + ny * o), (x2 + nx * o, y2 + ny * o)))
    res = [segs[0][0]]
    for ((x1, y1), (x2, y2)), ((x3, y3), (x4, y4)) in zip(segs, segs[1:]):
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9:
            res.append((x2, y2))
            continue
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        res.append((x1 + t * (x2 - x1), y1 + t * (y2 - y1)))
    res.append(segs[-1][1])
    return res


def rounded(pts, r=14):
    """SVG path through pts with each corner rounded by up to r pixels."""
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        l1, l2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        rr = min(r, l1 / 2, l2 / 2)
        ax, ay = x1 - (x1 - x0) / l1 * rr, y1 - (y1 - y0) / l1 * rr
        bx, by = x1 + (x2 - x1) / l2 * rr, y1 + (y2 - y1) / l2 * rr
        d += f" L{ax:.1f},{ay:.1f} Q{x1:.1f},{y1:.1f} {bx:.1f},{by:.1f}"
    d += f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}"
    return d


def build_svg():
    out = []
    lines = (
        []
    )  # (colour name, points); each gets a white edge so crossings read as bridges
    steps = []  # (x, y top, y bottom) of each white step mark
    files = []  # (x, y, file type, description)

    def line(name, pts, o=0):
        lines.append((name, offset(pts, o)))

    def text(x, y, s, size=12, weight="normal", anchor="middle", fill="#222"):
        out.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" fill="{fill}">{s}</text>'
        )

    def label(x, y, name, sub=None, anchor="middle"):
        """Bold step name, with optional grey lines under it ('|' splits lines)."""
        text(x, y, name, 12, "bold", anchor)
        if sub:
            for i, s in enumerate(sub.split("|")):
                text(x, y + 13 * (i + 1), s, 10.5, "normal", anchor, "#555")

    def step(x, y1, y2):
        steps.append((x, y1, y2))

    def icon(x, y, kind, desc, link):
        if link:
            out.append(
                f'<line x1="{x}" y1="{y}" x2="{link[0]}" y2="{link[1]}" '
                f'stroke="{LINK_GREY}" stroke-width="1.3"/>'
            )
        files.append((x, y, kind, desc))

    # ---------------- lines ----------------
    # Reads: real runs (top) and the simulated control (bottom) join at fastp.
    o4 = dict(splash=-10.5, counts=-3.5, dekupl=3.5, kamrat=10.5)
    for n, o in o4.items():
        line(n, [(140, 150), (185, 150), (220, 225), (250, 225)], o)
        line(n, [(140, 300), (185, 300), (220, 225), (250, 225)], o)
    # fastp -> splash (top pair) and fastp -> dekupl (bottom pair)
    for n, o in (("splash", -3.5), ("counts", 3.5)):
        line(n, [(250, 218), (290, 218), (330, 140), (390, 140)], o)
    for n, o in (("dekupl", -3.5), ("kamrat", 3.5)):
        line(n, [(250, 232), (290, 232), (330, 360), (390, 360)], o)

    # after splash
    line(
        "splash",
        [
            (390, 136.5),
            (440, 136.5),
            (465, 105),
            (600, 105),
            (630, 135.5),
            (740, 135.5),
        ],
    )  # compactors -> search
    line(
        "splash", [(390, 136.5), (520, 136.5), (520, 319.5), (740, 319.5)]
    )  # -> control_truth
    line(
        "counts",
        [
            (390, 143.5),
            (440, 143.5),
            (465, 175),
            (527, 175),
            (527, 326.5),
            (740, 326.5),
        ],
    )  # anchor_counts -> control_truth
    line(
        "counts",
        [(527, 175), (648, 175), (668, 156.5), (740, 156.5)],
    )  # anchor_counts -> compactors_nominal -> search
    # after dekupl
    line(
        "dekupl", [(390, 356.5), (700, 356.5), (720, 333.5), (740, 333.5)]
    )  # -> control_truth
    line(
        "dekupl", [(390, 356.5), (680, 356.5), (680, 142.5), (740, 142.5)]
    )  # -> search
    line(
        "kamrat",
        [
            (390, 363.5),
            (440, 363.5),
            (465, 400),
            (710, 400),
            (730, 340.5),
            (740, 340.5),
        ],
    )  # kamrat -> control_truth
    line(
        "kamrat",
        [
            (390, 363.5),
            (440, 363.5),
            (465, 400),
            (695, 400),
            (695, 149.5),
            (740, 149.5),
        ],
    )  # kamrat -> search

    # search -> paper_recovery: protein hits of SPLASH and of the anchor-count anchors
    slots = dict(protein=212.5, splash=219.5, counts=226.5, check=233.5)
    for n, y in (("protein", 128.5), ("splash", 135.5), ("counts", 156.5)):
        line(n, [(740, y), (895, y), (940, slots[n]), (1000, slots[n])])
    line("check", [(1000, 380), (1000, 233.5)])  # paper_gene_lists -> paper_recovery

    # ---------------- steps ----------------
    step(140, 139.5, 160.5)  # download
    step(140, 289.5, 310.5)  # simulate_control
    step(250, 214.5, 235.5)  # fastp
    step(390, 136.5, 143.5)  # splash
    step(390, 356.5, 363.5)  # dekupl
    step(500, 105, 105)  # compactors
    step(500, 175, 175)  # anchor_counts
    step(500, 400, 400)  # kamrat
    step(588, 175, 175)  # compactors_nominal
    step(740, 128.5, 156.5)  # search
    step(740, 319.5, 347.5)  # control_truth
    step(1000, 212.5, 233.5)  # paper_recovery
    step(1000, 380, 380)  # paper_gene_lists

    # ---------------- header, legend, section bands ----------------
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        'font-family="Arial, Helvetica, sans-serif">'
    )
    out.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    text(
        20,
        28,
        "Botryllus electrical stimulation (PEC), notebook 361: Nextflow pipeline "
        "nextflow-runs/pec-domen2026",
        15,
        "bold",
        "start",
    )
    # legend comes before the marks in reading order
    lx, ly = 20, 52
    legend = [
        ("splash", "SPLASH anchors"),
        ("counts", "anchor counts, PyDESeq2"),
        ("dekupl", "DE-kupl k-mers, limma"),
        ("kamrat", "KaMRaT contigs, t-test"),
        ("protein", "protein search hits"),
        ("check", "known answers: control, paper gene lists"),
    ]
    x = lx
    for n, s in legend:
        dash = ' stroke-dasharray="7 4"' if n == "check" else ""
        out.append(
            f'<line x1="{x}" y1="{ly}" x2="{x + 26}" y2="{ly}" stroke="{COLOURS[n]}" stroke-width="5"{dash}/>'
        )
        text(x + 32, ly + 4, s, 11.5, "normal", "start")
        x += 32 + 6.3 * len(s) + 22
    y2 = ly + 22
    out.append(
        f'<rect x="{lx}" y="{y2 - 7}" width="12" height="14" rx="6" fill="#fff" stroke="#222" stroke-width="2"/>'
    )
    text(
        lx + 18,
        y2 + 4,
        "step (white mark across the lines it reads)",
        11.5,
        "normal",
        "start",
    )
    out.append(
        f'<path d="M{lx + 300},{y2 - 8} h9 l4,4 v12 h-13 z" fill="#fff" stroke="#222" stroke-width="1.2"/>'
    )
    text(
        lx + 320,
        y2 + 4,
        "file, with its type and what it holds; thin grey link = read or written by the step",
        11.5,
        "normal",
        "start",
    )

    bands = [
        (20, 280, "1", "Reads"),
        (290, 665, "2", "Find changed sequences, PEC vs control"),
        (675, 885, "3", "Search proteins;|score the control"),
        (890, 1180, "4", "Paper gene lists, notebook"),
    ]
    for x1, x2, num, name in bands:
        out.append(
            f'<rect x="{x1}" y="92" width="{x2 - x1}" height="{H - 122}" rx="8" fill="#f3f3f3"/>'
        )
        out.append(f'<circle cx="{x1 + 14}" cy="108" r="9" fill="#333"/>')
        text(x1 + 14, 112, num, 11, "bold", "middle", "#fff")
        for i, s in enumerate(name.split("|")):
            text(x1 + 28, 112 + 15 * i, s, 12.5, "bold", "start", "#444")
    out.append('<g transform="translate(0,45)">')

    # ---------------- files and their thin links (drawn under the lines) ----------------
    icon(62, 150, "FASTQ.GZ", "47 of 64 runs|78 GB", (140, 150))
    icon(62, 262, "FA", "control|3,100 transcripts", (140, 300))
    icon(62, 330, "TSV", "control|planted changes", (140, 300))
    icon(390, 250, "TSV", "", None)  # sample design, read by splash and dekupl
    out.append(
        f'<line x1="390" y1="148" x2="390" y2="236" stroke="{LINK_GREY}" stroke-width="1.3"/>'
    )
    out.append(
        f'<line x1="390" y1="266" x2="390" y2="352" stroke="{LINK_GREY}" stroke-width="1.3"/>'
    )
    icon(820, 215, "PARQUET", "peptide hits,|alignments", (746, 161))
    icon(820, 106, "ROCKSDB", "", (748, 124))
    text(846, 104, "3 protein", 10, "normal", "start", "#444")
    text(846, 116, "indexes", 10, "normal", "start", "#444")
    icon(800, 410, "JSON", "control recall", (744, 352))
    icon(920, 380, "XLSX", "PNAS Dataset S1", (1000, 380))
    icon(1100, 230, "IPYNB", "notebook 361,|figures, report|(planned)", (1006, 230))

    for n, pts in lines:
        d = rounded(pts)
        out.append(
            f'<path d="{d}" fill="none" stroke="#fff" stroke-width="8.5" '
            'stroke-linecap="round" stroke-linejoin="round"/>'
        )
        dash = ' stroke-dasharray="7 4"' if n == "check" else ""
        out.append(
            f'<path d="{d}" fill="none" stroke="{COLOURS[n]}" stroke-width="4.5" '
            f'stroke-linejoin="round"{dash}/>'
        )

    for x, y1, y2_ in steps:
        out.append(
            f'<rect x="{x - 6.5}" y="{y1 - 6.5}" width="13" height="{y2_ - y1 + 13}" rx="6.5" '
            'fill="#fff" stroke="#222" stroke-width="2"/>'
        )

    for x, y, kind, desc in files:
        w, h = (40 if len(kind) > 6 else 34 if len(kind) > 4 else 26), 32
        out.append(
            f'<path d="M{x - w / 2},{y - h / 2} h{w - 8} l8,8 v{h - 8} h-{w} z" '
            'fill="#fff" stroke="#222" stroke-width="1.3"/>'
        )
        fs = 6.5 if len(kind) > 7 else 7 if len(kind) > 5 else 8.5
        text(x, y + 4, kind, fs, "bold")
        for i, s in enumerate([s for s in desc.split("|") if s]):
            text(x, y + h / 2 + 12 + 12 * i, s, 10, "normal", "middle", "#444")
    text(410, 247, "design:", 10, "normal", "start", "#444")
    text(410, 259, "PEC +1,", 10, "normal", "start", "#444")
    text(410, 271, "control -1", 10, "normal", "start", "#444")

    # ---------------- step labels ----------------
    label(140, 116, "download", "md5, read count")
    label(140, 330, "simulate_control", "12 samples,|1 M read pairs each")
    label(250, 258, "fastp")
    label(390, 120, "splash")
    label(500, 90, "compactors")
    label(493, 192, "anchor_counts", "PyDESeq2", anchor="end")
    label(390, 386, "dekupl", "DE-kupl 1.3.5")
    label(500, 418, "kamrat", "KaMRaT 1.2")
    label(588, 162, "compactors_nominal")
    label(735, 112, "search", anchor="end")
    label(800, 300, "control_truth")
    label(1000, 190, "paper_recovery", "share of each list reached")
    label(1012, 398, "paper_gene_lists", "(script)", anchor="start")
    out.append("</g>")
    text(
        20,
        H - 12,
        "Every step is a process of the pipeline unless marked (script) or (planned). "
        "A line through a step means the step reads that data.",
        11,
        "normal",
        "start",
        "#555",
    )
    out.append("</svg>")
    return "\n".join(out)


def check_processes(main_nf):
    """Compare PROCESSES with the `process` blocks of main_nf. Returns 0 if they match."""
    with open(main_nf) as fh:
        found = [
            m.group(1) for m in (re.match(r"^process\s+(\w+)", ln) for ln in fh) if m
        ]
    missing = sorted(set(found) - set(PROCESSES))
    extra = sorted(set(PROCESSES) - set(found))
    if missing:
        print(f"Processes in {main_nf} with no step on the map: {', '.join(missing)}")
    if extra:
        print(
            f"Steps on the map that are not processes in {main_nf}: {', '.join(extra)}"
        )
    if missing or extra:
        return 1
    print(
        f"OK: {len(found)} of {len(found)} processes in {main_nf} have a step on the map"
    )
    return 0


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("svg", nargs="?", help="SVG file to write")
    p.add_argument(
        "png", nargs="?", help="PNG file to write, at 2x zoom (needs resvg-py)"
    )
    p.add_argument(
        "--check-processes",
        metavar="MAIN_NF",
        help="compare the map's steps with the process blocks of this main.nf",
    )
    args = p.parse_args()
    if args.check_processes:
        status = check_processes(args.check_processes)
        if status or not args.svg:
            return status
    if not args.svg:
        p.error("give an SVG path to write, or --check-processes")
    svg = build_svg()
    with open(args.svg, "w") as fh:
        fh.write(svg + "\n")
    if args.png:
        import resvg_py

        png = resvg_py.svg_to_bytes(svg_path=args.svg, zoom=2)
        with open(args.png, "wb") as fh:
            fh.write(bytes(png))
    return 0


if __name__ == "__main__":
    sys.exit(main())
