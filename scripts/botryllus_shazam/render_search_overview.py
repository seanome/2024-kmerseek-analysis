"""Compact, folded SVG overview; detailed nf-metro maps remain separate."""

from pathlib import Path
import html
import textwrap

out = (
    Path(__file__).resolve().parents[2]
    / "reports/botryllus-shazam/metro/search-collections.svg"
)
parts = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="940" viewBox="0 0 1200 940" role="img"><title>Swiss-Prot protein search and public SRA occurrence search</title><rect width="1200" height="940" fill="white"/><style>text{font-family:Arial,sans-serif;fill:#1c211f}.route{fill:none;stroke-width:4;stroke-linecap:round;stroke-linejoin:round}.sra{stroke:#D55E00;stroke-dasharray:12 9}.protein{stroke:#0072B2}.file{fill:white;stroke:#1c211f;stroke-width:1.4}</style>'
]


def text(x, y, label, size=15, bold=False, anchor="middle", width=25):
    lines = textwrap.wrap(
        label, width, break_long_words=False, break_on_hyphens=False
    ) or [""]
    for i, line in enumerate(lines):
        parts.append(
            f'<text x="{x}" y="{y+i*(size+3)}" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{anchor}">{html.escape(line)}</text>'
        )


def path(d, kind):
    parts.append(f'<path d="{d}" class="route {kind}"/>')


def station(x, y, label, kind, above=True):
    parts.append(
        f'<circle cx="{x}" cy="{y}" r="7" fill="white" stroke="{ "#D55E00" if kind=="sra" else "#0072B2"}" stroke-width="3"/>'
    )
    text(x, y - 70 if above else y + 30, label, 16, True, width=23)


def file(x, y, format, label):
    parts.append(
        f'<path d="M{x-19},{y-23} h25 l13,13 v33 h-38 Z M{x+6},{y-23} v13 h13" class="file"/>'
    )
    text(x, y + 4, format, 9 if len(format) > 6 else 11, True, width=20)
    text(x, y + 48, label, 14, width=24)


text(
    25, 32, "Swiss-Prot and public SRA are searched separately", 24, True, "start", 100
)
path("M25 65 H65", "protein")
text(80, 70, "Swiss-Prot protein search", 16, False, "start", 50)
path("M480 65 H520", "sra")
text(535, 70, "Public SRA sequence occurrence", 16, False, "start", 50)
text(
    25,
    103,
    "Circle = program / transformation. Document = one file type. Arrows show reading direction.",
    15,
    False,
    "start",
    130,
)
parts.append('<rect x="20" y="128" width="1160" height="488" rx="12" fill="#f7f7f5"/>')
text(
    40,
    158,
    "PUBLIC SRA RNA-seq — 5,776 tunicate runs inventoried; collection indexing/search still pending",
    18,
    True,
    "start",
    120,
)
# The upper row reads left to right; the second row returns right to left.
path(
    "M75 240 H245 H390 Q410 240 410 218 L430 210 M390 240 Q410 240 410 260 L430 300 M449 210 H480 Q500 210 520 240 H610 M449 300 H480 Q500 300 520 240 M610 240 H730 Q750 240 750 218 H800 M730 240 Q750 240 750 300 H800 M819 218 H850 Q875 218 890 240 H1020 M819 300 H850 Q875 300 890 240",
    "sra",
)
file(75, 240, "TSV.gz", "SRA / ENA run inventory")
station(245, 240, "curl + MD5 + Rust FASTQ validation", "sra", False)
file(430, 210, "FASTQ.gz", "Raw read 1")
file(430, 300, "FASTQ.gz", "Raw read 2")
station(610, 240, "Trim Galore", "sra", False)
file(800, 210, "FQ.gz", "Trimmed read 1")
file(800, 300, "FQ.gz", "Trimmed read 2")
station(1020, 240, "Sourmash DNA and six-frame translation", "sra", False)
# Fold to the return row with a clear direction cue on each segment.
path(
    "M1020 240 H1110 Q1140 240 1140 270 V390 Q1140 420 1110 420 H1010 H810 H610 H390 H170",
    "sra",
)
file(1010, 420, "SIG.zip", "DNA31 / protein7 / Dayhoff12 / HP30; scaled 10 and 100")
station(810, 420, "Branchwater index", "sra", False)
file(610, 420, "RocksDB", "Searchable index directory")
station(390, 420, "Branchwater manysearch", "sra", False)
file(170, 420, "CSV", "Run / study occurrence matches")
text(
    610,
    575,
    "Same trimmed runs and queries at both scales. Empty query sketches are unsearchable, not biological absence.",
    15,
    False,
    width=130,
)
# Direction arrows are neutral because route identity is already on the rail.
for x, y, direction in [
    (150, 240, 1),
    (545, 240, 1),
    (930, 240, 1),
    (920, 420, -1),
    (705, 420, -1),
    (500, 420, -1),
    (275, 420, -1),
]:
    points = f"{x-5*direction},{y-5} {x+4*direction},{y} {x-5*direction},{y+5}"
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="#1c211f" stroke-width="1.5"/>'
    )
parts.append('<rect x="20" y="638" width="1160" height="276" rx="12" fill="#f7f7f5"/>')
text(
    40,
    671,
    "SWISS-PROT — 575,748 reference proteins; pilot search execution complete, match validation pending",
    18,
    True,
    "start",
    125,
)
path(
    "M80 740 H145 Q165 740 180 780 H300 M80 825 H145 Q165 825 180 780 M300 780 H535 H760 H1035",
    "protein",
)
file(80, 740, "FASTA.gz", "Swiss-Prot proteins")
file(80, 825, "FASTA", "Peptide queries")
station(300, 780, "Sourmash / Branchwater sketching", "protein")
file(535, 780, "SIG.zip", "Protein7 / Dayhoff12 / HP30; scaled 5")
station(760, 780, "Sourmash / Branchwater multisearch", "protein")
file(1035, 780, "CSV", "Swiss-Prot protein candidate matches")
for x in [215, 420, 655, 900]:
    parts.append(
        f'<polyline points="{x-5},775 {x+4},780 {x-5},785" fill="none" stroke="#1c211f" stroke-width="1.5"/>'
    )
text(
    610,
    923,
    "Botryllus-experiment DNA / peptide FASTA queries feed the SRA search. No SPLASH or DE-kupl across public archive runs.",
    14,
    False,
    width=145,
)
parts.append("</svg>")
out.write_text("\n".join(parts))
