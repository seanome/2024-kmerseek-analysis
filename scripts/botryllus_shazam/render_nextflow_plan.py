"""Readable metro plan: separate document marks for every input/output file type."""

from pathlib import Path
import html
import textwrap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/botryllus-shazam/nextflow-plan"
# A numbered panel is a stage, not a completion claim. Dependencies are explicit.
STAGES = [
    (
        "1. Input and trimming",
        "Sources: 64 PRJNA1219309 libraries; preserve existing trimmed outputs.",
        [
            "Raw mate 1 | FASTQ.gz",
            "Raw mate 2 | FASTQ.gz",
            "Sample design | TSV",
            "Frozen contrasts | JSON",
        ],
        [
            "MD5 + paired FASTQ validation",
            "Trim Galore",
            "Paired FASTQ validation + SHA256",
        ],
        [
            "Trimmed mate 1 | FQ.gz",
            "Trimmed mate 2 | FQ.gz",
            "Trimming report | TXT",
            "Validated reads + checksums | JSON",
        ],
    ),
    (
        "2. SPLASH discovery and sequence context",
        "Depends on validated stage 1. Every adjusted P < 0.05 anchor; no top-300 cap.",
        [
            "Trimmed mate 1 | FQ.gz",
            "Trimmed mate 2 | FQ.gz",
            "Validated read manifest | JSON",
        ],
        [
            "SPLASH: anchor31 / target31",
            "Select adjusted P < 0.05",
            "Compactors: all selected anchors",
        ],
        [
            "All anchor statistics | TSV",
            "Selected anchors | TSV",
            "Anchor-target counts | SATC",
            "Compactors assembly table | TSV",
            "DNA extensions | FASTA",
            "Anchor / extension membership | TSV",
            "Discovery checksums | JSON",
        ],
    ),
    (
        "3. K-mer and gene quantification",
        "Depends on stage 1. Same 64 libraries for both assays. No native condition-only DE test.",
        [
            "Trimmed mate 1 | FQ.gz",
            "Trimmed mate 2 | FQ.gz",
            "Reference transcripts | FASTA",
            "Reference gene annotation | GTF",
        ],
        [
            "DE-kupl counter: canonical k31",
            "Kallisto index + quantification",
            "Strict transcript-to-gene mapping",
        ],
        [
            "Per-sample k-mer counts | TSV.gz",
            "Kallisto index | .idx",
            "Per-sample abundance | TSV",
            "Per-sample abundance | H5",
            "Per-sample run information | JSON",
            "Transcript-to-gene map | TSV",
            "Gene count matrix | TSV.gz",
        ],
    ),
    (
        "4. Adjusted differential expression",
        "Depends on stage 3 + frozen design. Collection balance within genotype; validate model rank.",
        [
            "K-mer count matrix | TSV.gz",
            "Gene count matrix | TSV.gz",
            "Sample metadata | TSV",
            "Frozen contrast design | JSON",
        ],
        [
            "Pool balanced collection strata",
            "computeNF normalization",
            "DESeq2: genotype + treatment",
            "Select strict adjusted P < 0.05",
        ],
        [
            "Pooled counts, each assay | TSV.gz",
            "Normalization factors | TSV",
            "All tested gene results | TSV.gz",
            "All tested k-mer results | TSV.gz",
            "Selected k-mers per contrast | TSV",
            "Design + completion checks | JSON",
        ],
    ),
    (
        "5. Native DE-kupl contigs",
        "Depends on stage 4. Assemble separately by contrast and fold-change direction.",
        ["Selected 31-mers + adjusted tests | TSV"],
        [
            "mergeTags -k 31 -m 15 -n",
            "Check every seed occurs in output",
            "Preserve representative seed ID",
        ],
        [
            "Native assembly table | TSV",
            "Native contig sequences | FASTA",
            "Representative tests + origins | JSON",
            "mergeTags execution log | LOG",
        ],
    ),
    (
        "6. Read-supported extensions",
        "Depends on stages 1 + 4. This supplements native contigs; keep all original DNA and IDs.",
        [
            "Selected canonical seeds | TSV",
            "Trimmed mate 1 | FQ.gz",
            "Trimmed mate 2 | FQ.gz",
        ],
        [
            "Exact 31-mer read recruitment",
            "Retain matching pair once",
            "Compactors context extension",
        ],
        [
            "Recruited mate 1 | FASTQ.gz",
            "Recruited mate 2 | FASTQ.gz",
            "Read / seed / position / strand | TSV.gz",
            "Read recruitment counts | JSON",
            "Compactors assembly table | TSV",
            "Read-supported DNA context | FASTA",
            "Extension / seed membership | JSON",
        ],
    ),
    (
        "7. Translation and query sketches",
        "Depends on stages 2, 5 and 6. Direct 31-mers remain DNA queries; they are too short for 30-aa peptides.",
        [
            "SPLASH DNA context | FASTA",
            "DE-kupl native contigs | FASTA",
            "Read-supported DNA context | FASTA",
            "Direct significant DNA seeds | FASTA",
        ],
        [
            "All six translation frames",
            "Split ONLY at stops (*); retain X",
            "Keep stretches >=30 amino acids",
            "Branchwater manysketch",
        ],
        [
            "All original DNA queries | FASTA",
            "Retained peptide stretches | FAA",
            "Strand / frame / DNA origins | JSON",
            "DNA31 query sketches | .sig.zip",
            "Protein7 query sketches | .sig.zip",
            "Dayhoff12 query sketches | .sig.zip",
            "HP30 query sketches | .sig.zip",
            "Searchable / empty query counts | JSON",
        ],
    ),
    (
        "8. Search reference collections",
        "Separate Swiss-Prot proteins from public SRA runs. 5,776 tunicate runs inventoried; only 4 prepared so far.",
        [
            "Swiss-Prot reviewed proteins | FASTA.gz",
            "Swiss-Prot annotations | TSV",
            "SRA / ENA run + study metadata | TSV.gz",
            "Public tunicate mate 1 | FASTQ.gz",
            "Public tunicate mate 2 | FASTQ.gz",
        ],
        [
            "Public-read validation + trimming",
            "Six-frame protocol from stage 7",
            "Branchwater manysketch",
            "Optional Branchwater index",
        ],
        [
            "Protein target sketches | .sig.zip",
            "Dayhoff target sketches | .sig.zip",
            "HP target sketches | .sig.zip",
            "SRA DNA target sketches | .sig.zip",
            "Optional index | RocksDB directory",
            "Run / study / species metadata | JSON",
            "Target checksums + scope | JSON",
        ],
    ),
    (
        "9. Parallel candidate retrieval",
        "Depends on stages 7 + 8. Branchwater manysearch for BOTH collections; no SBT indexes.",
        [
            "Query sketches, separate alphabets | .sig.zip",
            "Target signatures, streaming | .sig.zip",
            "Optional supported index | RocksDB directory",
        ],
        [
            "Branchwater manysearch",
            "Separate source / alphabet / scale",
            "Measure overlap and chance controls",
        ],
        [
            "Swiss-Prot candidates per alphabet | CSV",
            "SRA candidates per alphabet | CSV",
            "Query coverage + empty sketches | JSON",
            "Commands / versions / checksums | JSON",
            "Runtime / memory measurements | TXT",
        ],
    ),
    (
        "10. Gene, domain and paper connections",
        "Depends on stages 2, 4, 5, 6 and 9. Sequence candidates alone do not establish gene function or splicing.",
        [
            "Candidate matches | CSV",
            "Sequences and origin membership | FASTA",
            "Adjusted gene / k-mer tests | TSV.gz",
            "Botryllus gene annotation | GTF",
            "Paper genes and hub membership | TSV",
            "Swiss-Prot gene / species map | TSV",
            "Pfam profile database | HMM",
            "Gene Ontology definitions | OBO",
            "Pfam-to-GO mapping | TXT",
        ],
        [
            "Minimap2 genome / splice mapping",
            "Exact seed-to-transcript mapping",
            "hmmscan: query Pfam domains",
            "Join gene / protein / species metadata",
            "Compare paper genes and hubs",
        ],
        [
            "Alignment evidence | SAM or PAF",
            "Gene connection table | TSV",
            "Sequences + candidate annotations | JSON",
            "Paper recovery and overlaps | TSV",
            "Genotype-adjusted gene effects | TSV",
            "Ambiguities / missing evidence | JSON",
            "Query Pfam domain calls | domtblout",
        ],
    ),
    (
        "11. Results and job-status app",
        "Depends on validated stage 10 and scheduler/output checks. Publish only measured results with timestamps.",
        [
            "Validated gene evidence | TSV",
            "Sequence and annotation evidence | JSON",
            "Scheduler state + completed outputs | JSON",
        ],
        [
            "Build explorer + validate joins",
            "Check links and record denominators",
            "Publish reviewed app build",
        ],
        [
            "Explorer page | HTML",
            "Explorer code | JS",
            "Timestamped status snapshot | JSON",
            "Downloadable sequence records | FASTA",
            "Downloadable evidence tables | TSV",
            "Workflow metro diagram | SVG",
            "Workflow source diagram | MMD",
        ],
    ),
]


def render():
    width = 1040
    heights = [max(len(a), len(c), len(b)) * 43 + 108 for _, _, a, b, c in STAGES]
    height = sum(heights) + 180
    p = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>Botryllus Nextflow migration plan with programs and separate file types</title><desc>Plan only. Eleven panels show dependencies, inputs, programs and outputs. No stage is marked complete.</desc><rect width="1040" height="{height}" fill="white"/><style>text{{font-family:Arial,sans-serif;fill:#152a33}}.rail{{fill:none;stroke:#29434d;stroke-width:3}}.doc{{fill:white;stroke:#75838a;stroke-width:1}}.station{{fill:white;stroke:#29434d;stroke-width:3}}</style>'
    ]

    def txt(x, y, s, size=14, bold=False):
        p.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{700 if bold else 400}">{html.escape(s)}</text>'
        )

    txt(24, 32, "Botryllus: proposed Nextflow pipeline", 25, True)
    txt(
        24,
        58,
        "PLAN ONLY — the Nextflow rewrite has not been implemented or launched.",
        16,
        True,
    )
    txt(
        24,
        83,
        "Circle = program. Document = one file type. Within each panel: input → processing → results.",
        14,
    )
    txt(
        24,
        106,
        "Panels state their dependencies; independent branches can run in parallel. No colour-coded status.",
        14,
    )
    y = 132
    for (title, note, inputs, programs, outputs), h in zip(STAGES, heights):
        p.append(
            f'<rect x="14" y="{y}" width="1012" height="{h-12}" rx="10" fill="#f4f6f8" stroke="#d8dfe2"/>'
        )
        txt(28, y + 26, title, 19, True)
        for j, line in enumerate(textwrap.wrap(note, 132)):
            txt(28, y + 47 + j * 16, line, 12)
        top = y + 80
        txt(35, top, "INPUT", 12, True)
        txt(390, top, "PROCESSING", 12, True)
        txt(715, top, "RESULTS", 12, True)
        for x, items, station in [
            (35, inputs, False),
            (390, programs, True),
            (715, outputs, False),
        ]:
            if station:
                p.append(
                    f'<path fill="none" stroke="#29434d" stroke-width="3" d="M{x+9},{top+22} V{top+22+43*(len(items)-1)}"/>'
                )
            for i, label in enumerate(items):
                yy = top + 22 + i * 43
                if station:
                    p.append(
                        f'<circle fill="white" stroke="#29434d" stroke-width="3" cx="{x+9}" cy="{yy}" r="6"/>'
                    )
                else:
                    p.append(
                        f'<path fill="white" stroke="#75838a" stroke-width="1" d="M{x},{yy-11} h13 l5,5 v18 h-18 Z M{x+13},{yy-11} v5 h5"/>'
                    )
                for k, line in enumerate(
                    textwrap.wrap(label, 39 if x != 390 else 34, break_long_words=False)
                ):
                    txt(x + 26, yy - 1 + k * 15, line, 12)
        # Horizontal connectors communicate only the panel's left-to-right flow.
        for xx in (350, 675):
            p.append(
                f'<path fill="none" stroke="#29434d" stroke-width="3" d="M{xx-16},{top+22} h20 l-5,-4 m5,4 l-5,4"/>'
            )
        y += h
    txt(
        24,
        y + 15,
        "Validation before migration: fixtures → real completed sample → cached outputs → full-scale launch.",
        14,
        True,
    )
    p.append("</svg>")
    OUT.mkdir(exist_ok=True)
    (OUT / "metro-plan.svg").write_text("\n".join(p))


if __name__ == "__main__":
    render()
