"""Command line: geo2kmers fetch, sequences and sketch."""

import argparse
import http.client
import statistics
import sys
from pathlib import Path

from geo2kmers import geo, landing, schema, sequences, sketch


def run_fetch(args):
    provenance = geo.fetch(args.gse, args.outdir, args.gpl, args.sample_tables)
    print(f"wrote {len(provenance['files'])} files and provenance.json to {args.outdir}")


def run_sequences(args):
    out = Path(args.out)
    table = args.table or out.with_name("probe_sequence_source.tsv")
    cache = args.cache or out.with_name("genbank_cache.fasta")
    counts = sequences.make_probe_fasta(args.gpl_table, out, table, cache, max_genbank_nt=args.max_genbank_nt)
    print(f"probes by source: {counts}")
    lengths = [len(s) for s in sketch.read_fasta(out).values()]
    if lengths:
        median = statistics.median(lengths)
        rule = sketch.platform_rule(median)
        residues = sketch.residues_per_frame(int(median))
        print(f"median probe length {median:g} nt, {residues} residues per frame: rule {rule}")


def run_proteome_index(args):
    picker = sketch.ProteomeFrames(args.proteome, args.k, args.alphabet)
    picker.save(args.out)
    print(f"{len(picker.known)} distinct k-mers in {', '.join(picker.sources)}")


def run_land(args):
    probes = sketch.read_fasta(args.fasta)
    rule, k, rows, touched = landing.land_probes(probes, args.proteome, args.k)
    landed = landing.passing_landings(rule, rows, lambda probe: [probe.rsplit("/", 1)[0]])
    with open(args.out, "w") as f:
        f.write("feature\tprotein\tsetting\n")
        for name in sorted(landed):
            f.writelines(f"{name}\t{p}\t{schema.setting_name('protein', k)}\n" for p in sorted(landed[name]))
    print(f"rule {rule}, k {k}: {touched} of {len(probes)} probes have a k-mer in the proteome")


def run_sketch(args):
    values, detection, source = sketch.load_intensities(args.intensities, args.sample, args.channel, args.column)
    probes = sketch.match_names(sketch.read_fasta(args.fasta, whole_name=True), values)
    print(f"intensities come from {source}")
    table, report = sketch.build_sketch(probes, values, args.k, args.alphabet, args.mode, detection, args.proteome)
    report["intensity_source"] = source
    sketch.write_sketch(table, report, args.out, args.sample, args.channel)
    print(f"rule: {report['rule']}")
    print(f"probes used: {report['probes_used']}, distinct k-mers: {report['distinct_kmers']}")
    if "frame_rule" in report:
        print(f"frame rule: {report['frame_rule']}")
    if "probes_with_no_stretch" in report:
        print(f"probes with no stretch: {report['probes_with_no_stretch']}")
    if not report["signature_written"]:
        print("sourmash is not installed, so no signature was written")


def parser():
    p = argparse.ArgumentParser(prog="geo2kmers", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download a GEO series and record where it came from")
    f.add_argument("gse")
    f.add_argument("--outdir", required=True)
    f.add_argument("--gpl", help="platform to use, needed when a series has several")
    f.add_argument("--sample-tables", action="store_true", help="also download every GSM table (holds detection calls)")
    f.set_defaults(run=run_fetch)

    s = sub.add_parser("sequences", help="get one sequence per probe")
    s.add_argument("gpl_table")
    s.add_argument("--out", required=True)
    s.add_argument("--table", help="default: probe_sequence_source.tsv beside --out")
    s.add_argument("--cache", help="default: genbank_cache.fasta beside --out")
    s.add_argument(
        "--max-genbank-nt",
        type=int,
        default=sequences.MAX_GENBANK_NT,
        help="a GenBank record longer than this is not used as a probe (default: %(default)s)",
    )
    s.set_defaults(run=run_sequences)

    k = sub.add_parser("sketch", help="make one sample's k-mer table")
    k.add_argument("--fasta", required=True)
    k.add_argument("--intensities", required=True, help="series matrix, GSM table, or folder of GSM tables")
    k.add_argument("--sample", required=True)
    k.add_argument("--k", type=int, required=True)
    k.add_argument("--alphabet", choices=["dna", "protein", "dayhoff", "hp"], required=True)
    k.add_argument("--mode", choices=["present", "weighted"], required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--channel", choices=["ch1", "ch2"], help="needed for two-colour arrays")
    k.add_argument("--column", help="name the intensity column of a GSM table yourself")
    k.add_argument(
        "--proteome",
        nargs="+",
        help="protein FASTA files, gzipped or not, or one .npz from proteome-index: pick each probe's frame (not dna)",
    )
    k.set_defaults(run=run_sketch)

    n = sub.add_parser("land", help="find the protein each probe, or on 25-letter arrays each probe set, lands on")
    n.add_argument("--fasta", required=True, help="on 25-letter arrays, name each probe <probe set>/<number>")
    n.add_argument("--proteome", nargs="+", required=True, help="protein FASTA files, gzipped or not")
    n.add_argument("--k", type=int, help="default: the k of the platform's rule")
    n.add_argument("--out", required=True)
    n.set_defaults(run=run_land)

    x = sub.add_parser("proteome-index", help="save a proteome's k-mers once, for many sketches to read")
    x.add_argument("--proteome", nargs="+", required=True, help="protein FASTA files, gzipped or not")
    x.add_argument("--k", type=int, required=True)
    x.add_argument("--alphabet", choices=["protein", "dayhoff", "hp"], required=True)
    x.add_argument("--out", required=True, help="a .npz file")
    x.set_defaults(run=run_proteome_index)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        args.run(args)
    except (geo.Geo2KmersError, OSError, http.client.HTTPException) as e:
        print(f"geo2kmers: {e}", file=sys.stderr)
        return 1
    return 0
