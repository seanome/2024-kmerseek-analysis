#!/usr/bin/env python3
"""Build the reviewed Swiss-Prot reference with the query's own clade removed.

The proteome-annotate workflow asks whether a family label transfers onto a proteome that
has no curated features. If the reference still contains the query's own class, the easy
half of that question is answered by near-identical relatives and the hard half never gets
measured. Removing the clade forces every transfer to cross at least a class boundary.

Exclusion is by lineage, read off the OC lines of the Swiss-Prot flat file. That is exact
and self-contained: no taxonomy service, no id mapping, and a clade name that never matches
is an error rather than a silent no-op that leaves the clade in.

Three products from one pass over the 648 MB file:

  <out>.fasta               the reference sequences, for the sequence arms and kmerseek
  <out>_pfam.parquet        accession -> Pfam families, from the DR Pfam lines. This is
                            what a structural hit against a reference protein transfers.
  <out>_accessions.txt      one accession per line, for fetching the reference structures

Reviewed entries only. TrEMBL is unreviewed by definition and the point of the reference is
that it is curated.
"""

import argparse
import gzip
import sys
from pathlib import Path

import polars as pl


def stream_records(dat_path: Path):
    """Yield one dict per Swiss-Prot record.

    Same streaming shape as build_swissprot_truth.py -- accumulate until '//', never hold
    the file. Collects the fields this product needs (lineage, Pfam cross-refs, sequence),
    which are not the fields that one collects.
    """
    acc = None
    lineage: list[str] = []
    pfam: list[str] = []
    seq_lines: list[str] = []
    in_seq = False
    reviewed = False

    with gzip.open(dat_path, "rt") as fh:
        for line in fh:
            if line.startswith("ID "):
                reviewed = "Reviewed;" in line
            elif line.startswith("AC ") and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif line.startswith("OC "):
                lineage.extend(t.strip().rstrip(".") for t in line[5:].split(";") if t.strip())
            elif line.startswith("DR   Pfam;"):
                parts = [p.strip() for p in line[5:].split(";")]
                if len(parts) > 1:
                    pfam.append(parts[1])
            elif line.startswith("SQ "):
                in_seq = True
            elif in_seq and line.startswith("     "):
                seq_lines.append(line.strip().replace(" ", ""))
            elif line.startswith("//"):
                if acc and reviewed:
                    yield {"accession": acc, "lineage": lineage,
                           "pfam": pfam, "sequence": "".join(seq_lines)}
                acc, lineage, pfam, seq_lines, in_seq, reviewed = None, [], [], [], False, False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--swissprot-dat", type=Path, required=True)
    ap.add_argument("--exclude-clade", required=True,
                    help="clade name as it appears on an OC line, e.g. Bivalvia")
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.swissprot_dat.is_file():
        sys.exit(f"no Swiss-Prot flat file at {args.swissprot_dat}. "
                 f"Download uniprot_sprot.dat.gz before building the reference.")

    clade = args.exclude_clade.strip()
    fasta_path = args.out_prefix.with_suffix(".fasta")
    acc_path = Path(str(args.out_prefix) + "_accessions.txt")

    kept = excluded = 0
    pfam_rows: list[dict] = []
    with open(fasta_path, "w") as fa, open(acc_path, "w") as accfh:
        for rec in stream_records(args.swissprot_dat):
            if clade in rec["lineage"]:
                excluded += 1
                continue
            kept += 1
            fa.write(f">{rec['accession']}\n")
            for i in range(0, len(rec["sequence"]), 60):
                fa.write(rec["sequence"][i:i + 60] + "\n")
            accfh.write(rec["accession"] + "\n")
            for fam in rec["pfam"]:
                pfam_rows.append({"accession": rec["accession"], "pfam_id": fam})

    # A clade name that matched nothing means the reference was never actually reduced, and
    # every downstream number would be a within-clade transfer wearing a leave-clade-out
    # label. Fail rather than ship that.
    if excluded == 0:
        sys.exit(
            f"'{clade}' matched no OC lineage in {args.swissprot_dat.name}, so nothing was "
            f"excluded and this is not a clade-excluded reference.\n"
            f"Check the spelling against a lineage line; the name must be the clade as "
            f"Swiss-Prot writes it (e.g. 'Bivalvia', 'Ascidiacea')."
        )

    pl.DataFrame(pfam_rows, schema={"accession": pl.Utf8, "pfam_id": pl.Utf8}).write_parquet(
        Path(str(args.out_prefix) + "_pfam.parquet"), compression="zstd")

    print(f"[reference] excluded={excluded} ({clade}) kept={kept} "
          f"pfam_annotations={len(pfam_rows)}", file=sys.stderr)

    if args.summary_out:
        import json
        args.summary_out.write_text(json.dumps({
            "excluded_clade": clade,
            "entries_excluded": excluded,
            "entries_kept": kept,
            "pfam_annotations": len(pfam_rows),
        }, indent=2))


if __name__ == "__main__":
    main()
