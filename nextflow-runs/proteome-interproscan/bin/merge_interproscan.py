#!/usr/bin/env python3
"""Merge per-chunk InterProScan output into one XML and one parquet.

The XML is what UniFire consumes, so it has to be a single well-formed document rather than
concatenated fragments: each chunk's <protein-matches> children are lifted into one root.

The parquet is the analysis product, and the number the proteome-annotate claim rests on
comes straight out of it -- how many proteins in a non-vertebrate proteome InterProScan
leaves with no signature at all. That is reported here rather than left to be derived,
because it is the denominator everything else is a fraction of.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    tsvs = sorted(args.indir.glob("chunk_*.tsv"))
    xmls = sorted(args.indir.glob("chunk_*.xml"))
    if not tsvs and not xmls:
        raise SystemExit(
            f"no InterProScan output under {args.indir} for '{args.species}'.\n"
            f"Chunks are written as chunk_NNNN.tsv/.xml by `make run-interproscan "
            f"SPECIES={args.species}`. An empty merge would read downstream as a proteome "
            f"with no annotation, which is the one thing this run exists to measure."
        )

    # ---- XML, for UniFire ----
    if xmls:
        first = ET.parse(xmls[0])
        root = first.getroot()
        for x in xmls[1:]:
            for child in ET.parse(x).getroot():
                root.append(child)
        out_xml = Path(str(args.out_prefix) + ".xml")
        ET.ElementTree(root).write(out_xml, encoding="UTF-8", xml_declaration=True)
        n_prot_xml = len(list(root))
    else:
        out_xml, n_prot_xml = None, 0

    # ---- TSV, for analysis ----
    # InterProScan's TSV has no header and a ragged tail: 11 columns always, 13 when
    # -goterms and -pa added theirs, and a line can carry fewer if a field was empty.
    COLS = ["accession", "md5", "length", "analysis", "signature_acc", "signature_desc",
            "start", "end", "score", "status", "date", "interpro_acc", "interpro_desc",
            "go_terms", "pathways"]
    frames = []
    for t in tsvs:
        if t.stat().st_size == 0:
            continue
        df = pl.read_csv(t, separator="\t", has_header=False, truncate_ragged_lines=True,
                         infer_schema_length=0)
        df.columns = COLS[:df.width]
        frames.append(df)
    matches = pl.concat(frames, how="diagonal") if frames else pl.DataFrame(
        {c: [] for c in COLS})
    out_parquet = Path(str(args.out_prefix) + "_matches.parquet")
    matches.write_parquet(out_parquet, compression="zstd")

    n_with_sig = matches["accession"].n_unique() if matches.height else 0
    n_with_ipr = (matches.filter(pl.col("interpro_acc").is_not_null()
                                 & (pl.col("interpro_acc") != "-"))["accession"].n_unique()
                  if matches.height else 0)

    summary = {
        "species": args.species,
        "chunks_tsv": len(tsvs),
        "chunks_xml": len(xmls),
        "proteins_in_xml": n_prot_xml,
        "match_rows": matches.height,
        "proteins_with_any_signature": n_with_sig,
        "proteins_with_interpro_entry": n_with_ipr,
        "analyses": sorted(matches["analysis"].unique().to_list()) if matches.height else [],
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
