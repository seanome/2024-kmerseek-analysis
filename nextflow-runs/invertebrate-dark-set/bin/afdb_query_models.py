#!/usr/bin/env python3
"""Collect the AlphaFold DB models of one query proteome, and say which proteins have one.

    afdb_query_models.py --proteome <species>.query.fasta --out-dir models \
        --coverage coverage.tsv --summary summary.json \
        [--archive UP..._v6.tar ...] [--extra-dir <dir of AF-*-model_v*.cif[.gz]>]

Two sources, read in this order:
  --archive    AlphaFold DB's proteome tar files (one per model organism). Only the mmCIF
               members of proteins in the proteome are written out.
  --extra-dir  models fetched one accession at a time (tools/fetch-afdb-models), for a
               species with no proteome archive (Borrelia burgdorferi B31) and for proteins
               the archive lacks.

The model version is read from the file names AlphaFold DB serves (AF-<acc>-F<n>-model_v<k>),
never assumed: on 2026-10-16 the server serves v6, not v4. When a protein has models of two
versions, only the highest is kept, with all of its fragments. AlphaFold DB splits a protein
over 2_700 residues into 1_400-residue fragments F1, F2, ... on a 200-residue stride; every
fragment is kept and the search step moves each hit back to protein coordinates.

coverage.tsv has one row per proteome protein: accession, length, has_model (0/1),
n_fragments, model_version, source (archive file name, "extra" or "none"). A protein with
has_model 0 is "not run" for Foldseek, never a protein Foldseek searched and failed on.

Python 3.6 standard library only.
"""
import argparse
import json
import os
import re
import shutil
import sys
import tarfile

MODEL = re.compile(r"^AF-([A-Za-z0-9]+)-F(\d+)-model_v(\d+)\.cif(\.gz)?$")


def read_lengths(path):
    lengths, name, n = {}, None, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if name is not None:
                    lengths[name] = n
                name, n = line[1:].split()[0], 0
            else:
                n += len(line.strip())
    if name is not None:
        lengths[name] = n
    return lengths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proteome", required=True)
    ap.add_argument("--archive", action="append", default=[])
    ap.add_argument("--extra-dir", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--species", default="")
    args = ap.parse_args()

    lengths = read_lengths(args.proteome)
    os.makedirs(args.out_dir, exist_ok=True)
    # accession -> {version: {fragment: (file name, source)}}
    found = {}

    for arch in args.archive:
        n_written = 0
        with tarfile.open(arch, mode="r|*") as tf:
            for member in tf:
                m = MODEL.match(os.path.basename(member.name))
                if not m or not member.isfile() or m.group(1) not in lengths:
                    continue
                acc, frag, ver = m.group(1), int(m.group(2)), int(m.group(3))
                name = os.path.basename(member.name)
                src = tf.extractfile(member)
                with open(os.path.join(args.out_dir, name), "wb") as out:
                    shutil.copyfileobj(src, out)
                found.setdefault(acc, {}).setdefault(ver, {})[frag] = (name, os.path.basename(arch))
                n_written += 1
        print("%s: %d model files of proteome proteins" % (os.path.basename(arch), n_written))

    if args.extra_dir and os.path.isdir(args.extra_dir):
        n_extra = 0
        for name in sorted(os.listdir(args.extra_dir)):
            m = MODEL.match(name)
            if not m or m.group(1) not in lengths or m.group(1) in found:
                continue
            acc, frag, ver = m.group(1), int(m.group(2)), int(m.group(3))
            shutil.copyfile(os.path.join(args.extra_dir, name), os.path.join(args.out_dir, name))
            found.setdefault(acc, {}).setdefault(ver, {})[frag] = (name, "extra")
            n_extra += 1
        print("%s: %d model files" % (args.extra_dir, n_extra))

    # Keep the highest version of each protein; remove the files of any lower one.
    versions = {}
    rows = []
    for acc in sorted(lengths):
        by_ver = found.get(acc)
        if not by_ver:
            rows.append((acc, lengths[acc], 0, 0, "", "none"))
            continue
        top = max(by_ver)
        for ver, frags in by_ver.items():
            if ver != top:
                for name, _src in frags.values():
                    os.remove(os.path.join(args.out_dir, name))
        frags = by_ver[top]
        if 1 not in frags:
            sys.exit("%s: model v%d has fragments %s but no F1" % (acc, top, sorted(frags)))
        versions["v%d" % top] = versions.get("v%d" % top, 0) + 1
        sources = sorted(set(s for _n, s in frags.values()))
        rows.append((acc, lengths[acc], 1, len(frags), "v%d" % top, ",".join(sources)))

    with open(args.coverage, "w") as fh:
        fh.write("accession\tlength\thas_model\tn_fragments\tmodel_version\tsource\n")
        for r in rows:
            fh.write("%s\t%d\t%d\t%d\t%s\t%s\n" % r)
    n_with = sum(r[2] for r in rows)
    summary = {
        "species": args.species,
        "n_queries": len(rows),
        "n_with_model": n_with,
        "share_with_model": (n_with / len(rows)) if rows else None,
        "model_versions": versions,
        "archives": [os.path.basename(a) for a in args.archive],
        "extra_dir": args.extra_dir,
        "n_model_files": len(os.listdir(args.out_dir)),
    }
    with open(args.summary, "w") as fh:
        json.dump(summary, fh, indent=2)
    print("%s: %d of %d proteins have an AlphaFold DB model (%s)"
          % (args.species, n_with, len(rows), versions))


if __name__ == "__main__":
    main()
