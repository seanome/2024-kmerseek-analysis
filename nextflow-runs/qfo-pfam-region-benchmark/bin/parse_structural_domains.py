#!/usr/bin/env python3
"""Parse domain boundaries out of a proteome's AlphaFold models, with per-domain pLDDT.

This is the query-side answer key for the proteome-annotate workflow, and it replaces the
Pfam key rather than supplementing it. Pfam cannot score the claim that workflow tests: an
instance Pfam never annotated is absent from a Pfam key by construction, and that absent
set is exactly the set the claim is about. A structure-derived key has no such hole -- a
domain is in the key because the model has a domain there, whether or not any HMM knows it.

Chainsaw does the parsing (see the chainsaw-domain-parsing skill for its install and its
pydantic-v2 patch). This script drives it over a structure directory and turns its TSV into
the same parquet shape build_domain_truth.py emits, so evaluate_domain_calls.py scores
against it unchanged.

Two things Chainsaw's own output does not carry, both of which the workflow needs:

  pLDDT   per domain, the mean over that domain's residues, read from the B-factor column
          the AFDB writes it into. The QfO run showed kmerseek's accuracy tracks disorder,
          so a key that cannot be cut by confidence averages that signal away.
  segments  Chainsaw domains are frequently discontinuous ('7-39_96-193' is one domain, not
          two). A key storing a single (start, end) per domain silently swallows the gap
          and inflates every IoU measured against it. Both are kept: start/end as the
          envelope for schema compatibility, and the segment list for honest overlap.

Family labels are NOT assigned here -- that is label_domains_foldseek.py, which runs after
this and fills `family`. Everything this script emits carries family='unlabelled_domain'.
"""

import argparse
import gzip
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import polars as pl

CHAINSAW_DIR = Path(os.environ.get("CHAINSAW_DIR", "/Users/olga/code/chainsaw"))
UNLABELLED = "unlabelled_domain"

# The schema build_domain_truth.py writes, so the scorer needs no branch for which key it
# was handed. `pfam_id` keeps its name while holding a structural label: renaming it would
# fork evaluate_domain_calls.py, and the column means "the family this instance is".
TRUTH_COLUMNS = [
    "accession", "pfam_id", "domain_start", "domain_end", "protein_length",
    "domain_plddt", "n_segments", "segments", "chainsaw_confidence",
]


def parse_chopping(chopping: str) -> list[list[tuple[int, int]]]:
    """'7-39_96-193,42-91' -> [[(7,39),(96,193)], [(42,91)]]. One entry per domain."""
    if not chopping or chopping in ("NULL", "None", ""):
        return []
    domains = []
    for dom in chopping.split(","):
        segs = []
        for seg in dom.split("_"):
            start, _, end = seg.partition("-")
            segs.append((int(start), int(end)))
        domains.append(segs)
    return domains


def open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "rt")


def plddt_by_residue(path: Path) -> dict[int, float]:
    """Per-residue pLDDT from an AFDB model, keyed by author residue number.

    AFDB writes pLDDT into the B-factor column, one value per atom and constant across a
    residue's atoms, so taking the first atom seen per residue is exact rather than an
    approximation. Handles both mmCIF (the v6 cache) and PDB.

    Returns author residue numbers. Chainsaw's boundaries are NOT author numbers -- its
    README is explicit that they are consecutive indices from 1 -- so the caller must check
    the two coincide before using one to index the other. They do for AFDB models, whose
    auth numbering runs 1..N with no gaps, and that is asserted rather than assumed.
    """
    out: dict[int, float] = {}
    with open_maybe_gz(path) as fh:
        if path.name.endswith((".cif", ".cif.gz")):
            # Column order in an mmCIF atom_site loop is declared, not fixed, so read the
            # header rather than assuming positions.
            cols: dict[str, int] = {}
            in_loop = False
            for line in fh:
                if line.startswith("_atom_site."):
                    cols[line.strip().split(".", 1)[1]] = len(cols)
                    in_loop = True
                    continue
                if in_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
                    f = line.split()
                    try:
                        resnum = int(f[cols["auth_seq_id"]])
                        bfac = float(f[cols["B_iso_or_equiv"]])
                    except (KeyError, IndexError, ValueError):
                        continue
                    out.setdefault(resnum, bfac)
                elif in_loop and line.startswith("#") and out:
                    break
        else:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        resnum = int(line[22:26])
                        bfac = float(line[60:66])
                    except ValueError:
                        continue
                    out.setdefault(resnum, bfac)
    return out


def run_chainsaw(structure_dir: Path, out_tsv: Path, python: str,
                 models: list[Path] | None = None) -> None:
    """Invoke Chainsaw over a directory of models, from a sandbox of symlinks.

    Chainsaw cannot hand an mmCIF to stride, so it converts each one to PDB and writes the
    result next to the file it was given: feed it a directory of .cif and you get a
    parallel set of .cif.pdb in that same directory. params.structures is the SHARED
    AlphaFold cache that the foldseek arm builds its database from, so writing there would
    put every protein into that database twice, and re-running this would parse each model
    twice from its own leftovers. Neither failure announces itself.

    The sandbox is symlinks, so nothing is copied and the conversions land on scratch.
    """
    script = CHAINSAW_DIR / "get_predictions.py"
    if not script.is_file():
        sys.exit(
            f"Chainsaw not found at {script}.\n"
            f"  git clone --depth 1 https://github.com/JudeWells/chainsaw.git {CHAINSAW_DIR}\n"
            f"  cd {CHAINSAW_DIR}/stride && tar -xzf stride.tgz && make\n"
            f"Then apply the pydantic-v2 patch: chain_id/chopping/time_sec need '= None' "
            f"defaults in src/models/results.py, or every prediction dies on serialisation."
        )
    env = dict(os.environ)
    env.setdefault("STRIDE_EXE", str(CHAINSAW_DIR / "stride" / "stride"))

    with tempfile.TemporaryDirectory(prefix="chainsaw_in_") as tmp:
        sandbox = Path(tmp)
        for m in (models if models is not None else sorted(structure_dir.iterdir())):
            (sandbox / m.name).symlink_to(m.resolve())
        cmd = [python, str(script),
               "--structure_directory", str(sandbox),
               "--output", str(out_tsv)]
        print(f"[chainsaw] {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.run(cmd, cwd=CHAINSAW_DIR, env=env)
    if proc.returncode != 0:
        sys.exit(f"chainsaw failed with exit {proc.returncode}")


def accession_of(chain_id: str) -> str:
    """'AF-A0A023GPK8-F1-model_v6' -> 'A0A023GPK8'.

    A chain_id that is not an AFDB filename is passed through unchanged, so a proteome
    folded locally under its own gene-model ids still keys correctly.
    """
    if chain_id.startswith("AF-") and "-F1" in chain_id:
        return chain_id.split("-")[1]
    return chain_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structures", type=Path, required=True,
                    help="directory of AlphaFold .cif/.pdb models for the query proteome")
    ap.add_argument("--out", type=Path, required=True, help="truth parquet to write")
    ap.add_argument("--chainsaw-tsv", type=Path, default=None,
                    help="reuse an existing chainsaw TSV instead of re-running it")
    ap.add_argument("--species", required=True, help="registry label, for the error message")
    ap.add_argument("--plddt-min", type=float, default=70.0,
                    help="drop domains whose mean pLDDT is below this (default 70)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for chainsaw")
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    models = sorted(
        p for p in args.structures.iterdir()
        if p.name.endswith((".cif", ".pdb", ".cif.gz", ".pdb.gz"))
    )
    # Mirrors what c32772e does for Pfam: an absent input is named, not silently turned
    # into an empty key that reads downstream as "this proteome has no domains".
    if not models:
        sys.exit(
            f"no AlphaFold models for '{args.species}' in {args.structures}.\n"
            f"The structural answer key cannot be built without them, and an empty key is "
            f"indistinguishable from a proteome with no domains.\n"
            f"If this species has no UniProt accessions (Botryllus's FUN* gene models are "
            f"the case in hand), AFDB has nothing to fetch and the proteome must be folded "
            f"before this workflow can score it."
        )

    tsv = args.chainsaw_tsv
    if tsv is None:
        tsv = args.out.with_suffix(".chainsaw.tsv")
        run_chainsaw(args.structures, tsv, args.python, models=models)

    chainsaw = pl.read_csv(tsv, separator="\t")
    by_chain = {p.name.split(".")[0]: p for p in models}

    rows = []
    kept_at = defaultdict(int)
    for rec in chainsaw.iter_rows(named=True):
        chain_id = rec["chain_id"]
        domains = parse_chopping(rec.get("chopping") or "")
        if not domains:
            continue
        model = by_chain.get(chain_id)
        plddt = plddt_by_residue(model) if model else {}
        # Chainsaw indexes residues consecutively from 1; plddt is keyed by author number.
        # On an AFDB model those are the same sequence, and this is where that stops being
        # a silent assumption. A structure with gaps or a non-1 start would otherwise get
        # pLDDT read off the wrong residues and no error anywhere.
        if plddt:
            expected = set(range(1, int(rec["nres"]) + 1))
            if set(plddt) != expected:
                sys.exit(
                    f"{chain_id}: author residue numbering is not 1..{rec['nres']} "
                    f"(got {min(plddt)}..{max(plddt)}, {len(plddt)} residues). Chainsaw "
                    f"boundaries are consecutive indices from 1, so pLDDT cannot be looked "
                    f"up by author number for this model without a mapping."
                )
        for segs in domains:
            residues = [r for s, e in segs for r in range(s, e + 1)]
            vals = [plddt[r] for r in residues if r in plddt]
            mean_plddt = round(sum(vals) / len(vals), 2) if vals else None
            # Count what each candidate floor would keep, so the 50-vs-70 choice is made
            # against real numbers instead of a guess.
            for floor in (50.0, 70.0):
                if mean_plddt is not None and mean_plddt >= floor:
                    kept_at[floor] += 1
            kept_at["all"] += 1
            rows.append({
                "accession": accession_of(chain_id),
                "pfam_id": UNLABELLED,
                "domain_start": min(s for s, _ in segs),
                "domain_end": max(e for _, e in segs),
                "protein_length": int(rec["nres"]),
                "domain_plddt": mean_plddt,
                "n_segments": len(segs),
                "segments": ";".join(f"{s}-{e}" for s, e in segs),
                "chainsaw_confidence": float(rec["confidence"]),
            })

    df = pl.DataFrame(rows, schema_overrides={"domain_plddt": pl.Float64})
    total = df.height
    if args.plddt_min is not None:
        df = df.filter(
            pl.col("domain_plddt").is_not_null() & (pl.col("domain_plddt") >= args.plddt_min)
        )
    df.select(TRUTH_COLUMNS).write_parquet(args.out, compression="zstd")

    print(f"[{args.species}] models={len(models)} proteins_with_domains="
          f"{df['accession'].n_unique()} domains_parsed={total} "
          f"kept_at_50={kept_at[50.0]} kept_at_70={kept_at[70.0]} "
          f"written={df.height} (floor={args.plddt_min})", file=sys.stderr)

    if args.summary_out:
        import json
        args.summary_out.write_text(json.dumps({
            "species": args.species,
            "models": len(models),
            "domains_parsed": total,
            "domains_kept_plddt_50": kept_at[50.0],
            "domains_kept_plddt_70": kept_at[70.0],
            "domains_written": df.height,
            "plddt_floor": args.plddt_min,
            "proteins_with_domains": int(df["accession"].n_unique()),
        }, indent=2))


if __name__ == "__main__":
    main()
