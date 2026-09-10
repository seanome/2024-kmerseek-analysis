#!/opt/conda/bin/python3
"""One chunk of ortholog pairs: protein alignment (MAFFT) -> codon back-translation
-> pairwise maximum-likelihood dN, dS, omega (PAML codeml, runmode=-2).

Pairwise ML is the only option here. Site models (M7 vs M8) and branch-site tests
need a multi-taxon phylogeny; with two sequences there is one number to estimate and
no power for anything finer.

The codeml output parser differs from the one in nextflow-runs/human-mouse-dnds-omega,
which reports the wrong dS. That parser searched for r"dS\\s*=\\s*([\\d.eE+-]+)" with
re.search, and the first match of that pattern in codeml's result line falls inside
"dN/dS=", not inside "dS =". Its dS column therefore holds a copy of omega, verified
identical in all 1_335 rows of its results file. The fix is to match the whole result
line in one pattern so each field is anchored by its neighbours:

    t= 0.9041  S=   475.3  N=  1591.7  dN/dS=  0.1440  dN = 0.1273  dS = 0.8842

Checked against a real PAML 4.10.10 .mlc from that pipeline's work directory: this
parser returns dS 0.8842 where the old one returned 0.1440.

Saturation is reported, not silently passed on. Beyond roughly dS > 2 the synonymous
sites have been hit often enough that the count of observed differences no longer
tracks the number of substitutions, and codeml's dS becomes an extrapolation with
enormous variance. It does not error out, it returns a number. `dS_saturated` marks
the rows where that number should not be believed, and omega with it, since omega is
dN/dS.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

CTL_TEMPLATE = """\
seqfile = {phy}
outfile = {mlc}
noisy = 0
verbose = 0
runmode = -2
seqtype = 1
CodonFreq = 2
model = 0
NSsites = 0
icode = {icode}
Mgene = 0
fix_kappa = 0
kappa = 2
fix_omega = 0
omega = 0.5
getSE = 0
RateAncestor = 0
Small_Diff = 0.5e-6
cleandata = 0
"""

# codeml prints t, S, N, dN/dS, dN and dS on one line in a fixed order. Matching the
# line as a whole rather than each label on its own is what keeps "dS" from being
# found inside "dN/dS". Spacing around "=" varies between PAML versions, so every
# separator is \s*.
RESULT_LINE = re.compile(
    r"t\s*=\s*(?P<t>-?[\d.]+)\s+"
    r"S\s*=\s*(?P<S>-?[\d.]+)\s+"
    r"N\s*=\s*(?P<N>-?[\d.]+)\s+"
    r"dN/dS\s*=\s*(?P<omega>-?[\d.]+)\s+"
    r"dN\s*=\s*(?P<dN>-?[\d.]+)\s+"
    r"dS\s*=\s*(?P<dS>-?[\d.]+)"
)
# lnL and kappa come from the block above the result line. Both are optional: they
# are diagnostics, and a missing one should not throw away a usable omega.
LNL = re.compile(r"lnL\s*=\s*(-?[\d.]+)")
PARAMS = re.compile(r"^\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$", re.MULTILINE)


def run(cmd, **kwargs):
    if "stdout" in kwargs:
        kwargs.setdefault("stderr", subprocess.PIPE)
        result = subprocess.run(cmd, text=True, **kwargs)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {result.stderr[-2000:]}")
    return result


def backtranslate(aligned: dict[str, str], cds: dict[str, str]) -> dict[str, str]:
    """Map a gapped protein alignment back onto codons. The CDS dict here has already
    had any terminal stop removed upstream, so every codon corresponds to one residue
    and there is no last-codon special case."""
    out = {}
    for name, gapped in aligned.items():
        codons = [cds[name][i : i + 3] for i in range(0, len(cds[name]), 3)]
        n_residues = len(gapped) - gapped.count("-")
        if n_residues != len(codons):
            raise ValueError(f"{name}: {n_residues} aligned residues but {len(codons)} codons")
        rebuilt, index = [], 0
        for residue in gapped:
            if residue == "-":
                rebuilt.append("---")
            else:
                rebuilt.append(codons[index])
                index += 1
        out[name] = "".join(rebuilt)
    return out


def write_phylip(alignment: dict[str, str], path: Path):
    names = list(alignment)
    with open(path, "w") as handle:
        handle.write(f" {len(names)} {len(alignment[names[0]])}\n")
        for name in names:
            handle.write(f"{name}\n{alignment[name]}\n")


def parse_mlc(text: str) -> dict:
    match = RESULT_LINE.search(text)
    if match is None:
        raise ValueError("no codeml result line (t=/S=/N=/dN/dS=/dN=/dS=) in .mlc")
    parsed = {k: float(v) for k, v in match.groupdict().items()}
    lnl = LNL.search(text)
    parsed["lnL"] = float(lnl.group(1)) if lnl else ""
    # The line under lnL is "t kappa omega"; take kappa from it when it is there.
    params = PARAMS.search(text, lnl.end() if lnl else 0)
    parsed["kappa"] = float(params.group(2)) if params else ""
    return parsed


FIELDS = [
    "pair_id", "species", "mya", "ortholog_source",
    "human_accession", "human_gene", "target_accession", "target_gene",
    "human_cds_exact_match",
    "dN", "dS", "omega", "t", "kappa", "lnL", "N", "S",
    "n_codons", "aln_pid", "dS_saturated", "status", "detail",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk_fasta", required=True, type=Path,
                    help="2 records per pair, named '<pair_id>|query' and '<pair_id>|target'")
    ap.add_argument("--manifest", required=True, type=Path, help="pairs TSV from build_ortholog_pairs.py")
    ap.add_argument("--outfile", required=True, type=Path)
    ap.add_argument("--icode", type=int, default=0,
                    help="PAML genetic code index; 0 = universal, 10 = bacterial/table 11")
    ap.add_argument("--ds_saturated_above", type=float, default=2.0)
    ap.add_argument("--workdir", type=Path, default=Path("codeml_work"))
    args = ap.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    meta = {}
    with open(args.manifest) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            row = dict(zip(header, values))
            meta[row["pair_id"]] = row

    by_pair = {}
    for record in SeqIO.parse(args.chunk_fasta, "fasta"):
        pair_id, side = record.id.rsplit("|", 1)
        by_pair.setdefault(pair_id, {})[side] = str(record.seq).upper()

    with open(args.outfile, "w") as out:
        out.write("\t".join(FIELDS) + "\n")
        for pair_id, sides in by_pair.items():
            info = meta.get(pair_id, {})
            row = {k: "" for k in FIELDS}
            row.update(
                pair_id=pair_id,
                species=info.get("species", ""),
                mya=info.get("mya", ""),
                ortholog_source=info.get("ortholog_source", ""),
                human_accession=info.get("human_accession", ""),
                human_gene=info.get("human_gene", ""),
                target_accession=info.get("target_accession", ""),
                target_gene=info.get("target_gene", ""),
                human_cds_exact_match=info.get("human_cds_exact_match", ""),
            )
            try:
                if set(sides) != {"query", "target"}:
                    raise ValueError(f"expected query+target records, got {sorted(sides)}")

                table = 11 if args.icode == 10 else 1
                protein = {
                    name: str(Seq(seq).translate(table=table))
                    for name, seq in sides.items()
                }

                pair_dir = args.workdir / pair_id
                pair_dir.mkdir(parents=True, exist_ok=True)
                protein_fa = pair_dir / "protein.fa"
                with open(protein_fa, "w") as handle:
                    for name, seq in protein.items():
                        handle.write(f">{name}\n{seq}\n")

                aln_fa = pair_dir / "protein.aln.fa"
                with open(aln_fa, "w") as handle:
                    run(["mafft", "--auto", "--quiet", str(protein_fa)], stdout=handle)

                aligned = {rec.id: str(rec.seq) for rec in SeqIO.parse(aln_fa, "fasta")}
                codon_aln = backtranslate(aligned, sides)

                columns = list(zip(aligned["query"], aligned["target"]))
                ungapped = [(a, b) for a, b in columns if a != "-" and b != "-"]
                pid = sum(a == b for a, b in ungapped) / len(ungapped) if ungapped else 0.0

                write_phylip(codon_aln, pair_dir / "codon.phy")
                (pair_dir / "codeml.ctl").write_text(
                    CTL_TEMPLATE.format(phy="codon.phy", mlc="out.mlc", icode=args.icode)
                )
                run(["codeml", "codeml.ctl"], cwd=str(pair_dir))
                parsed = parse_mlc((pair_dir / "out.mlc").read_text())

                dS = parsed["dS"]
                # codeml signals a dS it could not estimate with a negative value or a
                # pegged ceiling; both mean the same thing as an over-threshold dS.
                saturated = dS <= 0 or dS > args.ds_saturated_above or dS >= 49.0
                row.update(
                    dN=parsed["dN"], dS=dS, omega=parsed["omega"], t=parsed["t"],
                    kappa=parsed["kappa"], lnL=parsed["lnL"], N=parsed["N"], S=parsed["S"],
                    n_codons=len(next(iter(codon_aln.values()))) // 3,
                    aln_pid=round(pid, 4),
                    dS_saturated="true" if saturated else "false",
                    status="ok", detail="",
                )
            except Exception as exc:  # one bad pair must not lose the other 249
                row["status"] = "fail"
                row["detail"] = str(exc).replace("\t", " ").replace("\n", " ")[:300]
                print(f"FAILED {pair_id}: {exc}", file=sys.stderr)
            out.write("\t".join(str(row[k]) for k in FIELDS) + "\n")


if __name__ == "__main__":
    main()
