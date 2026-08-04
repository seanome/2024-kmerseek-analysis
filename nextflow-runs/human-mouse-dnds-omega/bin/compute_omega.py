#!/usr/bin/env python3
"""Per-gene-pair: CDS -> protein alignment (MAFFT) -> codon back-translation ->
pairwise dN/dS/omega (PAML codeml, runmode=-2). One gene pair per invocation —
Nextflow parallelizes across the gene list by running this once per task.

NOTE: the codeml `.mlc` output parser below has not been validated against a real
PAML run (PAML isn't installed in the environment this was written in — see
notebook 206 section 8 for why). Sanity-check it on the `test` profile's handful
of gene pairs before trusting a full-scale run; PAML's pairwise-output formatting
has drifted across versions before.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError

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
icode = 0
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


def translate_cds(cds: str, gene_label: str) -> str:
    seq = Seq(cds)
    try:
        return str(seq.translate(cds=True, to_stop=False)).rstrip("*")
    except TranslationError as e:
        raise ValueError(f"{gene_label}: not a clean CDS ({e})")


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstdout={r.stdout}\nstderr={r.stderr}")
    return r


def backtranslate(aligned_records: dict[str, str], cds: dict[str, str]) -> dict[str, str]:
    """aligned_records: id -> gapped protein string. cds: id -> ungapped CDS string
    (already validated as a clean, in-frame, single-stop CDS)."""
    out = {}
    for seq_id, prot_aln in aligned_records.items():
        codons = [cds[seq_id][i : i + 3] for i in range(0, len(cds[seq_id]) - 3, 3)]  # drop stop codon
        n_aa = len(prot_aln.replace("-", ""))
        if n_aa != len(codons):
            raise ValueError(f"{seq_id}: protein length {n_aa} != codon count {len(codons)} after back-translation")
        rebuilt, ci = [], 0
        for aa in prot_aln:
            if aa == "-":
                rebuilt.append("---")
            else:
                rebuilt.append(codons[ci])
                ci += 1
        out[seq_id] = "".join(rebuilt)
    return out


def write_phylip_sequential(codon_aln: dict[str, str], path: Path):
    ids = list(codon_aln)
    aln_len = len(codon_aln[ids[0]])
    with open(path, "w") as f:
        f.write(f" {len(ids)} {aln_len}\n")
        for seq_id in ids:
            f.write(f"{seq_id}\n{codon_aln[seq_id]}\n")


def parse_mlc(mlc_text: str) -> dict:
    # codeml pairwise output has one line per pair containing all of these labels;
    # matched independently (not by fixed column position) since spacing varies
    # across PAML versions, e.g. "dN =" vs "dN=".
    patterns = {
        "t": r"t\s*=\s*([\d.eE+-]+)",
        "S": r"\bS\s*=\s*([\d.eE+-]+)",
        "N": r"\bN\s*=\s*([\d.eE+-]+)",
        "omega": r"dN/dS\s*=\s*([\d.eE+-]+)",
        "dN": r"dN\s*=\s*([\d.eE+-]+)",
        "dS": r"dS\s*=\s*([\d.eE+-]+)",
    }
    result = {}
    for key, pat in patterns.items():
        m = re.search(pat, mlc_text)
        if m is None:
            raise ValueError(f"could not find '{key}' in codeml .mlc output")
        result[key] = float(m.group(1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cds_fasta", required=True, help="2-record FASTA, ids 'human' and 'mouse'")
    ap.add_argument("--human_gene", required=True)
    ap.add_argument("--mouse_gene", required=True)
    ap.add_argument("--outfile", required=True, help="one-row TSV output path")
    ap.add_argument("--workdir", default=".")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    fields = ["human_gene", "mouse_gene", "dN", "dS", "omega", "t", "N", "S", "n_codons", "status", "detail"]
    row = {k: "" for k in fields}
    row["human_gene"], row["mouse_gene"] = args.human_gene, args.mouse_gene

    try:
        cds = {rec.id: str(rec.seq).upper() for rec in SeqIO.parse(args.cds_fasta, "fasta")}
        if set(cds) != {"human", "mouse"}:
            raise ValueError(f"expected records 'human'+'mouse', got {sorted(cds)}")

        protein = {seq_id: translate_cds(seq, f"{args.human_gene}/{args.mouse_gene}:{seq_id}") for seq_id, seq in cds.items()}

        protein_fa = workdir / "protein.fa"
        with open(protein_fa, "w") as f:
            for seq_id, seq in protein.items():
                f.write(f">{seq_id}\n{seq}\n")

        aln_fa = workdir / "protein.aln.fa"
        with open(aln_fa, "w") as f:
            run(["mafft", "--auto", "--quiet", str(protein_fa)], stdout=f)

        aligned = {rec.id: str(rec.seq) for rec in SeqIO.parse(aln_fa, "fasta")}
        codon_aln = backtranslate(aligned, cds)

        phy_path = workdir / "codon.phy"
        write_phylip_sequential(codon_aln, phy_path)

        mlc_path = workdir / "out.mlc"
        ctl_path = workdir / "codeml.ctl"
        ctl_path.write_text(CTL_TEMPLATE.format(phy=phy_path.name, mlc=mlc_path.name))
        run(["codeml", ctl_path.name], cwd=str(workdir))

        parsed = parse_mlc(mlc_path.read_text())
        row.update({
            "dN": parsed["dN"], "dS": parsed["dS"], "omega": parsed["omega"],
            "t": parsed["t"], "N": parsed["N"], "S": parsed["S"],
            "n_codons": len(next(iter(codon_aln.values()))) // 3,
            "status": "ok", "detail": "",
        })
    except Exception as e:
        row["status"] = "fail"
        row["detail"] = str(e)
        print(f"FAILED {args.human_gene}/{args.mouse_gene}: {e}", file=sys.stderr)

    with open(args.outfile, "w") as f:
        f.write("\t".join(fields) + "\n")
        f.write("\t".join(str(row[k]) for k in fields) + "\n")


if __name__ == "__main__":
    main()
