"""Build the Botryllus truth set for testing the read frame choice.

1. Check every new-genome CDS: it must translate (standard code, trailing stop removed) to exactly
   its protein in protein.fasta. Only CDS that pass are used.
2. Link the paper-list genes (old gene models g<N>) to new-genome genes (FUN...) with a DIAMOND
   blastp of the list proteins against protein.fasta: best hit with identity >= 90 % over >= 80 %
   of the old protein.
3. Gene set: every linked list gene, plus a random sample of other genes, about 2,000 in total.
4. Simulated reads of the real read length cut from each CDS at evenly spaced starts, on a
   random strand, so each read's true frame is known. Frame index follows pec-tools:
   0,1,2 = read offsets 0,1,2 on the read; 3,4,5 = offsets 0,1,2 on its reverse complement.

Usage: python sra_frame_test_prepare.py <cds.fasta> <protein.fasta> <query_map.tsv> <diamond_list_vs_fun.tsv> <out_dir> [read_len=150] [n_genes=2000] [reads_per_gene=20]
"""

import csv
import random
import sys
from pathlib import Path

CODE = "KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF"
IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
SEED = 361


def read_fasta(p):
    """Records in file order. Isoforms share a header name here, so the n-th record with a
    given name gets the id name (n = 1) or name#n (n > 1)."""
    recs, name, buf = [], None, []
    for line in open(p):
        line = line.rstrip()
        if line.startswith(">"):
            if name:
                recs.append((name, "".join(buf)))
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line)
    if name:
        recs.append((name, "".join(buf)))
    seen, out = {}, {}
    for name, s in recs:
        seen[name] = seen.get(name, 0) + 1
        out[name if seen[name] == 1 else f"{name}#{seen[name]}"] = s
    return out


def translate(s):
    s = s.upper()
    aa = []
    for i in range(0, len(s) - 2, 3):
        c = s[i : i + 3]
        aa.append(
            CODE[16 * IDX[c[0]] + 4 * IDX[c[1]] + IDX[c[2]]]
            if all(x in IDX for x in c)
            else "X"
        )
    return "".join(aa)


def revcomp(s):
    return s.translate(COMP)[::-1]


def main():
    cds_p, prot_p, qmap_p, dmnd_p, out = sys.argv[1:6]
    read_len = int(sys.argv[6]) if len(sys.argv) > 6 else 150
    n_genes = int(sys.argv[7]) if len(sys.argv) > 7 else 2000
    per_gene = int(sys.argv[8]) if len(sys.argv) > 8 else 20
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cds, prot = read_fasta(cds_p), read_fasta(prot_p)
    ok = {}
    fail = {
        "missing_protein": 0,
        "length_not_multiple_of_3": 0,
        "translation_differs": 0,
    }
    for name, s in cds.items():
        if name not in prot:
            fail["missing_protein"] += 1
            continue
        if len(s) % 3:
            fail["length_not_multiple_of_3"] += 1
        if translate(s).rstrip("*") == prot[name].rstrip("*"):
            ok[name] = s
        elif any(
            translate(s).rstrip("*") == prot[o].rstrip("*")
            for o in prot
            if o.split("#")[0] == name.split("#")[0]
        ):
            ok[name] = s  # same gene name, isoforms listed in a different order
        else:
            fail["translation_differs"] += 1
    print(
        f"CDS records {len(cds)}; translate to their protein: {len(ok)}; failures {fail}"
    )

    # paper-list genes -> FUN genes
    qmap = list(csv.DictReader(open(qmap_p), delimiter="\t"))
    list_prot = {r["query"]: r for r in qmap if r["set"] == "paper_list"}
    best = {}
    for r in csv.reader(open(dmnd_p), delimiter="\t"):
        q, s, pid, qcov, bits = r[0], r[1], float(r[2]), float(r[3]), float(r[4])
        if (
            q in list_prot
            and pid >= 90
            and qcov >= 80
            and (q not in best or bits > best[q][1])
        ):
            best[q] = (s, bits)
    linked = {}
    for q, (s, _) in best.items():
        # DIAMOND reports the header name; take that name's first isoform whose CDS passed
        s_ok = next((o for o in ok if o.split("#")[0] == s), None)
        if s_ok:
            linked.setdefault(s_ok, set()).add(list_prot[q]["gene"])
    print(
        f"list proteins {len(list_prot)}; with a FUN hit >= 90 % identity over >= 80 %: {len(best)}; distinct FUN genes with passing CDS: {len(linked)}"
    )
    rng = random.Random(SEED)
    others = sorted(set(ok) - set(linked))
    rand = rng.sample(others, max(0, n_genes - len(linked)))
    genes = [(g, "paper_list") for g in sorted(linked)] + [
        (g, "random") for g in sorted(rand)
    ]
    with open(out / "truth_genes.tsv", "w") as f:
        f.write("fun_gene\tset\told_genes\tcds_len\n")
        for g, st in genes:
            f.write(f"{g}\t{st}\t{','.join(sorted(linked.get(g, [])))}\t{len(ok[g])}\n")
    n_reads = short = 0
    with open(out / "sim_reads.fastq", "w") as f:
        for g, st in genes:
            s = ok[g]
            if len(s) < read_len:
                short += 1
                continue
            span = len(s) - read_len
            starts = sorted(
                {round(i * span / max(1, per_gene - 1)) for i in range(per_gene)}
            )
            for st0 in starts:
                piece = s[st0 : st0 + read_len]
                off = (3 - st0 % 3) % 3
                if rng.random() < 0.5:
                    seq, frame = piece, off
                else:
                    seq, frame = revcomp(piece), 3 + off
                f.write(f"@sim|{g}|{st}|{st0}|{frame}\n{seq}\n+\n{'I' * len(seq)}\n")
                n_reads += 1
    with open(out / "cds_check.tsv", "w") as f:
        f.write("item\tcount\n")
        f.write(f"cds_records\t{len(cds)}\ncds_translate_to_protein\t{len(ok)}\n")
        for k, v in fail.items():
            f.write(f"{k}\t{v}\n")
        f.write(
            f"list_proteins\t{len(list_prot)}\nlist_proteins_linked\t{len(best)}\nlist_fun_genes\t{len(linked)}\n"
        )
        f.write(
            f"random_fun_genes\t{len(rand)}\ngenes_shorter_than_read\t{short}\nsimulated_reads\t{n_reads}\n"
        )
    with open(out / "cds_ok.fasta", "w") as f:
        for g, s in ok.items():
            f.write(f">{g}\n{s}\n")
    with open(out / "protein_ok.fasta", "w") as f:
        for g in ok:
            f.write(f">{g}\n{translate(ok[g]).rstrip('*')}\n")
    print(
        f"genes {len(genes)} ({len(linked)} list, {len(rand)} random); shorter than {read_len} nt: {short}; simulated reads {n_reads}"
    )


if __name__ == "__main__":
    main()
