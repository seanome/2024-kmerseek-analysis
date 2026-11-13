"""Labeled synthetic fixtures for experiment 358. They exercise the software only.

Writes, into the folder given (default fixtures/):
    proteins_human.fasta, proteins_contaminants.fasta, calls.tsv, accepted.tsv,
    error_labels.tsv (which error each call was given), inputs.json (data_origin =
    synthetic_fixture, so every run on them ends in INSUFFICIENT_EVIDENCE).

The proteome has random proteins with human residue frequencies, paralog families,
isoforms that skip a segment, a 12-residue segment shared by 3-8 proteins, repeat-rich
proteins and contaminants. Calls are tryptic peptides with labeled errors.
"""

import json
import random
import sys
from pathlib import Path

# Human proteome residue frequencies (UniProt UP000005640 statistics, rounded).
FREQ = dict(A=7.0, C=2.3, D=4.7, E=7.1, F=3.7, G=6.6, H=2.6, I=4.3, K=5.7, L=10.0, M=2.1,
            N=3.6, P=6.3, Q=4.8, R=5.6, S=8.3, T=5.4, V=6.0, W=1.2, Y=2.7)
AA = list(FREQ)
W = list(FREQ.values())
N_CALLS = 500


def rand_seq(rng, n):
    return "".join(rng.choices(AA, W, k=n))


def mutate(rng, seq, identity):
    return "".join(c if rng.random() < identity else rng.choice(AA) for c in seq)


def tryptic(seq, lo=7, hi=25):
    out, start = [], 0
    for i, c in enumerate(seq):
        if c in "KR" and (i + 1 == len(seq) or seq[i + 1] != "P"):
            pep = seq[start:i + 1]
            if lo <= len(pep) <= hi:
                out.append(pep)
            start = i + 1
    return out


def make_proteome(rng):
    human = []
    for i in range(300):
        human.append((f"H{i:04d}", rand_seq(rng, rng.randint(150, 700))))
    for f in range(30):  # paralog families
        base = rand_seq(rng, rng.randint(200, 500))
        for c in range(4):
            human.append((f"PAR{f:02d}_{c}", base if c == 0 else mutate(rng, base, rng.uniform(0.7, 0.95))))
    for i in range(20):  # isoforms skipping 20-60 residues
        acc, seq = human[i]
        a = rng.randint(20, len(seq) // 2)
        human.append((f"{acc}-2", seq[:a] + seq[a + rng.randint(20, 60):]))
    for g in range(10):  # one segment in several proteins
        seg = rand_seq(rng, 11) + "K"
        for j in rng.sample(range(20, 300), rng.randint(3, 8)):
            acc, seq = human[j]
            p = rng.randint(10, len(seq) - 10)
            human[j] = (acc, seq[:p] + "K" + seg + seq[p:])
    for i in range(5):  # repeat-rich
        unit = rng.choice(["GPP", "PPGPPGQ", "QQQQP", "SSGS", "EEKK"])
        human.append((f"REP{i}", rand_seq(rng, 60) + "K" + unit * 20 + "R" + rand_seq(rng, 60)))
    contaminants = [(f"CON_{i:02d}", rand_seq(rng, rng.randint(200, 600))) for i in range(15)]
    return human, contaminants


ERRORS = [("none", 40), ("il_swap", 10), ("kq_swap", 5), ("adjacent_swap", 15),
          ("n_to_gg", 5), ("deamidation", 5), ("substitution", 10), ("wrong_call", 5), ("x", 5)]


def add_error(rng, pep, kind):
    s, bad = list(pep), []
    if kind == "il_swap":
        idx = [i for i, c in enumerate(s) if c in "IL"]
        for i in idx[:1]:
            s[i] = "L" if s[i] == "I" else "I"
    elif kind == "kq_swap":
        idx = [i for i, c in enumerate(s[:-1]) if c in "KQ"]
        for i in idx[:1]:
            s[i] = "Q" if s[i] == "K" else "K"
            bad.append(i)
    elif kind == "adjacent_swap":
        i = rng.randrange(1, len(s) - 2)
        s[i], s[i + 1] = s[i + 1], s[i]
        bad += [i, i + 1]
    elif kind == "n_to_gg":
        idx = [i for i, c in enumerate(s) if c == "N"]
        if idx:
            i = idx[0]
            s[i:i + 1] = ["G", "G"]
            bad += [i, i + 1]
    elif kind == "deamidation":
        idx = [i for i, c in enumerate(s) if c in "NQ"]
        for i in idx[:1]:
            s[i] = "D" if s[i] == "N" else "E"
            bad.append(i)
    elif kind == "substitution":
        i = rng.randrange(len(s))
        s[i] = rng.choice([a for a in AA if a != s[i]])
        bad.append(i)
    elif kind == "wrong_call":
        s = list(rand_seq(rng, len(s) - 1)) + ["K"]
        bad = list(range(len(s)))
    elif kind == "x":
        i = rng.randrange(len(s))
        s[i] = "X"
        bad.append(i)
    return "".join(s), bad


def main(out=Path(__file__).resolve().parent / "fixtures"):
    rng = random.Random(358)
    out.mkdir(exist_ok=True)
    human, cons = make_proteome(rng)
    for name, recs in [("proteins_human.fasta", human), ("proteins_contaminants.fasta", cons)]:
        with open(out / name, "w") as f:
            for acc, seq in recs:
                f.write(f">{acc}\n{seq}\n")
    pool = [(acc, p) for acc, seq in human + cons for p in tryptic(seq)]
    calls, accepted, labels = [], [], []
    kinds = [k for k, w in ERRORS for _ in range(w)]
    for i in range(N_CALLS):
        acc, pep = rng.choice(pool)
        kind = rng.choice(kinds)
        call, bad = add_error(rng, pep, kind)
        scores = [round(rng.uniform(0.3, 0.6) if j in bad else rng.uniform(0.8, 1.0), 3)
                  for j in range(len(call))]
        alts = ""
        if bad and kind in ("kq_swap", "deamidation", "substitution") and rng.random() < 0.5:
            alts = f"{bad[0] + 1}:{pep[bad[0]]}"
        cid = f"C{i:04d}"
        calls.append(dict(call_id=cid, run_id="R1", scan_id=str(1000 + i), called_sequence=call,
                          call_score=round(sum(scores) / len(scores), 4),
                          residue_scores=",".join(map(str, scores)), alternatives=alts))
        q = 0.001 if i % 50 else 0.05  # 10 calls fail the q-value filter
        accepted_pep = pep if i % 100 != 7 else rand_seq(rng, len(pep))  # 5 not in collection
        accepted.append(dict(call_id=cid, accepted_peptide=accepted_pep, evidence_tool="fixture",
                             evidence_qvalue=q, evidence_score=""))
        labels.append(dict(call_id=cid, error=kind, drawn_from=acc))
    for name, rows in [("calls.tsv", calls), ("accepted.tsv", accepted), ("error_labels.tsv", labels)]:
        cols = list(rows[0])
        with open(out / name, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")
    (out / "inputs.json").write_text(json.dumps({
        "data_origin": "synthetic_fixture",
        "proteins": [{"path": "proteins_human.fasta", "role": "human"},
                     {"path": "proteins_contaminants.fasta", "role": "contaminant"}],
        "calls": "calls.tsv", "accepted": "accepted.tsv"}, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "fixtures")
