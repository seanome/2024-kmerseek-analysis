#!/usr/bin/env python3
"""Notebook 301, step 2: build the query set for kill test 1.

Standard library only, so it runs on Sherlock with `module load python/3.12.1`.

Inputs
  --sprot-dat     uniprot_sprot.dat.gz, the current Swiss-Prot release
  --elm           ELM true-positive instances under 60 residues: instances_kept.tsv from
                  closed PR 181 (seanome/2024-kmerseek-analysis, commit e08c5ca3), ELM
                  download of 2026-10-20

Real queries (2,000): Swiss-Prot proteins with at least one feature, half human (taxon
9606), half microbial (lineage starting with Bacteria or Archaea), drawn at random with
seed 301 from every protein that qualifies. A feature is
  * a Swiss-Prot SIGNAL, TRANSIT or TRANSMEM feature with experimental evidence
    (ECO:0000269), both ends known, shorter than 60 residues; or
  * an ELM true-positive instance on that Swiss-Prot entry whose residues match the
    current sequence.
Every feature of a chosen protein is recorded, not only the one that qualified it.

Chimeras (500): a Swiss-Prot host protein that is not a real query, has no SIGNAL or
TRANSIT feature of any evidence and no ELM degron (ELM class DEG_*), is 100 to 1,000
residues long, and is human or microbial (half each). Into it, at a random position at
least 10 residues from either end, goes the linker GGGGSGGGGS followed by one insert:
every ELM degron instance once (190 on 2026-10-20), and signal peptides from the
qualifying SIGNAL features for the rest. No insert is used twice. The true feature is the
insert's residues in the chimera.

Outputs in --out
  queries.fasta              real proteins then chimeras; ids are UniProt accessions and
                             chimera_NNN
  truth.tsv                  one row per feature: query_id, query_kind (real/chimera),
                             source_accession, feature_type, feature_class, start, end
                             (1-based, inclusive), length_aa, evidence, residues
  chimeras.tsv               host, insert source, insert type, position, true start/end
  feature_type_counts.tsv    features and proteins per type and organism group
  filter_counts.tsv          how many proteins and features survive each step
  accessions_for_uniref90.txt every accession whose UniRef90 cluster must be removed from
                             the database: real queries, chimera hosts, insert sources
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SEED = 301
N_HUMAN = 1000
N_MICROBIAL = 1000
N_CHIMERAS = 500
MAX_FEATURE_AA = 60  # a feature is shorter than this
SPROT_TYPES = ("SIGNAL", "TRANSIT", "TRANSMEM")
EXPERIMENTAL = "ECO:0000269"
LINKER = "GGGGS" * 2
HOST_MIN_AA, HOST_MAX_AA = 100, 1000
INSERT_MARGIN_AA = 10  # insertion point at least this far from either host end
HUMAN_TAXID = "9606"
MICROBIAL_DOMAINS = ("Bacteria", "Archaea")
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Entry:
    accessions: list[str]
    taxid: str = ""
    domain: str = ""
    sequence: str = ""
    # (type, start, end, evidence codes) for every FT line with a plain start..end range
    features: list[tuple[str, int, int, str]] = field(default_factory=list)
    # every feature type on the entry, whatever its location or evidence
    all_types: set[str] = field(default_factory=set)

    @property
    def accession(self) -> str:
        return self.accessions[0]

    @property
    def group(self) -> str:
        if self.taxid == HUMAN_TAXID:
            return "human"
        if self.domain in MICROBIAL_DOMAINS:
            return "microbial"
        return "other"


FT_START = re.compile(r"^FT   (\S+)\s+(\S+)")
RANGE = re.compile(r"^(\d+)\.\.(\d+)$")
EVIDENCE = re.compile(r'/evidence="([^"]*)"')


def parse_sprot_dat(lines) -> list[Entry]:
    """Parse UniProt flat-file lines into entries. Only the fields this step needs."""
    entries: list[Entry] = []
    e: Entry | None = None
    seq: list[str] = []
    in_seq = False
    cur = None  # [type, start, end, evidence] of the FT block being read
    for line in lines:
        tag = line[:2]
        if tag == "ID":
            e, seq, in_seq, cur = Entry(accessions=[]), [], False, None
        elif e is None:
            continue
        elif tag == "AC":
            e.accessions += [a for a in line[5:].replace(";", " ").split() if a]
        elif tag == "OX" and not e.taxid:
            m = re.search(r"NCBI_TaxID=(\d+)", line)
            if m:
                e.taxid = m.group(1)
        elif tag == "OC" and not e.domain:
            e.domain = line[5:].split(";")[0].strip().rstrip(".")
        elif tag == "FT":
            m = FT_START.match(line)
            if m and not line.startswith("FT    "):
                if cur is not None:
                    e.features.append(tuple(cur))
                ftype, loc = m.group(1), m.group(2)
                e.all_types.add(ftype)
                r = RANGE.match(loc)
                cur = [ftype, int(r.group(1)), int(r.group(2)), ""] if r else None
            elif cur is not None:
                ev = EVIDENCE.search(line)
                if ev:
                    cur[3] += ev.group(1)
                elif "/evidence=" in line:
                    # an evidence list that wraps onto the next FT line
                    cur[3] += line.split("/evidence=", 1)[1].strip('"\n ')
                elif cur[3] and not line[21:].lstrip().startswith("/"):
                    cur[3] += line[21:].strip().strip('"')
        elif tag == "SQ":
            in_seq = True
            if cur is not None:
                e.features.append(tuple(cur))
                cur = None
        elif tag == "//":
            e.sequence = "".join(seq)
            entries.append(e)
            e = None
        elif in_seq and line.startswith("     "):
            seq.append(line.replace(" ", "").strip())
    return entries


@dataclass
class Feature:
    accession: str
    feature_type: str  # SIGNAL / TRANSIT / TRANSMEM / ELM
    feature_class: str  # ELM class name, or the Swiss-Prot type again
    start: int
    end: int
    evidence: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def sprot_features(e: Entry) -> list[Feature]:
    out = []
    for ftype, s, t, ev in e.features:
        if ftype not in SPROT_TYPES or EXPERIMENTAL not in ev:
            continue
        if not (1 <= s <= t <= len(e.sequence)) or t - s + 1 >= MAX_FEATURE_AA:
            continue
        out.append(Feature(e.accession, ftype, ftype, s, t, EXPERIMENTAL))
    return out


def read_elm(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def elm_features(rows, by_acc: dict[str, Entry], counts: Counter) -> list[Feature]:
    """ELM instances on Swiss-Prot entries whose residues match the current sequence."""
    out = []
    for r in rows:
        e = by_acc.get(r["protein"])
        if e is None:
            counts["ELM instance not on a Swiss-Prot entry"] += 1
            continue
        s, t = int(r["start"]), int(r["end"])
        if e.sequence[s - 1 : t] != r["feature_residues"]:
            counts["ELM instance residues differ from Swiss-Prot sequence"] += 1
            continue
        if t - s + 1 >= MAX_FEATURE_AA:
            counts["ELM instance 60 residues or longer"] += 1
            continue
        out.append(
            Feature(e.accession, "ELM", r["elm_class"], s, t, f"ELM {r['instance_id']}")
        )
        counts["ELM instances kept"] += 1
    return out


def splice(host: str, position: int, insert: str) -> tuple[str, int, int]:
    """Host residues 1..position, linker, insert, the rest of the host.

    Returns the chimera and the insert's start and end in it (1-based, inclusive)."""
    chimera = host[:position] + LINKER + insert + host[position:]
    start = position + len(LINKER) + 1
    return chimera, start, start + len(insert) - 1


def write_fasta(path: Path, records) -> None:
    with open(path, "w") as fh:
        for name, s in records:
            fh.write(f">{name}\n")
            for i in range(0, len(s), 60):
                fh.write(s[i : i + 60] + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sprot-dat", type=Path, required=True)
    ap.add_argument("--elm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    counts: Counter = Counter()

    opener = gzip.open if str(a.sprot_dat).endswith(".gz") else open
    with opener(a.sprot_dat, "rt") as fh:
        entries = parse_sprot_dat(fh)
    counts["Swiss-Prot entries"] = len(entries)
    entries = [e for e in entries if set(e.sequence) <= AMINO_ACIDS]
    counts["entries with only the 20 standard residues"] = len(entries)
    by_acc = {acc: e for e in entries for acc in e.accessions}
    primary = {e.accession: e for e in entries}

    feats: dict[str, list[Feature]] = defaultdict(list)
    for e in entries:
        for f in sprot_features(e):
            feats[f.accession].append(f)
    counts["Swiss-Prot SIGNAL/TRANSIT/TRANSMEM, ECO:0000269, < 60 aa"] = sum(
        len(v) for v in feats.values()
    )
    elm_rows = read_elm(a.elm)
    counts["ELM instances read"] = len(elm_rows)
    elm = elm_features(elm_rows, by_acc, counts)
    for f in elm:
        feats[f.accession].append(f)

    pools = {
        g: sorted(acc for acc in feats if primary[acc].group == g)
        for g in ("human", "microbial")
    }
    for g, accs in pools.items():
        counts[f"{g} proteins with a feature"] = len(accs)
    chosen = {
        "human": sorted(rng.sample(pools["human"], N_HUMAN)),
        "microbial": sorted(rng.sample(pools["microbial"], N_MICROBIAL)),
    }
    real = [acc for g in ("human", "microbial") for acc in chosen[g]]
    real_set = set(real)

    # Chimera hosts and inserts.
    deg_accs = {f.accession for f in elm if f.feature_class.startswith("DEG_")}
    hosts = {
        g: sorted(
            e.accession
            for e in entries
            if e.group == g
            and e.accession not in real_set
            and not (e.all_types & {"SIGNAL", "TRANSIT"})
            and e.accession not in deg_accs
            and HOST_MIN_AA <= len(e.sequence) <= HOST_MAX_AA
        )
        for g in ("human", "microbial")
    }
    degrons = sorted(
        (f for f in elm if f.feature_class.startswith("DEG_")),
        key=lambda f: (f.accession, f.start),
    )
    signals = sorted(
        (f for v in feats.values() for f in v if f.feature_type == "SIGNAL"),
        key=lambda f: (f.accession, f.start),
    )
    n_signal = N_CHIMERAS - len(degrons)
    inserts = degrons + rng.sample(signals, n_signal)
    rng.shuffle(inserts)
    host_picks = rng.sample(hosts["human"], N_CHIMERAS // 2) + rng.sample(
        hosts["microbial"], N_CHIMERAS - N_CHIMERAS // 2
    )
    rng.shuffle(host_picks)

    fasta, truth, chim_rows = [], [], []
    for acc in real:
        e = primary[acc]
        fasta.append((acc, e.sequence))
        for f in sorted(feats[acc], key=lambda f: (f.start, f.end, f.feature_class)):
            truth.append(
                {
                    "query_id": acc,
                    "query_kind": "real",
                    "organism_group": e.group,
                    "source_accession": acc,
                    "feature_type": f.feature_type,
                    "feature_class": f.feature_class,
                    "start": f.start,
                    "end": f.end,
                    "length_aa": f.length,
                    "evidence": f.evidence,
                    "residues": e.sequence[f.start - 1 : f.end],
                }
            )
    for i, (host_acc, ins) in enumerate(zip(host_picks, inserts)):
        host = primary[host_acc].sequence
        pos = rng.randint(INSERT_MARGIN_AA, len(host) - INSERT_MARGIN_AA)
        insert_seq = primary[ins.accession].sequence[ins.start - 1 : ins.end]
        seq, s, t = splice(host, pos, insert_seq)
        assert seq[s - 1 : t] == insert_seq
        qid = f"chimera_{i:03d}"
        fasta.append((qid, seq))
        kind = "degron" if ins.feature_class.startswith("DEG_") else "signal peptide"
        chim_rows.append(
            {
                "query_id": qid,
                "host_accession": host_acc,
                "host_group": primary[host_acc].group,
                "host_length_aa": len(host),
                "insert_after_host_residue": pos,
                "linker": LINKER,
                "insert_kind": kind,
                "insert_source_accession": ins.accession,
                "insert_source_start": ins.start,
                "insert_source_end": ins.end,
                "true_start": s,
                "true_end": t,
            }
        )
        truth.append(
            {
                "query_id": qid,
                "query_kind": "chimera",
                "organism_group": primary[host_acc].group,
                "source_accession": ins.accession,
                "feature_type": ins.feature_type,
                "feature_class": ins.feature_class,
                "start": s,
                "end": t,
                "length_aa": t - s + 1,
                "evidence": ins.evidence,
                "residues": insert_seq,
            }
        )

    write_fasta(a.out / "queries.fasta", fasta)
    for name, rows in (("truth.tsv", truth), ("chimeras.tsv", chim_rows)):
        with open(a.out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)

    type_counts: Counter = Counter()
    type_proteins: dict[tuple, set] = defaultdict(set)
    for r in truth:
        key = (r["query_kind"], r["organism_group"], r["feature_type"])
        type_counts[key] += 1
        type_proteins[key].add(r["query_id"])
    with open(a.out / "feature_type_counts.tsv", "w") as fh:
        fh.write("query_kind\torganism_group\tfeature_type\tn_features\tn_queries\n")
        for k in sorted(type_counts):
            fh.write("\t".join(k) + f"\t{type_counts[k]}\t{len(type_proteins[k])}\n")

    counts["real queries"] = len(real)
    counts["chimeras"] = len(chim_rows)
    counts["chimera degron inserts"] = len(degrons)
    counts["chimera signal-peptide inserts"] = n_signal
    for g in hosts:
        counts[f"{g} chimera host candidates"] = len(hosts[g])
    with open(a.out / "filter_counts.tsv", "w") as fh:
        fh.write("step\tcount\n")
        for k, v in counts.items():
            fh.write(f"{k}\t{v}\n")

    need = set(real) | {r["host_accession"] for r in chim_rows}
    need |= {r["insert_source_accession"] for r in chim_rows}
    (a.out / "accessions_for_uniref90.txt").write_text("\n".join(sorted(need)) + "\n")
    for k, v in counts.items():
        print(f"{k}\t{v}")


if __name__ == "__main__":
    main()
