"""Protein evidence for changing sequences, one code path for both methods.

Uses the pieces of clam_context.core.analyze: translate_segments (six-frame,
stop-free peptides >= 20 aa, 0-based half-open DNA coordinates) and
ProteinIndex.search (sourmash FracMinHash words; a hit needs >= 2 shared
hashes, containment >= 0.1, and is kept only when the encoded words really
match, with the matched residue positions reported). Settings: protein k 7,
Dayhoff k 12, HP k 30 at scaled 10 (primary), 100 (sparse) and 5
(sensitivity); top 5 per setting; two indexes, oyster_v9 and Swiss-Prot.
A sequence with no peptide >= 20 aa gets no protein assignment.

Input TSV: seq_id, dna (other columns are carried through untouched).
Output: JSON lines, one per sequence, and a flat hits TSV.
"""
import argparse
import csv
import json
import multiprocessing as mp
import re
import time

from clam_context.core import MODES, ProteinIndex, translate_segments

SCALES = (10, 100, 5)
TOP = 5
TIE_CAP = 50
_indexes = {}


def _init(dbs):
    for name, path in dbs.items():
        _indexes[name] = ProteinIndex(path)


def organism(description):
    m = re.search(r"\[([^\[\]]+)\]\s*$", description)
    return m.group(1) if m else ""


def annotate_one(row):
    started = time.monotonic()
    fragments = []
    for i, frag in enumerate(translate_segments(row["dna"], min_aa=20)):
        frag = dict(frag, fragment=i + 1, aa_length=len(frag["peptide"]), hits=[])
        for db, index in _indexes.items():
            for mode in MODES:
                for scaled in SCALES:
                    found = index.search(frag["peptide"], mode, min_shared=2, min_containment=0.1,
                                         top=TIE_CAP, scaled=scaled)
                    if not found:
                        continue
                    best = (found[0]["shared_hashes"], found[0]["shared_encoded_kmers"])
                    ties = sum((h["shared_hashes"], h["shared_encoded_kmers"]) == best for h in found)
                    for rank, h in enumerate(found[:TOP], 1):
                        frag["hits"].append(dict(
                            db=db, mode=mode, scaled=scaled, rank=rank, protein_id=h["protein_id"],
                            description=h["description"], organism=organism(h["description"]),
                            containment=round(h["containment"], 4), shared_hashes=h["shared_hashes"],
                            query_hashes=h["query_hashes"], shared_encoded_kmers=h["shared_encoded_kmers"],
                            matched_positions=h["matched_positions"], shared_query_words=h["shared_query_words"],
                            k=h["k"], tying_best=ties, tie_count_capped=len(found) == TIE_CAP and ties == TIE_CAP))
        fragments.append(frag)
    status = ("no peptide of 20 aa or more" if not fragments else
              "protein evidence" if any(f["hits"] for f in fragments) else "no match at these settings")
    return dict(seq_id=row["seq_id"], status=status, peptides=len(fragments), fragments=fragments,
                seconds=round(time.monotonic() - started, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences", required=True)
    p.add_argument("--oyster", required=True)
    p.add_argument("--swissprot", required=True)
    p.add_argument("--out-jsonl", required=True)
    p.add_argument("--out-hits", required=True)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    with open(a.sequences) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    dbs = {"oyster_v9": a.oyster, "swissprot": a.swissprot}
    start = time.monotonic()
    fields = ["seq_id", "fragment", "strand", "frame", "nt_start", "nt_end", "aa_length", "db", "mode",
              "scaled", "rank", "protein_id", "organism", "description", "containment", "shared_hashes",
              "query_hashes", "shared_encoded_kmers", "tying_best", "tie_count_capped",
              "matched_positions", "peptide"]
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init, initargs=(dbs,)) as pool, \
            open(a.out_jsonl, "w") as js, open(a.out_hits, "w") as ht:
        w = csv.DictWriter(ht, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for n, res in enumerate(pool.imap(annotate_one, rows, chunksize=4), 1):
            js.write(json.dumps(res) + "\n")
            for frag in res["fragments"]:
                for h in frag["hits"]:
                    w.writerow(dict(h, seq_id=res["seq_id"], fragment=frag["fragment"], strand=frag["strand"],
                                    frame=frag["frame"], nt_start=frag["nt_start"], nt_end=frag["nt_end"],
                                    aa_length=frag["aa_length"], peptide=frag["peptide"],
                                    matched_positions=",".join(map(str, h["matched_positions"]))))
            if n % 500 == 0:
                print(f"{n}/{len(rows)} sequences, {time.monotonic() - start:.0f} s", flush=True)
    print(json.dumps(dict(sequences=len(rows), seconds=round(time.monotonic() - start, 1))), flush=True)


if __name__ == "__main__":
    main()
