#!/usr/bin/env python3
"""Independent recount of the notebook 360 kill test for one organism.

Written by Claude Code (independent checker agent), 2026-11-01. It uses only the
Python standard library and runs on Python 3.6 (Sherlock's system python3).
It reads the raw tables (DESeq2, DIAMOND blastx, kmerseek regions, the
search-database Pfam table and the answer key) and does not import or read the
pipeline's own landing or calling code.

Subcommands
-----------
setting   Stream one kmerseek regions.csv.zst and write the families its
          regions land on, as JSON, at several region E-value cut-offs.
summarize Count significant DESeq2 features per labelling, rebuild the
          standard path's family sets, read the per-setting JSON files, pick R
          by the frozen rule and print the verdict and tables.

A region "lands" on a family when at least half of its target interval
[target_start, target_end) (0-based, end exclusive) lies inside one Pfam domain
[start0, end) of that same target protein.
"""
import argparse
import collections
import glob
import gzip
import json
import math
import os
import statistics
import subprocess
import sys

# E-value cut-offs to record. "any" keeps every region, even those with an
# infinite E-value (settings whose Karlin-Altschul K was fitted as 0).
CUTOFFS = [("1", 1.0), ("10", 10.0), ("any", None)]
PADJ = 0.05


def read_tsv(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            yield dict(zip(header, line.rstrip("\n").split("\t")))


def load_pfam(path):
    """target -> list of (start0, end, pfam_acc)."""
    by_target = collections.defaultdict(list)
    for row in read_tsv(path):
        by_target[row["target"]].append(
            (int(row["start0"]), int(row["end"]), row["pfam_acc"])
        )
    return by_target


def landed_families(domains, start, end):
    """Families whose single domain holds at least half of [start, end)."""
    length = end - start
    out = set()
    if length <= 0:
        return out
    for d0, d1, acc in domains:
        overlap = min(end, d1) - max(start, d0)
        if overlap > 0 and 2 * overlap >= length:
            out.add(acc)
    return out


def load_peptide_labelling(path):
    lab = {}
    for row in read_tsv(path):
        lab.setdefault(row["peptide_id"], set()).add(row["labelling"])
    return lab


# --------------------------------------------------------------------------
# setting
# --------------------------------------------------------------------------
def cmd_setting(args):
    pfam = load_pfam(args.pfam)
    peptide_lab = load_peptide_labelling(args.peptide_map)
    name = os.path.basename(args.regions)
    setting = name.split(".", 1)[1].replace(".regions.csv.zst", "")
    status = ""
    status_path = args.regions.replace(".regions.csv.zst", ".status")
    if os.path.exists(status_path):
        with open(status_path) as fh:
            status = fh.read().strip()

    fams = {}  # (labelling, cutoff) -> set
    stats = collections.Counter()
    min_e = float("inf")
    len_sum = 0
    len_sum_finite = 0
    proc = subprocess.Popen(["zstdcat", args.regions], stdout=subprocess.PIPE)
    header = proc.stdout.readline().decode().rstrip("\n").split(",")
    ix = {h: i for i, h in enumerate(header)}
    iq, it = ix["query_name"], ix["target_name"]
    i0, i1, ie = ix["target_start"], ix["target_end"], ix["region_evalue"]
    for raw in proc.stdout:
        f = raw.rstrip(b"\n").split(b",")
        stats["rows"] += 1
        q = f[iq].decode()
        labs = peptide_lab.get(q)
        if labs is None:
            stats["rows_unknown_peptide"] += 1
            continue
        try:
            e = float(f[ie])
        except ValueError:
            e = float("nan")
        start, end = int(f[i0]), int(f[i1])
        len_sum += end - start
        if math.isfinite(e):
            stats["rows_finite_evalue"] += 1
            len_sum_finite += end - start
            if e < min_e:
                min_e = e
            if e <= 1:
                stats["rows_evalue_le_1"] += 1
            if e <= 10:
                stats["rows_evalue_le_10"] += 1
        domains = pfam.get(f[it].decode())
        if not domains:
            stats["rows_target_without_pfam"] += 1
            continue
        landed = landed_families(domains, start, end)
        if not landed:
            continue
        stats["rows_landing_any_evalue"] += 1
        for lab in labs:
            for cname, cval in CUTOFFS:
                if cval is None or (math.isfinite(e) and e <= cval):
                    fams.setdefault((lab, cname), set()).update(landed)
    proc.wait()
    if proc.returncode != 0:
        sys.exit("zstdcat failed on %s" % args.regions)
    out = {
        "setting": setting,
        "status": status,
        "header_has_regions": "region_evalue" in ix,
        "stats": dict(stats),
        "min_finite_evalue": None if min_e == float("inf") else min_e,
        "mean_target_interval": (len_sum / stats["rows"]) if stats["rows"] else None,
        "mean_target_interval_finite_e": (
            len_sum_finite / stats["rows_finite_evalue"]
            if stats["rows_finite_evalue"]
            else None
        ),
        "families": {"%s|%s" % k: sorted(v) for k, v in fams.items()},
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------
def count_sig(path):
    n = up = down = na = total = 0
    for row in read_tsv(path):
        total += 1
        p = row["padj"]
        if p in ("", "NA", "nan"):
            na += 1
            continue
        if float(p) < PADJ:
            n += 1
            if float(row["log2FoldChange"]) > 0:
                up += 1
            else:
                down += 1
    return total, na, n, up, down


def standard_families(deseq_path, blastx_path, pfam, mode):
    """Families the standard path reports for one labelling.

    mode "all": every Pfam family of the best hit's protein.
    mode "aligned": only domains whose span overlaps the aligned part of the
    hit protein by at least half the domain's length.
    """
    de = set()
    for row in read_tsv(deseq_path):
        p = row["padj"]
        if p not in ("", "NA", "nan") and float(p) < PADJ:
            de.add(row["feature"])
    best = {}
    with open(blastx_path) as fh:
        for line in fh:
            # qseqid sseqid pident length evalue bitscore qstart qend sstart send
            f = line.rstrip("\n").split("\t")
            q, s, ev, bits = f[0], f[1], float(f[4]), float(f[5])
            sstart, send = int(f[8]), int(f[9])
            key = (-bits, ev, s)
            if q not in best or key < best[q][0]:
                best[q] = (key, s, min(sstart, send) - 1, max(sstart, send))
    fams = set()
    n_hit = 0
    for q in de:
        if q not in best:
            continue
        n_hit += 1
        _, s, a0, a1 = best[q]
        for d0, d1, acc in pfam.get(s, []):
            if mode == "all":
                fams.add(acc)
            else:
                ov = min(a1, d1) - max(a0, d0)
                if ov > 0 and 2 * ov >= (d1 - d0):
                    fams.add(acc)
    return fams, len(de), n_hit


def cmd_summarize(args):
    key = list(read_tsv(args.answer_key))
    changed = set(r["pfam_acc"] for r in key if r["status"] in ("up", "down"))
    unchanged = set(r["pfam_acc"] for r in key if r["status"] == "unchanged")
    pfam = load_pfam(args.pfam)
    db_fams = set(acc for doms in pfam.values() for _, _, acc in doms)
    labellings = ["real"] + ["swap%d" % i for i in range(1, 10)]
    out = {}

    print("== answer key")
    print("changed families", len(changed), "unchanged families", len(unchanged),
          "overlap", len(changed & unchanged))
    print("changed families absent from search db:", sorted(changed - db_fams))
    print("unchanged families absent from search db:", sorted(unchanged - db_fams))

    print("\n== DESeq2 significant features (padj < 0.05)")
    print("labelling\tkmers_total\tkmers_na\tkmers_sig\tkmers_up\tkmers_down"
          "\ttx_total\ttx_na\ttx_sig\ttx_up\ttx_down")
    sig = {}
    for lab in labellings:
        k = count_sig(os.path.join(args.deseq2_dir, "%s.kmers.%s.deseq2.tsv.gz" % (args.organism, lab)))
        t = count_sig(os.path.join(args.deseq2_dir, "%s.transcripts.%s.deseq2.tsv.gz" % (args.organism, lab)))
        sig[lab] = (k, t)
        print("\t".join([lab] + [str(x) for x in k + t]))
    out["deseq2"] = sig

    print("\n== standard path")
    print("labelling\tmode\tde_transcripts\tde_with_hit\tfamilies\trecovered\toutside\toutside_unchanged_key")
    std = {}
    for lab in labellings:
        for mode in ("all", "aligned"):
            fams, n_de, n_hit = standard_families(
                os.path.join(args.deseq2_dir, "%s.transcripts.%s.deseq2.tsv.gz" % (args.organism, lab)),
                args.blastx, pfam, mode)
            rec = len(fams & changed)
            outside = len(fams - changed)
            std[(lab, mode)] = (rec, outside, sorted(fams))
            print("\t".join(str(x) for x in (lab, mode, n_de, n_hit, len(fams), rec, outside, len(fams & unchanged))))

    print("\n== kmerseek settings (real labels)")
    rows = []
    for path in sorted(glob.glob(os.path.join(args.setting_dir, "*.json"))):
        with open(path) as fh:
            js = json.load(fh)
        row = {"setting": js["setting"], "status": js["status"], "stats": js["stats"],
               "min_e": js["min_finite_evalue"], "mean_len": js["mean_target_interval"]}
        for cname, _ in CUTOFFS:
            for lab in labellings:
                fams = set(js["families"].get("%s|%s" % (lab, cname), []))
                row[(lab, cname)] = (len(fams & changed), len(fams - changed), len(fams & unchanged))
        rows.append(row)
    print("setting\tstatus\trows\trows_finite_e\tmin_e\tmean_target_len\t"
          "rec_e1\tout_e1\trec_e10\tout_e10\trec_any\tout_any")
    for r in rows:
        s = r["stats"]
        print("\t".join(str(x) for x in (
            r["setting"], r["status"], s.get("rows", 0), s.get("rows_finite_evalue", 0),
            r["min_e"], None if r["mean_len"] is None else round(r["mean_len"], 1),
            r[("real", "1")][0], r[("real", "1")][1],
            r[("real", "10")][0], r[("real", "10")][1],
            r[("real", "any")][0], r[("real", "any")][1])))

    reduced = [r for r in rows if not r["setting"].startswith("protein20.")]
    reduced.sort(key=lambda r: (-r[("real", "1")][0], r[("real", "1")][1], r["setting"]))
    R = reduced[0]
    r_real = R[("real", "1")][0]
    r_swaps = [R[("swap%d" % i, "1")][0] for i in range(1, 10)]
    med = statistics.median(r_swaps)
    std_real = std[("real", "all")][0]
    part1 = r_real >= std_real
    part2 = r_real >= 5 * med
    print("\n== verdict")
    print("R =", R["setting"], "recovered", r_real, "outside", R[("real", "1")][1])
    print("runner-up =", reduced[1]["setting"], reduced[1][("real", "1")])
    print("standard recovered", std_real, "outside", std[("real", "all")][1])
    print("part 1 (R >= standard):", r_real, ">=", std_real, "->", part1)
    print("part 2 (R >= 5 x median swaps):", r_real, ">= 5 x", med, "swaps", r_swaps, "->", part2)
    print("VERDICT:", "PASS" if (part1 and part2) else "FAIL")
    best10 = max(reduced, key=lambda r: r[("real", "10")][0])
    print("best reduced at E <= 10:", best10["setting"], best10[("real", "10")])
    bestany = max(reduced, key=lambda r: r[("real", "any")][0])
    print("best reduced with no E-value filter:", bestany["setting"], bestany[("real", "any")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("setting")
    s.add_argument("--regions", required=True)
    s.add_argument("--pfam", required=True)
    s.add_argument("--peptide-map", required=True)
    s.add_argument("--out", required=True)
    m = sub.add_parser("summarize")
    m.add_argument("--organism", default="drosophila")
    m.add_argument("--answer-key", required=True)
    m.add_argument("--pfam", required=True)
    m.add_argument("--deseq2-dir", required=True)
    m.add_argument("--blastx", required=True)
    m.add_argument("--setting-dir", required=True)
    args = ap.parse_args()
    if args.cmd == "setting":
        cmd_setting(args)
    elif args.cmd == "summarize":
        cmd_summarize(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
