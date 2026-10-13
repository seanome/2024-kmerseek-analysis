#!/usr/bin/env python3
"""Build notebook 290's task table from the finished dark-set run, and check it against the run.

    tasks.py <run_dir> <run_repo_dir> <out_dir>

<run_dir>       runs/ladder-0.4-midi of the dark-set run (PR 57): its trace and work dirs.
<run_repo_dir>  the run's pipeline folder (nextflow-runs/invertebrate-dark-set of the clone
                the run used), for assets/kappa_by_alphabet.tsv and nextflow.config.
<out_dir>       where chunks.tsv, tasks.tsv, resources.tsv and command_check.tsv go.

What it does, and what it refuses:
  1. The six kmerseek settings (alphabet, k, scaled 1, mask off, extend at the alphabet's
     own mismatch penalty) are the ones whose region files notebook 279 sections 7 and 9
     read. It reads the setting names off results-0.4/protein-min-evalue/ and stops unless
     they are exactly SIX_SETTINGS.
  2. The extension penalty is computed as main.nf's optimalPenalty does: kappa from
     assets/kappa_by_alphabet.tsv, classes from the trailing number of the alphabet name,
     C = -ln(1 - kappa) / ln(1 + kappa * classes - kappa), written with two decimals by
     penaltyString (Java's %.2f, half-up on the shortest decimal form). The X-drop is
     penaltyString(4 * C) as xdropFor does, not the table's xdrop_4c column.
  3. For each of the six settings and every species, it finds the run's kmerseekSearch
     tasks in the trace, reads each task's .command.sh, and rebuilds the kmerseek command
     this experiment will run. The two must be the same text except the query file and the
     index path (the run stages the index under its own name in the work dir; the
     resolved path must be the stored index). Any other difference stops the script.
  4. The real query chunks are the ones those tasks staged. Each chunk's md5, protein count
     and residue count go into chunks.tsv. All six settings of a species must have searched
     the same chunk files.
  5. Memory and time per (species, setting) come from the trace: the largest peak_rss and
     the longest realtime over that species' chunks. Ask = 1.2 x the peak, rounded up to
     5 GB; time = 1.5 x the longest search + 30 min for reading the output, rounded up to
     30 min, at least 1 h.
Python 3.6 or later, standard library only.
"""
import csv
import decimal
import glob
import hashlib
import math
import os
import re
import sys

SIX_SETTINGS = [("hp_lehninger2", 17), ("hp_lehninger2", 25), ("hp_lehninger_c_nonpolar2", 19),
                ("hp_pbotc_1st_ed2", 19), ("hp_pbotc_1st_ed2", 21), ("hp_pbotc_1st_ed2", 23)]
SPECIES = ["human", "worm", "yeast", "borrelia", "botryllus"]
CLADE = {"human": "Mammalia", "worm": "Chromadorea", "yeast": "Saccharomycetes",
         "borrelia": "Spirochaetia", "botryllus": "Ascidiacea"}
KINDS = ["dipeptide", "window10"]
RESULTS = "/scratch/users/olgabot/invertebrate-dark-set/results-0.4"
# The flags main.nf's nextflow.config gave every search of this run (read back from each
# task's .command.sh below, so a wrong value here stops the script rather than slipping by).
FLAGS = {"threshold": "0.0", "min_shared_kmers": "2", "max_query_pvalue": "0.05",
         "min_region_score": "1.3", "chain_max_gap": "30", "chain_max_shift": "10",
         "xdrop_per_penalty": "4", "max_region_evalue": "10000"}


def penalty_string(c):
    """main.nf penaltyString: String.format('%.2f') then trailing zeros and point removed."""
    s = str(decimal.Decimal(repr(c)).quantize(decimal.Decimal("0.01"), rounding=decimal.ROUND_HALF_UP))
    return s.rstrip("0").rstrip(".") if "." in s else s


def kappa_table(path):
    out = {}
    for line in open(path):
        if line.startswith("#") or line.startswith("alphabet") or not line.strip():
            continue
        f = line.rstrip("\n").split("\t")
        out[f[0]] = float(f[2])
    return out


def optimal_penalty(alphabet, kappas):
    kappa = kappas[alphabet]
    classes = int(re.search(r"(\d+)$", alphabet).group(1))
    f = 1.0 / classes
    return penalty_string(-math.log(1.0 - kappa) / math.log(1.0 + kappa / f - kappa))


def xdrop_for(penalty):
    return penalty_string(float(FLAGS["xdrop_per_penalty"]) * float(penalty))


def search_command(alphabet, k, penalty, query, index):
    """The kmerseek command of one search, as main.nf's kmerseekSearch writes it."""
    return ("kmerseek search --alphabet {a} --ksize {k} --query {q} --target {t} "
            "--extend-mismatch-penalty {c} --extend-xdrop {x} --ka-queries 0 "
            "--chain-max-gap {g} --chain-max-shift {s} --threshold {th} --min-shared-kmers {m} "
            "--max-query-pvalue {p} --min-region-score {r}").format(
        a=alphabet, k=k, q=query, t=index, c=penalty, x=xdrop_for(penalty), g=FLAGS["chain_max_gap"],
        s=FLAGS["chain_max_shift"], th=FLAGS["threshold"], m=FLAGS["min_shared_kmers"],
        p=FLAGS["max_query_pvalue"], r=FLAGS["min_region_score"])


def command_from_script(text):
    """The kmerseek search command of a .command.sh, joined onto one line, single-spaced."""
    m = re.search(r"kmerseek search.*?2>", text, re.S)
    if not m:
        raise ValueError("no kmerseek search in .command.sh")
    return " ".join(m.group(0)[:-2].replace("\\\n", " ").split())


def awk_filter_from_script(text):
    m = re.search(r"\| (awk -F, -v max=\S+ '.*?') \\\n", text, re.S)
    if not m:
        raise ValueError("no region E-value awk filter in .command.sh")
    return m.group(1)


def secs(s):
    tot = 0.0
    for v, u in re.findall(r"([\d.]+)(ms|d|h|m|s)", s):
        tot += float(v) * {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}[u]
    return tot


def gigabytes(s):
    v, u = s.split()
    return float(v) * {"KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1, "TB": 1024}[u]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fasta_counts(path):
    n = res = 0
    for line in open(path):
        if line.startswith(">"):
            n += 1
        else:
            res += len(line.strip())
    return n, res


def main(run_dir, repo_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    # 1. the six settings notebook 279 sections 7 and 9 read
    names = set()
    for sp in SPECIES:
        for f in glob.glob(os.path.join(RESULTS, "protein-min-evalue", sp, "*.csv.gz")):
            m = re.match(r"chunk_\d+\.(.+)\.k(\d+)\.s(\d+)\.lc(\w+)\.(extend-c\d+(?:\.\d+)?)\.csv\.gz$", os.path.basename(f))
            names.add((m.group(1), int(m.group(2)), m.group(3), m.group(4), m.group(5)))
    read = sorted(set((a, k) for a, k, s, lc, ext in names))
    if read != sorted(SIX_SETTINGS) or set((s, lc) for _, _, s, lc, _ in names) != {("1", "false")}:
        sys.exit("protein-min-evalue holds {} (scaled, mask: {}), not the six settings".format(
            read, sorted(set((s, lc) for _, _, s, lc, _ in names))))
    kappas = kappa_table(os.path.join(repo_dir, "assets", "kappa_by_alphabet.tsv"))
    penalty = {a: optimal_penalty(a, kappas) for a, _ in SIX_SETTINGS}
    for a, k, s, lc, ext in names:
        if ext != "extend-c" + penalty[a]:
            sys.exit("{} k{}: file says {}, optimalPenalty gives {}".format(a, k, ext, penalty[a]))

    # 3-5. the run's tasks for these settings
    trace = list(csv.DictReader(open(os.path.join(run_dir, "trace.ladder-0.4-midi.txt")), delimiter="\t"))
    want = {"{}.k{}.s1.lcfalse.extend-c{}".format(a, k, penalty[a]): (a, k) for a, k in SIX_SETTINGS}
    pat = re.compile(r"kmerseekSearch \((\w+)\.(chunk_\d+)\.(.+)\)$")
    rows, checks, chunk_of = [], [], {}
    awk_filters = set()
    for t in trace:
        m = pat.search(t["name"])
        if not m or m.group(3) not in want:
            continue
        sp, chunk, slug = m.groups()
        a, k = want[slug]
        if t["status"] not in ("COMPLETED", "CACHED") or t["exit"] != "0":
            sys.exit("{}: status {} exit {}".format(t["name"], t["status"], t["exit"]))
        wd = glob.glob(os.path.join(run_dir, "work", t["hash"] + "*"))
        if len(wd) != 1:
            sys.exit("{}: {} work dirs for {}".format(t["name"], len(wd), t["hash"]))
        wd = wd[0]
        text = open(os.path.join(wd, ".command.sh")).read()
        awk_filters.add(awk_filter_from_script(text))
        chunk_path = os.path.realpath(os.path.join(wd, chunk + ".fasta"))
        index_name = "minus_{}.{}.k{}.lcfalse.kmerseek.rocksdb".format(CLADE[sp], a, k)
        index_real = os.path.realpath(os.path.join(wd, index_name))
        index_store = os.path.join(RESULTS, "kmerseek_index", index_name)
        if index_real != os.path.realpath(index_store):
            sys.exit("{}: index {} is not the stored {}".format(t["name"], index_real, index_store))
        ran = command_from_script(text)
        ours = search_command(a, k, penalty[a], chunk + ".fasta", index_name)
        checks.append([sp, chunk, a, k, wd, "same" if ran == ours else "DIFFERENT", ran, ours])
        if ran != ours:
            sys.exit("{}: the run's command differs from ours\n run: {}\nours: {}".format(t["name"], ran, ours))
        key = (sp, chunk)
        if chunk_of.setdefault(key, chunk_path) != chunk_path:
            sys.exit("{} {}: two different chunk files across settings".format(sp, chunk))
        rows.append({"species": sp, "chunk": chunk, "alphabet": a, "ksize": k, "penalty": penalty[a],
                     "peak_gb": gigabytes(t["peak_rss"]), "hours": secs(t["realtime"]) / 3600,
                     "run_regions": os.path.join(RESULTS, sp, "kmerseek", "{}.{}.regions.csv.zst".format(chunk, slug)),
                     "index": index_store})
    if len(awk_filters) != 1:
        sys.exit("the run used {} different region E-value filters".format(len(awk_filters)))
    expect = {(sp, a, k) for sp in SPECIES for a, k in SIX_SETTINGS}
    if {(r["species"], r["alphabet"], r["ksize"]) for r in rows} != expect:
        sys.exit("not every species has every setting in the trace")
    for r in rows:
        if not os.path.exists(r["run_regions"]):
            sys.exit("missing run region file " + r["run_regions"])
        if not os.path.isdir(r["index"]):
            sys.exit("missing index " + r["index"])

    with open(os.path.join(out_dir, "awk_filter.txt"), "w") as fh:
        fh.write(awk_filters.pop() + "\n")
    with open(os.path.join(out_dir, "command_check.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "chunk", "alphabet", "ksize", "work_dir", "check", "run_command", "our_command"])
        w.writerows(sorted(checks))

    chunks = []
    for (sp, chunk), path in sorted(chunk_of.items(), key=lambda x: (SPECIES.index(x[0][0]), x[0][1])):
        n, res = fasta_counts(path)
        chunks.append([sp, CLADE[sp], chunk, path, md5_of(path), n, res])
    with open(os.path.join(out_dir, "chunks.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "clade", "chunk", "path", "md5", "proteins", "residues"])
        w.writerows(chunks)

    res = {}
    for r in rows:
        g = res.setdefault((r["species"], r["alphabet"], r["ksize"]), {"peak": 0, "hours": 0, "chunks": 0, "sum_h": 0})
        g["peak"] = max(g["peak"], r["peak_gb"])
        g["hours"] = max(g["hours"], r["hours"])
        g["chunks"] += 1
        g["sum_h"] += r["hours"]
    with open(os.path.join(out_dir, "resources.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "alphabet", "ksize", "chunks", "run_max_peak_gb", "run_max_hours",
                    "run_sum_hours", "mem_gb", "time_h"])
        for (sp, a, k), g in sorted(res.items(), key=lambda x: (SIX_SETTINGS.index(x[0][1:]), SPECIES.index(x[0][0]))):
            mem = 5 * math.ceil(1.2 * g["peak"] / 5)
            hours = max(1.0, math.ceil((1.5 * g["hours"] + 0.5) * 2) / 2)
            w.writerow([sp, a, k, g["chunks"], round(g["peak"], 1), round(g["hours"], 2), round(g["sum_h"], 2), mem, hours])

    with open(os.path.join(out_dir, "tasks.tsv"), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["species", "clade", "chunk", "kind", "alphabet", "ksize", "penalty", "xdrop", "index", "run_regions"])
        for r in sorted(rows, key=lambda r: (SIX_SETTINGS.index((r["alphabet"], r["ksize"])), SPECIES.index(r["species"]), r["chunk"])):
            for kind in KINDS + ["real"]:
                w.writerow([r["species"], CLADE[r["species"]], r["chunk"], kind, r["alphabet"], r["ksize"],
                            r["penalty"], xdrop_for(r["penalty"]), r["index"], r["run_regions"]])
    print("settings and penalties:", {"{} k{}".format(a, k): (penalty[a], xdrop_for(penalty[a])) for a, k in SIX_SETTINGS})
    print("{} run tasks checked, every command the same; {} chunks; awk filter: one".format(len(checks), len(chunks)))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
