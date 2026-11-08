"""Compare each probe set's change after training in GSE27536 and GSE28998, both on U133 Plus 2.

Usage: series_agreement.py <GSE28998.probe_set_values.tsv.gz> <GSE27536.probe_set_values.tsv.gz> <groups.tsv>
       <series folder> <notebooks folder> <out folder>
Values are hp_kmer_table.py's (the authors' RMA, log2). GSE28998's change is each person's trained arm minus their
untrained arm, averaged over people; GSE27536's is the mean after training minus the mean before. Writes
series_agreement.tsv (one row per probe set), series_agreement_summary.tsv, and series_agreement.png and .svg.
"""

import csv
import sys

import numpy as np

from geo2kmers.people import person_of, sample_fields

first, second, groups_file, series, notebooks, out = sys.argv[1:]
sys.path.insert(0, notebooks)
import pubfig as pf  # noqa: E402

with open(groups_file) as f:
    post_of = {r["sample"]: r["group"] == "1" for r in csv.DictReader(f, delimiter="\t") if r["comparison"] == "main"}


def read(path):
    import gzip

    with gzip.open(path, "rt") as f:
        samples = f.readline().rstrip("\n").split("\t")[1:]
        rows = {line.split("\t", 1)[0]: line.rstrip("\n").split("\t")[1:] for line in f}
    keep = [i for i, s in enumerate(samples) if s in post_of]
    return [samples[i] for i in keep], {k: np.array([float(v[i]) for i in keep]) for k, v in rows.items()}


def change(gse, path):
    """Each probe set's mean within-person change, post minus pre, and how many people."""
    samples, values = read(path)
    fields = sample_fields(series, gse)
    people = [person_of(gse, fields[s]) for s in samples]
    pairs = []
    for person in sorted(set(people)):
        pre = [i for i, s in enumerate(samples) if people[i] == person and not post_of[s]]
        post = [i for i, s in enumerate(samples) if people[i] == person and post_of[s]]
        if pre and post:
            pairs.append((pre, post))
    names = sorted(values)
    table = np.array([values[n] for n in names])
    each = np.array([table[:, post].mean(1) - table[:, pre].mean(1) for pre, post in pairs]).T
    return names, each.mean(1), each, len(pairs)


names_a, change_a, each_a, people_a = change("GSE28998", first)
names_b, change_b, _, people_b = change("GSE27536", second)
assert names_a == names_b
up_b = change_b > 0
r = np.corrcoef(change_a, change_b)[0, 1]
summary = {
    "probe_sets": len(names_a),
    "people_GSE28998": people_a,
    "people_GSE27536": people_b,
    "pearson_r_of_changes": f"{r:.3f}",
    "share_up_in_GSE27536": f"{up_b.mean():.3f}",
    "GSE28998_mean_change_where_GSE27536_up": f"{change_a[up_b].mean():.4f}",
    "GSE28998_mean_change_where_GSE27536_down": f"{change_a[~up_b].mean():.4f}",
    "GSE28998_share_higher_in_trained_arm_by_person": ",".join(f"{v:.2f}" for v in (each_a > 0).mean(0)),
    "GSE28998_mean_change_over_all_probe_sets_by_person": ",".join(f"{v:.3f}" for v in each_a.mean(0)),
}
with open(f"{out}/series_agreement_summary.tsv", "w") as f:
    f.write("\t".join(summary) + "\n" + "\t".join(map(str, summary.values())) + "\n")
with open(f"{out}/series_agreement.tsv", "w") as f:
    f.write("probe_set\tchange_GSE28998\tchange_GSE27536\n")
    f.writelines(f"{n}\t{a:.4f}\t{b:.4f}\n" for n, a, b in zip(names_a, change_a, change_b, strict=True))

pf.use_style()
fig, ax = pf.figure(pf.ONE_COLUMN_MM, 80)
ax.plot(change_b, change_a, ".", color=pf.GREY, markersize=1, alpha=0.3, rasterized=True, label="One probe set")
slope, intercept = np.polyfit(change_b, change_a, 1)
x = np.array([change_b.min(), change_b.max()])
ax.plot(x, slope * x + intercept, color=pf.OKABE_ITO["blue"], label=f"Straight-line fit, Pearson r {r:.2f}")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
ax.set_xlabel(f"GSE27536 (thigh, cycling): after 8 weeks\nminus before, log2, mean of {people_b} people")
ax.set_ylabel(f"GSE28998 (biceps, weights): trained arm\nminus untrained arm, log2, mean of {people_a} people")
pf.shared_legend(fig, ncol=1)
pf.save(fig, f"{out}/series_agreement")
print(summary)
