import polars as pl, math
d = pl.read_csv("/Users/olga/code/2024-kmerseek-analysis-280/nextflow-runs/time-split-pfam/tables/pilot_partial_2026-10-14_tfidf/per_setting_found.csv")
km = d.filter(pl.col("tool") == "kmerseek")
rows = [("exact, Poisson score", km.filter((pl.col("exact_score") == "poisson") & (pl.col("arm") == "exact"))),
        ("exact, tf-idf score", km.filter((pl.col("exact_score") == "tfidf") & (pl.col("arm") == "exact"))),
        ("extend, E-value", km.filter((pl.col("exact_score") == "tfidf") & (pl.col("arm") == "extend")))]
comp = d.filter((pl.col("exact_score") == "tfidf") & (pl.col("tool") != "kmerseek"))
X0, X1 = 222, 640
SC = (X1 - X0) / 115
def x(v): return X0 + v * SC
def r(n): return 2.0 * math.sqrt(n)
out = []
out.append('<text class="th" x="40" y="26">Tune-half instances each search setting finds at its own decoy-matched cutoff</text>')
out.append('<text class="ts" x="40" y="44">nb 280 pilot: 115 tune-half Pfam instances, domains placed through the hit, MMseqs2 one round.</text>')
ly = 72
out.append(f'<circle cx="46" cy="{ly-4}" r="{r(1):.1f}" fill="#7F77DD" fill-opacity="0.75"/><circle cx="60" cy="{ly-4}" r="{r(10):.1f}" fill="#7F77DD" fill-opacity="0.75"/><circle cx="86" cy="{ly-4}" r="{r(50):.1f}" fill="#7F77DD" fill-opacity="0.75"/>')
out.append(f'<text class="ts" x="108" y="{ly}">kmerseek settings at that count (area: 1, 10, 50 settings)</text>')
out.append(f'<line x1="440" y1="{ly-11}" x2="440" y2="{ly+2}" stroke="#0F6E56" stroke-width="2"/><text class="ts" x="448" y="{ly}">another tool</text>')
out.append(f'<text class="ts" x="40" y="{ly+24}">More is better. 0 means the setting\'s cutoff lets nothing through.</text>')
top, rowh = 160, 62
y_top, y_bot = top - 26, top + rowh * 2 + 24
labs = {"MMseqs2": ("MMseqs2", "end", -4, 0), "hmmscan Pfam 33.1": ("hmmscan", "middle", 0, -13), "phmmer": ("phmmer", "start", 4, 0)}
for name, (short, anchor, dx, dy) in labs.items():
    v = comp.filter(pl.col("setting") == name)["found"][0]
    out.append(f'<line x1="{x(v):.1f}" y1="{y_top}" x2="{x(v):.1f}" y2="{y_bot}" stroke="#0F6E56" stroke-width="1.5"/>')
    out.append(f'<text class="ts" x="{x(v)+dx:.1f}" y="{y_top-6+dy}" text-anchor="{anchor}">{short} {v}</text>')
for i, (lab, g) in enumerate(rows):
    y = top + i * rowh
    out.append(f'<line x1="{X0}" y1="{y}" x2="{X1}" y2="{y}" stroke="var(--border)" stroke-width="0.5"/>')
    out.append(f'<text class="th" x="{X0-30}" y="{y+1}" text-anchor="end">{lab}</text>')
    zero = g.filter(pl.col("found") == 0).height
    out.append(f'<text class="ts" x="{X0-30}" y="{y+17}" text-anchor="end">{g.height} settings, {zero} find 0</text>')
    for v, n in g.group_by("found").len().sort("found").rows():
        out.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="{r(n):.1f}" fill="#7F77DD" fill-opacity="0.75"><title>{n} settings find {v}</title></circle>')
    mx = g["found"].max()
    out.append(f'<text class="ts" x="{x(mx)+8:.1f}" y="{y-9}">best {mx}</text>')
ay = top + rowh * 2 + 40
out.append(f'<line x1="{X0}" y1="{ay}" x2="{X1}" y2="{ay}" stroke="var(--text-secondary)" stroke-width="0.5"/>')
for v in (0, 20, 40, 60, 80, 100, 115):
    out.append(f'<line x1="{x(v):.1f}" y1="{ay}" x2="{x(v):.1f}" y2="{ay+4}" stroke="var(--text-secondary)" stroke-width="0.5"/><text class="ts" x="{x(v):.1f}" y="{ay+17}" text-anchor="middle">{v}</text>')
out.append(f'<text class="ts" x="{x(57.5):.1f}" y="{ay+34}" text-anchor="middle">tune-half instances found (of 115)</text>')
out.append(f'<text class="ts" x="40" y="{ay+60}">tf-idf gives every exact setting a working cutoff; with the Poisson score 102 of 107 found nothing.</text>')
out.append(f'<text class="ts" x="40" y="{ay+78}">Best kmerseek setting: 24 instances. Other tools: 54 to 68. No instance is found by kmerseek alone.</text>')
H = ay + 96
svg = (f'<svg width="100%" viewBox="0 0 680 {H}" role="img"><title>Instances found per setting</title><desc>Dot plot of tune-half Pfam instances each kmerseek setting finds at its matched cutoff, for exact with Poisson score, exact with tf-idf score, and extend, against MMseqs2, hmmscan and phmmer.</desc>'
       + "".join(out) + "</svg>")
open("found_per_setting.svg", "w").write(svg)
print("H", H, "x(68)", round(x(68)), "bubble r at 0 for 102:", round(r(102), 1), "label right edge", X0 - 30)
for lab, g in rows: print(lab, g.group_by("found").len().sort("len", descending=True).head(2).rows())
