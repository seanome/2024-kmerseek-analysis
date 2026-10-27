"""Compact exports for the demo page, read only from result files.

For every depth that has analysis/<depth>/complete, writes in --out:
  demo.json            everything the page draws (per-depth table, events,
                       volcano points, alphabet table, timing, trimming)
  events_<depth>.tsv   one row per exported event (sequence, counts, grade, best hits)
  events_<depth>.fasta DNA of every exported event's sequences
  peptides_<depth>.fasta peptides with at least one protein hit
Which events are exported (fixed here, applied to both methods the same way):
  the 150 top-ranked units of each method, then up to 40 per method from
  each of: no exact 31-mer and no close BLAST match; no exact 31-mer but a
  near copy (likely another allele); words in several genes; published DE
  gene recovered by this method but not by kallisto. Only units that went
  through the protein search are exported.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd

SAMPLES = ["DW1", "DW2", "DW3", "ZW1", "ZW2", "ZW3"]
COND = {s: ("wild" if s.startswith("D") else "breed") for s in SAMPLES}
TOP, PER_GROUP = 150, 40


def jload(p):
    return json.loads(Path(p).read_text())


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def scrub(x):
    """NaN and infinity become null, everywhere in a nested structure."""
    if isinstance(x, dict):
        return {k: scrub(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scrub(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    return clean(x)


def load_annotation(path):
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["seq_id"]] = r
    return out


def compact_hits(fragments, per_setting=3):
    """Keep the best few hits per database, alphabet and scaled value."""
    hits = []
    for fr in fragments:
        for h in fr["hits"]:
            if h["rank"] <= per_setting:
                hits.append(dict(db=h["db"], mode=h["mode"], scaled=h["scaled"], rank=h["rank"],
                                 protein=h["protein_id"], desc=h["description"].split(" ", 1)[-1],
                                 org=h["organism"], containment=h["containment"], shared=h["shared_hashes"],
                                 qh=h["query_hashes"], words=h["shared_encoded_kmers"], tie=h["tying_best"],
                                 pos=h["matched_positions"], pep=fr["peptide"], strand=fr["strand"],
                                 frame=fr["frame"], nt=[fr["nt_start"], fr["nt_end"]]))
    return hits


def depth_block(root, depth, protein_gene, published):
    w = root / "analysis" / depth
    comp = jload(w / "comparison.json")
    sel = jload(w / "review_selection.json")
    anchors = pd.read_csv(w / "splash_anchors_graded.tsv", sep="\t", dtype={"gene": str})
    contigs = pd.read_csv(w / "dekupl_contigs_graded.tsv", sep="\t", dtype={"gene": str})
    sp_seqs = pd.read_csv(w / "splash_sequences.tsv", sep="\t")
    genes = pd.read_csv(w / "gene_table.tsv", sep="\t", dtype=str).fillna("")
    kal = pd.read_csv(root / "baseline" / depth / "de" / "de_results.tsv", sep="\t", dtype={"feature": str})
    kal = kal.set_index("feature")
    near = pd.read_csv(w / "near_reference.tsv", sep="\t", dtype={"gene": str}).set_index("seq_id")
    rec_by = dict(zip(genes.gene, genes.published_recovered_by))
    ann = {"splash": load_annotation(w / "annotation_splash.jsonl"),
           "dekupl": load_annotation(w / "annotation_dekupl.jsonl")}
    seqs_by_group = defaultdict(list)
    for r in sp_seqs.itertuples():
        seqs_by_group[r.group_id].append((r.seq_id, r.dna, r.sequence_kind))

    def kallisto_for(gene):
        if not gene or gene not in kal.index:
            return None
        r = kal.loc[gene]
        return dict(log2fc=clean(round(float(r.log2FoldChange), 3)), padj=clean(float(r.padj)),
                    significant=bool(r.significant))

    def event(method, row):
        if method == "splash":
            uid, ids = row.group_id, [s[0] for s in seqs_by_group[row.group_id]]
            seqs = [dict(id=s[0], dna=s[1], kind=s[2]) for s in seqs_by_group[row.group_id]]
            counts = {s: int(getattr(row, f"count_{s}")) for s in SAMPLES}
            per_m = {s: round(float(getattr(row, f"per_million_{s}")), 2) for s in SAMPLES}
            stat = dict(q=clean(float(row.pval_opt_corrected)), effect_size_bin=clean(float(row.effect_size_bin)),
                        log2_descriptive=round(float(row.log2_breed_over_wild_descriptive), 3),
                        targets=int(row.anch_uniqTargs), M=int(row.M))
            direction = row.direction_descriptive
            seed = row.anchor
        else:
            uid, ids = row.contig_id, [row.contig_id]
            seqs = [dict(id=row.contig_id, dna=row.contig, kind="DE-kupl contig")]
            counts = {s: round(float(getattr(row, s)), 2) for s in SAMPLES}
            per_m = None
            stat = dict(padj=clean(float(row.pvalue)), log2fc=round(float(row.log2FC), 3),
                        kmers=int(row.nb_merged_kmers), meanA=round(float(row.meanA), 2),
                        meanB=round(float(row.meanB), 2))
            direction = row.direction
            seed = row.tag
        frags = [f for i in ids if i in ann[method] for f in ann[method][i]["fragments"]]
        hits = compact_hits(frags)
        for h in hits:
            h["gene"] = protein_gene.get(h["protein"], {}).get("gene_id", "") if h["db"] == "oyster_v9" else ""
        gene = row.gene if isinstance(row.gene, str) else ""
        nr = None
        for i in ids:
            if i in near.index and isinstance(near.loc[i, "transcript"], str):
                r = near.loc[i]
                cand = dict(cls=r["class"], transcript=r.transcript, gene=r.gene, identity=float(r.identity),
                            aligned=int(r.aligned_length), qlen=int(r.query_length), mismatches=int(r.mismatches))
                if nr is None or cand["identity"] > nr["identity"]:
                    nr = cand
            elif i in near.index and nr is None:
                nr = dict(cls=near.loc[i, "class"])
        pub = published.get(gene)
        return dict(method=method, id=uid, rank=int(row.rank), seed=seed, direction=direction, stat=stat,
                    counts=counts, per_million=per_m, status=row.status, gene=gene,
                    genes_hit=[g for g in str(row.genes_hit).split(";") if g and g != "nan"],
                    near=nr, published=pub, recovered_by=rec_by.get(gene, ""), kallisto=kallisto_for(gene),
                    sequences=seqs, peptides=len(frags), hits=hits)

    events, flags = [], []
    for method, df, idcol in (("splash", anchors, "group_id"), ("dekupl", contigs, "contig_id")):
        searched = set(ann[method]) if method == "dekupl" else {g for g, s in seqs_by_group.items()
                                                                 if any(x[0] in ann[method] for x in s)}
        df = df[df[idcol].isin(searched)].sort_values("rank")
        groups = [("top", df.head(TOP))]
        nr_cls = df[idcol].map(lambda u: None)
        if "near_reference" in df.columns:
            nr_cls = df["near_reference"]
        groups.append(("absent_no_close_match", df[(df.status == "absent") & (nr_cls == "no_close_match")].head(PER_GROUP)))
        groups.append(("absent_near_copy", df[(df.status == "absent") & (nr_cls == "near_copy")].head(PER_GROUP)))
        groups.append(("multi_gene", df[df.status == "multi_gene"].head(PER_GROUP)))
        mine = "dekupl" if method == "dekupl" else "splash"
        only = df[df.gene.map(lambda g: mine in rec_by.get(g, "") and "kallisto" not in rec_by.get(g, ""))]
        groups.append(("published_not_kallisto", only.drop_duplicates("gene").head(PER_GROUP)))
        seen = set()
        for tag, sub in groups:
            for row in sub.itertuples(index=False):
                uid = getattr(row, idcol)
                if uid in seen:
                    events[[e["id"] for e in events].index(uid)]["tags"].append(tag)
                    continue
                seen.add(uid)
                e = event(method, row)
                e["tags"] = [tag]
                events.append(e)

    # Volcano: DE-kupl contigs and SPLASH anchors, downsampled to keep the page small.
    def thin(df, n):
        keep = df[df.status.isin(["absent", "multi_gene"]) | df.gene.isin(published)]
        rest = df.drop(keep.index)
        return pd.concat([keep.head(n // 2), rest.sample(min(len(rest), n - min(len(keep), n // 2)), random_state=0)])

    dk = thin(contigs, 4000)
    sp = thin(anchors, 4000)
    volcano = dict(
        dekupl=[[r.contig_id, round(r.log2FC, 2), round(-math.log10(max(r.pvalue, 1e-300)), 2), r.status,
                 int(r.gene in published)] for r in dk.itertuples()],
        splash=[[r.group_id, round(r.log2_breed_over_wild_descriptive, 2),
                 round(-math.log10(max(r.pval_opt_corrected, 1e-300)), 2), r.status, int(r.gene in published)]
                for r in sp.itertuples()],
        dekupl_total=len(contigs), splash_total=len(anchors))

    # Alphabet table: units with any hit per setting and database, and how often
    # the best oyster protein's gene equals the strict DNA gene.
    alpha = []
    for method in ("splash", "dekupl"):
        unit_of = {}
        if method == "splash":
            for g, s in seqs_by_group.items():
                for x in s:
                    unit_of[x[0]] = g
        strict_gene = (dict(zip(anchors.group_id, anchors.gene.where(anchors.status == "strict")))
                       if method == "splash" else dict(zip(contigs.contig_id, contigs.gene.where(contigs.status == "strict"))))
        best = defaultdict(dict)
        searched_units = set()
        for sid, r in ann[method].items():
            u = unit_of.get(sid, sid)
            searched_units.add(u)
            for fr in r["fragments"]:
                for h in fr["hits"]:
                    if h["rank"] != 1:
                        continue
                    key = (h["db"], h["mode"], h["scaled"])
                    old = best[key].get(u)
                    if old is None or (h["containment"], h["shared_encoded_kmers"]) > (old["containment"], old["shared_encoded_kmers"]):
                        best[key][u] = h
        for (db, mode, scaled), units in sorted(best.items()):
            agree = sum(1 for u, h in units.items() if isinstance(strict_gene.get(u), str)
                        and protein_gene.get(h["protein_id"], {}).get("gene_id") == strict_gene.get(u))
            with_strict = sum(1 for u in units if isinstance(strict_gene.get(u), str))
            alpha.append(dict(method=method, db=db, mode=mode, scaled=scaled, searched=len(searched_units),
                              with_hit=len(units), with_hit_and_strict_gene=with_strict,
                              best_oyster_protein_same_gene=agree if db == "oyster_v9" else None))
    trims = {s: jload(root / "trimmed" / depth / f"{s}.validated.json") for s in SAMPLES}
    return dict(comparison=comp, review_selection=sel, events=events, volcano=volcano, alphabet=alpha,
                trimming=trims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    protein_gene = jload(Path(a.code) / "sources" / "protein_gene_map.json")
    published = {str(r["gene_id"]): dict(log2fc=r["log2fc"], fdr=r["fdr"], desc=r["description"])
                 for r in jload(Path(a.code) / "sources" / "published_degs.json")}
    demo = dict(samples=[dict(sample=s, condition=COND[s]) for s in SAMPLES], depths={},
                runs=pd.read_csv(Path(a.code) / "sources" / "oyster_runs.tsv", sep="\t")
                .set_index("sample_alias")[["run_accession", "read_count"]].to_dict("index"),
                published_total=len(published), indexes={n: jload(root / "indexes" / f"{n}.meta.json")
                                                         for n in ("oyster_v9", "swissprot")})
    for depth in ("1m", "5m", "full"):
        if not (root / "analysis" / depth / "complete").exists():
            continue
        block = depth_block(root, depth, protein_gene, published)
        demo["depths"][depth] = block
        rows = []
        with open(out / f"events_{depth}.fasta", "w") as fa, open(out / f"peptides_{depth}.fasta", "w") as fp:
            peps = set()
            for e in block["events"]:
                for s in e["sequences"]:
                    fa.write(f">{s['id']} {e['method']} {e['status']} {e['direction']}\n{s['dna']}\n")
                for h in e["hits"]:
                    if h["pep"] not in peps:
                        peps.add(h["pep"])
                        fp.write(f">{e['id']}|{h['strand']}{h['frame']}|nt{h['nt'][0]}-{h['nt'][1]}\n{h['pep']}\n")
                top = {}
                for h in e["hits"]:
                    if h["mode"] == "protein" and h["scaled"] == 10 and h["rank"] == 1:
                        top[h["db"]] = f"{h['protein']} {h['desc']} ({h['words']} words, containment {h['containment']})"
                rows.append(dict(method=e["method"], id=e["id"], rank=e["rank"], tags=",".join(e["tags"]),
                                 direction=e["direction"], **{f"stat_{k}": v for k, v in e["stat"].items()},
                                 **{f"count_{s}": e["counts"][s] for s in SAMPLES}, grade=e["status"], gene=e["gene"],
                                 near_reference=(e["near"] or {}).get("cls", ""), published=bool(e["published"]),
                                 recovered_by=e["recovered_by"], best_oyster_protein_k7_s10=top.get("oyster_v9", ""),
                                 best_swissprot_k7_s10=top.get("swissprot", ""),
                                 sequences=";".join(s["dna"] for s in e["sequences"])))
        pd.DataFrame(rows).to_csv(out / f"events_{depth}.tsv", sep="\t", index=False)
    timing = root / "timing" / "speed.json"
    demo["timing"] = jload(timing) if timing.exists() else None
    stages = root / "timing" / "stages.tsv"
    demo["stages"] = pd.read_csv(stages, sep="\t").to_dict("records") if stages.exists() else None
    (out / "demo.json").write_text(json.dumps(scrub(demo), separators=(",", ":"), allow_nan=False))
    print({d: len(b["events"]) for d, b in demo["depths"].items()}, (out / "demo.json").stat().st_size)


if __name__ == "__main__":
    main()
