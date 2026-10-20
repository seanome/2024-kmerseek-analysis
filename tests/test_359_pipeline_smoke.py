"""End-to-end run of notebook 359's build, reduce and score scripts on three hand-built
genes, each with a cassette exon (exon 2, skipped by a second isoform that spans it):

* G1: nothing on exon 2 -> the one blank exon.
* G2: an InterProScan Pfam hit on exon 2 -> not blank.
* G3: an IsoAnnotLite PFAM DOMAIN on exon 2 -> not blank; InterProScan recovers it, so
  the sanity gate is 1 of 1.

Then two settings' calls: setting s1 lands on G1's exon 2 in the real protein and not in
the decoy; s2 lands on both, so its 1% threshold (cap 0 for one exon) excludes it.
"""

import gzip
import json
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "scripts" / "359_isoform_blank_exons"
PY = sys.executable


def write_inputs(d: Path):
    qc, flt = d / "qc", d / "filter"
    qc.mkdir()
    flt.mkdir()
    rng = random.Random(0)
    aa = "ACDEFGHIKLMNPQRSTVWY"
    rows, gtf, gff, ips, prots = [], [], [], [], {}
    for g, chrom in (("G1", "chr1"), ("G2", "chr2"), ("G3", "chr3")):
        prot = "M" + "".join(rng.choice(aa) for _ in range(68))  # 69 aa, 210 nt with stop
        for iso, exons in ((f"{g}.a", [(101, 160), (201, 260), (301, 390)]), (f"{g}.b", [(101, 160), (301, 390)])):
            coding = iso.endswith(".a")
            rows.append(
                {
                    "isoform": iso, "chrom": chrom, "strand": "+", "associated_gene": g,
                    "coding": "coding" if coding else "non_coding",
                    "CDS_start": "1.0" if coding else "NA", "CDS_end": "210" if coding else "NA",
                    "ORF_seq": prot if coding else "NA",
                }
            )  # fmt: skip
            for a, b in exons:
                gtf.append(f'{chrom}\tPacBio\texon\t{a}\t{b}\t.\t+\t.\tgene_id "{g}"; transcript_id "{iso}";')
        prots[f"{g}.a"] = prot
    # exon 2 = transcript bases 60..119 = codons 20..39 (0-based), residues 21..40 1-based
    ips.append("G2.a\tmd5\t69\tPfam\tPF00001\tdesc\t25\t35\t1e-5\tT\tdate")
    ips.append("G3.a\tmd5\t69\tPfam\tPF00002\tdesc\t22\t40\t1e-5\tT\tdate")
    gff.append("G3.a\tPFAM\tDOMAIN\t21\t40\t.\t+\t.\tID=PF00002; Name=x; Desc=x; PosType=P")
    gff.append("G1.a\ttappAS\ttranscript\t1\t210\t.\t+\t.\tID=G1; PosType=T")
    gff.append("G1.a\tNMD\tNMD\t1\t210\t.\t+\t.\tID=NMD; PosType=T")
    pl.DataFrame(rows).write_csv(qc / "x_classification.txt", separator="\t")
    (qc / "x_corrected.gtf").write_text("\n".join(gtf) + "\n")
    (qc / "x.gff3").write_text("\n".join(gff) + "\n")
    (flt / "x_pass_isoforms.txt").write_text("\n".join(r["isoform"] for r in rows) + "\n")
    (d / "pfam.tsv").write_text("\n".join(ips) + "\n")
    with gzip.open(d / "gencode.gtf.gz", "wt") as fh:
        fh.write('chr1\tHAVANA\texon\t201\t260\t.\t+\t.\ttranscript_id "T1"; tag "basic"; tag "CCDS";\n')
    return prots


def test_build_reduce_score(tmp_path):
    prots = write_inputs(tmp_path)
    build = tmp_path / "build"
    subprocess.run(
        [PY, S / "build_exons.py", "--sqanti-dir", tmp_path / "qc", "--filter-dir", tmp_path / "filter",
         "--prefix", "x", "--interproscan", tmp_path / "pfam.tsv", "--gencode-gtf", tmp_path / "gencode.gtf.gz",
         "--seed", "7", "--out-dir", build],
        check=True, capture_output=True,
    )  # fmt: skip
    bl = pl.read_parquet(build / "blank_exons.parquet")
    assert bl["gene"].to_list() == ["G1"]
    b = bl.row(0, named=True)
    assert (b["p_start"], b["p_end"]) == (20, 40)
    assert b["in_basic_ccds_isoform"] is True
    assert b["decoy_protein"][:20] == prots["G1.a"][:20] and b["decoy_protein"][40:] == prots["G1.a"][40:]
    assert sorted(b["decoy_segment"]) == sorted(b["segment"])
    gate = json.loads((build / "gate.json").read_text())
    assert gate["isoannot_pfam_features"] == 1 and gate["recovered_by_overlap"] == 1
    summ = json.loads((build / "build_summary.json").read_text())
    assert summ["alternative_exons"] == 3 and summ["blank_exons"] == 1

    # kmerseek-like CSV: s1 lands on the real exon only; s2 on both real and decoy
    head = "query_name,target_name,region_start,region_end,target_start,target_end,region_evalue,region_subseq,target_subseq\n"
    tgt = "P12345|DOMAIN|SH3|30-80"
    best = tmp_path / "best"
    best.mkdir()
    calls = {
        ("s1", "real"): f"real__G1.a,{tgt},18,36,25,43,1e-6,{'A'*18},{'A'*18}\nreal__G1.a,{tgt},50,60,1,11,1e-9,AAAAAAAAAA,AAAAAAAAAA\n",
        ("s1", "decoy"): f"decoy__{b['exon_id']},{tgt},0,15,0,15,1e-8,{'A'*15},{'A'*15}\n",
        ("s2", "real"): f"real__G1.a,{tgt},20,40,21,41,1e-3,{'A'*20},{'A'*20}\n",
        ("s2", "decoy"): f"decoy__{b['exon_id']},{tgt},20,40,21,41,1e-3,{'A'*20},{'A'*20}\n",
    }  # fmt: skip
    for (s, side), body in calls.items():
        r = subprocess.run(
            [PY, S / "reduce_calls.py", build / "blank_exons.parquet", side],
            input=head + body, text=True, check=True, capture_output=True,
        )
        (best / f"{s}.{side}.best.tsv").write_text(r.stdout)
    red = pl.read_csv(best / "s1.real.best.tsv", separator="\t")
    # the 1e-9 call does not cover half of residues 20..40; the 1e-6 call (18..36) covers 16 of 20
    assert red["evalue"].to_list() == [1e-6]
    assert pl.read_csv(best / "s1.decoy.best.tsv", separator="\t").height == 0

    pl.DataFrame(
        {"name": [tgt], "feature_type": ["DOMAIN"], "description": ["SH3"], "start": [30], "end": [80], "cut_start": [9]}
    ).write_parquet(tmp_path / "regions.parquet")
    (tmp_path / "settings.tsv").write_text("setting\nS1\n".replace("S1", "s1\ns2"))
    out = tmp_path / "score"
    subprocess.run(
        [PY, S / "score_landings.py", "--build-dir", build, "--best-dir", best, "--regions", tmp_path / "regions.parquet",
         "--settings", tmp_path / "settings.tsv", "--example-seed", "359", "--out-dir", out],
        check=True, capture_output=True,
    )  # fmt: skip
    summary = pl.read_csv(out / "summary.csv").filter(pl.col("stratum") == "all")
    s1 = summary.filter(pl.col("setting") == "s1").row(0, named=True)
    s2 = summary.filter(pl.col("setting") == "s2").row(0, named=True)
    assert (s1["real_landings"], s1["decoy_landings"], s1["net_landings"]) == (1, 0, 1)
    # target 25..43 in the entry; the SH3 feature is entry residues 21..72, so it is inside
    assert s1["named"] == 1
    assert (s2["real_landings"], s2["decoy_landings"]) == (0, 0)
    v = json.loads((out / "verdict.json").read_text())
    assert v["n_blank"] == 1 and v["verdict"] in ("PASS", "KILL")

    # the three figures draw from these tables without error
    sys.path.insert(0, str(ROOT / "notebooks"))
    import isoform_blank_exons_359 as ib

    ib.fig_gate(gate, str(tmp_path / "sfa"))
    ib.fig_landing(pl.read_csv(out / "summary.csv"), str(tmp_path / "sfb"))
    ex = json.loads((out / "examples.json").read_text())
    ib.fig_examples(pl.read_parquet(out / "survivors.parquet"), ex["exon_ids"], str(tmp_path / "sfc"))
    assert all((tmp_path / f"{n}.png").exists() for n in ("sfa", "sfb", "sfc"))
