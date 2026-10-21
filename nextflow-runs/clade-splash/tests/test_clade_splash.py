"""Tests of bin/clade_splash.py against brute force, scipy and a planted signal.

Run with the project env: make test. The OASIS.py these tests import is fetched from
refresh-bio/SPLASH at the pinned commit (make oasis), as the pipeline does.
"""

from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bin"))
import clade_splash as cs  # noqa: E402
import clade_splash_pfam as csp  # noqa: E402
import clade_splash_collect as csc  # noqa: E402
import clade_splash_test as cst  # noqa: E402

OASIS = HERE.parent / "assets" / "OASIS.py"
needs_oasis = pytest.mark.skipif(not OASIS.exists(), reason="run `make oasis` first")

BCL2_HUMAN = ("MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAA"
              "AGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVN"
              "REMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK")


def brute_anchors(seq: str, alphabet: str, k: int) -> list[tuple[int, str]]:
    """Usable anchors of one protein by direct string work: (start, reduced k-mer)."""
    groups = cs.ALPHABETS[alphabet]
    to_red = {aa: letter for letter, aas in groups.items() for aa in aas}
    out = []
    for i in range(len(seq) - k + 1):
        w = seq[i:i + k]
        if any(c not in cs.AMINO_ACIDS for c in w):
            continue
        if len(set(w)) == 1:
            continue
        red = "".join(to_red[c] for c in w)
        if alphabet == "hp" and len(set(red)) == 1:
            continue
        out.append((i, red))
    return out


@pytest.mark.parametrize("alphabet,k", [("dayhoff", 14), ("dayhoff", 5), ("hp", 36), ("hp", 6)])
def test_anchor_codes_match_brute_force(alphabet, k):
    rng = np.random.default_rng(0)
    seqs = [BCL2_HUMAN, "A" * 40 + "MKV" + "L" * 20, "ACDXEFGHIKLMNPQRSTVWYBZ" * 3,
            "".join(rng.choice(list(cs.AMINO_ACIDS), 300)), "MK", "AILMFVWYGP" * 6]
    aa, starts = cs.concat_proteome(seqs)
    codes, usable = cs.anchor_codes(aa, alphabet, k)
    got = []
    for gi in np.nonzero(usable)[0]:
        p = np.searchsorted(starts, gi, side="right") - 1
        got.append((int(p), int(gi - starts[p]), cs.decode_anchor(int(codes[gi]), alphabet, k)))
    want = [(p, i, red) for p, s in enumerate(seqs) for i, red in brute_anchors(s, alphabet, k)]
    assert got == want


def test_hp_mask_drops_encoded_runs_but_dayhoff_does_not():
    # 'AILMFVWY...' is all h in HP and all e/b/f in Dayhoff: HP drops it, Dayhoff keeps it.
    seq = "AILMVAILMVAILMVAILMV"  # Dayhoff: b e e e e ...; HP: all h
    aa, _ = cs.concat_proteome([seq])
    _, us_hp = cs.anchor_codes(aa, "hp", 10)
    _, us_d = cs.anchor_codes(aa, "dayhoff", 10)
    assert not us_hp.any()
    assert us_d[: len(seq) - 10 + 1].all()  # the last window runs into the separator


def test_target_codes_round_trip():
    aa, _ = cs.concat_proteome([BCL2_HUMAN])
    codes, valid = cs.target_codes(aa)
    for i in (0, 17, len(BCL2_HUMAN) - 10):
        assert valid[i]
        assert cs.decode_target(int(codes[i])) == BCL2_HUMAN[i:i + 10]
    assert not valid[len(BCL2_HUMAN) - 9]  # would run into the separator


def test_chi2_sf_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    for df in (1, 2, 5, 19, 57, 190, 1000):
        for x in (0.01, 0.5, df * 0.5, df, df + 3 * np.sqrt(2 * df), df * 4 + 50):
            assert cs.chi2_sf(x, df) == pytest.approx(scipy_stats.chi2.sf(x, df), rel=1e-9, abs=1e-300)


def test_pearson_chi2_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    t = np.array([[10, 0, 3], [2, 8, 4], [0, 0, 0], [5, 5, 5]])
    x2, df, p = cs.pearson_chi2(t)
    ref = scipy_stats.chi2_contingency(t[[0, 1, 3]], correction=False)
    assert x2 == pytest.approx(ref[0]) and df == ref[2] and p == pytest.approx(ref[1])


def test_bh_matches_statsmodels_formula():
    p = np.array([0.01, 0.04, 0.03, 0.2, 0.001])
    q = cs.bh(p)
    # Hand computation: sorted p 0.001,0.01,0.03,0.04,0.2 -> *5/rank = .005,.025,.05,.05,.2
    assert np.allclose(q, [0.025, 0.05, 0.05, 0.2, 0.005])


def test_composition_table_counts_every_residue():
    t1 = cs.target_codes(cs.concat_proteome(["AAAAACCCCC"])[0])[0][0]
    t2 = cs.target_codes(cs.concat_proteome(["WWWWWWWWWY"])[0])[0][0]
    counts = np.array([[2, 0], [1, 3]])
    comp = cs.composition_table(np.array([t1, t2]), counts)
    assert comp.sum() == 10 * counts.sum()
    assert comp[cs.AMINO_ACIDS.index("A"), 0] == 10 and comp[cs.AMINO_ACIDS.index("W"), 1] == 27


def species_table(n_classes=4, per_class=6, phyla=("P1", "P1", "P2", "P2")) -> pl.DataFrame:
    rows = []
    for c in range(n_classes):
        for s in range(per_class):
            rows.append({"species_idx": c * per_class + s, "class": f"C{c}", "phylum": phyla[c],
                         "kingdom": None, "domain": "D"})
    return pl.DataFrame(rows)


def test_labellings_keep_column_sizes_and_tree_stays_in_phylum():
    sp = species_table()
    labs = cs.labellings(sp, seed=357)
    real = labs["real"]
    for name in ("shuffle_random", "shuffle_tree"):
        assert sorted(labs[name]) == sorted(real)
    phylum_of_class = {"C0": "P1", "C1": "P1", "C2": "P2", "C3": "P2"}
    for i in range(len(real)):
        assert phylum_of_class[labs["shuffle_tree"][i]] == phylum_of_class[real[i]]
    assert (labs["shuffle_random"] != real).sum() > 0 and (labs["shuffle_tree"] != real).sum() > 0


@needs_oasis
def test_oasis_without_scipy_matches_with_scipy():
    """The finite-sample p-value is the same whether or not scipy is importable."""
    pytest.importorskip("scipy")
    x = np.array([[8, 1, 0, 2], [0, 7, 1, 1], [1, 0, 6, 5]])
    code = ("import sys, numpy as np; sys.modules['scipy'] = None; sys.path.insert(0, %r); "
            "import clade_splash as cs; sys.modules.pop('scipy'); "
            "import builtins; real_import = builtins.__import__\n"
            "def no_scipy(name, *a, **k):\n"
            "    if name.split('.')[0] == 'scipy' and name not in sys.modules: raise ImportError(name)\n"
            "    return real_import(name, *a, **k)\n"
            "builtins.__import__ = no_scipy\n"
            "m = cs.load_oasis(%r); print(float(m.OASIS_pvalue(np.array(%r))))") % (
        str(HERE.parent / "bin"), str(OASIS), x.tolist())
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    with_scipy = cs.load_oasis(str(OASIS)).OASIS_pvalue(x)
    assert float(out) == pytest.approx(with_scipy)


def write_fasta(path: Path, seqs: list[str]) -> None:
    with gzip.open(path, "wt") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">tr|P{i:05d}|X_{i}\n{s}\n")


@needs_oasis
def test_end_to_end_planted_clade_target(tmp_path):
    """Every species has the same Dayhoff anchor; class C0 follows it with one target and the
    other classes with another. That anchor must be significant on the real labels; a second
    anchor followed by random targets in every species must not be."""
    rng = np.random.default_rng(1)
    # 15 species per class: OASIS's finite-sample bound cannot reach q < 0.05 on very small
    # tables (a perfect 4 x 6 table gives p = 0.0055 at best).
    sp = species_table(n_classes=4, per_class=15)
    sp.write_csv(tmp_path / "species.tsv", separator="\t")
    anchor_planted = "MKVLAAGHTRDEWY"   # 14 residues
    anchor_null = "PPQRSNLIVKDEFG"
    targ_a, targ_b = "CCWWHHMMYY", "DDEEKKRRNN"
    fdir = tmp_path / "fasta"
    fdir.mkdir()
    for r in sp.iter_rows(named=True):
        filler = lambda n: "".join(rng.choice(list(cs.AMINO_ACIDS), n))
        t = targ_a if r["class"] == "C0" else targ_b
        prot1 = filler(30) + anchor_planted + t + filler(30)
        prot2 = filler(20) + anchor_null + filler(10) + filler(20)
        write_fasta(fdir / f"{r['species_idx']}.fasta.gz", [prot1, prot2, filler(200)])
    fastas = sorted(str(p) for p in fdir.glob("*.fasta.gz"))
    cs.main(["anchors", "--setting", "dayhoff:14", "--fastas", *fastas,
             "--outdir", str(tmp_path / "a"), "--stats", str(tmp_path / "a.json")])
    cands = []
    for b in range(cs.N_ANCHOR_BUCKETS):
        out = tmp_path / f"c{b}.npy"
        cs.main(["candidates", "--bucket", str(b), "--dirs", str(tmp_path / "a"),
                 "--out", str(out), "--hist", str(tmp_path / f"h{b}.parquet")])
        cands.append(str(out))
    total = sum(len(np.load(c)) for c in cands)
    assert total >= 2  # both anchors (60 species >= 15)
    cs.main(["targets", "--setting", "dayhoff:14", "--gaps", "0", "--candidates", *cands, "--fastas", *fastas,
             "--outdir", str(tmp_path / "t"), "--accessions", str(tmp_path / "acc.parquet")])
    outs = []
    for b in range(cs.N_TARGET_BUCKETS):
        o = tmp_path / f"test{b}.parquet"
        cst.main(["test", "--setting", "dayhoff:14", "--bucket", str(b), "--dirs", str(tmp_path / "t"),
                 "--species", str(tmp_path / "species.tsv"), "--oasis", str(OASIS), "--seed", "357", "--out", str(o)])
        outs.append(str(o))
    csc.main(["--inputs", *outs, "--all-out", str(tmp_path / "all"),
              "--sig-out", str(tmp_path / "sig.parquet"), "--counts-out", str(tmp_path / "counts.tsv")])
    res = pl.read_parquet(tmp_path / "all" / "dayhoff_14_gap0_real.parquet")
    aa_planted, _ = cs.concat_proteome([anchor_planted])
    code_planted = int(cs.anchor_codes(aa_planted, "dayhoff", 14)[0][0])
    aa_null, _ = cs.concat_proteome([anchor_null])
    code_null = int(cs.anchor_codes(aa_null, "dayhoff", 14)[0][0])
    planted = res.filter(pl.col("anchor") == code_planted)
    null = res.filter(pl.col("anchor") == code_null)
    assert planted.height == 1 and null.height == 1
    assert planted["n_clades"][0] == 4 and planted["n_targets"][0] == 2
    assert planted["n_species"][0] == 60 and planted["table_total"][0] == 60
    assert planted["q_oasis"][0] < 0.05
    assert null["q_oasis"][0] > 0.05
    # The composition scan sees it too (C0 is all C/W/H/M/Y): it is "explained by composition".
    assert planted["p_comp"][0] < 1e-6
    sig = pl.read_parquet(tmp_path / "sig.parquet").filter(pl.col("labelling") == "real")
    assert code_planted in sig["anchor"].to_list()
    assert sig.filter(pl.col("anchor") == code_planted)["anchor_letters"][0] == "edeebbbdbdccff"  # M K V L A A G H T R D E W Y


def test_stockholm_match_columns(tmp_path):
    sto = tmp_path / "x.sto"
    sto.write_text("# STOCKHOLM 1.0\n\nseq1  AC-Dk..E\nseq2  ACGDk..E\n#=GC RF xxxx...x\n//\n")
    cols = csp.read_stockholm_match_columns(str(sto))
    assert cols["seq1"] == [(0, "A"), (1, "C"), (-1, "-"), (2, "D"), (4, "E")]
    assert cols["seq2"] == [(0, "A"), (1, "C"), (2, "G"), (3, "D"), (5, "E")]


@needs_oasis
def test_occurrences_and_pfam_overlap(tmp_path):
    """Rerun the planted case, then put a fake Pfam domain over the planted window in the
    class C0 proteins only: the planted anchor is then inside Pfam (not outside), and its
    residues come back exactly."""
    test_end_to_end_planted_clade_target(tmp_path)
    occ_out, fa_out = [], []
    for b in range(cs.N_TARGET_BUCKETS):
        o, f = tmp_path / f"occ{b}.parquet", tmp_path / f"prot{b}.fa"
        cst.main(["occurrences", "--setting", "dayhoff:14", "--bucket", str(b), "--dirs", str(tmp_path / "t"),
                 "--accessions", str(tmp_path / "acc.parquet"), "--significant", str(tmp_path / "sig.parquet"),
                 "--fasta-dirs", str(tmp_path / "fasta"), "--out", str(o), "--fasta-out", str(f)])
        occ_out.append(str(o))
        fa_out.append(str(f))
    occ = pl.concat([pl.read_parquet(f) for f in occ_out])
    assert occ.height == 60  # one occurrence in each of 60 species
    assert set(occ["anchor_residues"]) == {"MKVLAAGHTRDEWY"}
    assert set(occ["target_residues"]) == {"CCWWHHMMYY", "DDEEKKRRNN"}
    assert set(occ["pos0"]) == {30}
    # Fake hmmscan rows: a domain over residues 31-54 (1-based) in species 0-14 (class C0).
    dom = tmp_path / "fake.domtblout"
    with open(dom, "w") as fh:
        fh.write("# fake\n")
        for s in range(15):
            # hmmsearch layout: target (protein), accession, length, query (family), accession, ...
            f = [f"s{s}_p0", "-", "84", "PF99999_x", "PF99999.1", "100"] + ["0"] * 6 + ["1e-10"] + ["0"] * 6 + ["31", "54", "0.9", "-"]
            fh.write(" ".join(f) + "\n")
    csp.main(["pfam-overlap", "--occurrences", *occ_out, "--domtblout", str(dom),
             "--occ-out", str(tmp_path / "po"), "--out", str(tmp_path / "pa.parquet"),
             "--significant", str(tmp_path / "sig.parquet"), "--top-out", str(tmp_path / "top.parquet")])
    pa = pl.read_parquet(tmp_path / "pa.parquet")
    assert pa.height == 1
    row = pa.row(0, named=True)
    assert row["n_occurrences"] == 60 and row["n_occ_in_pfam"] == 15 and not row["outside_pfam"]
    assert row["main_pfam"] == "PF99999"
    top = pl.read_parquet(tmp_path / "top.parquet")
    assert top.height == 60 and top["in_pfam"].sum() == 15


HMMER_BIN = Path("/Users/olga/anaconda3/envs/hmmer/bin")
needs_hmmer = pytest.mark.skipif(not (HMMER_BIN / "hmmalign").exists(), reason="HMMER not installed")


@needs_oasis
@needs_hmmer
def test_alignment_control_finds_the_planted_columns(tmp_path):
    """Make a fake Pfam family from the planted region (residues 21-64 of every species'
    first protein), align the envelopes to it with hmmalign as the pipeline does, and run the
    per-column test: the 10 target columns are found, and the class split is significant."""
    test_end_to_end_planted_clade_target(tmp_path)
    occ_out, fa_out = [], []
    for b in range(cs.N_TARGET_BUCKETS):
        o, f = tmp_path / f"occ{b}.parquet", tmp_path / f"prot{b}.fa"
        cst.main(["occurrences", "--setting", "dayhoff:14", "--bucket", str(b), "--dirs", str(tmp_path / "t"),
                 "--accessions", str(tmp_path / "acc.parquet"), "--significant", str(tmp_path / "sig.parquet"),
                 "--fasta-dirs", str(tmp_path / "fasta"), "--out", str(o), "--fasta-out", str(f)])
        occ_out.append(str(o))
        fa_out.append(str(f))
    seqs = {}
    for f in fa_out:
        name = None
        for line in open(f):
            if line.startswith(">"):
                name = line[1:].split()[0]
            elif name:
                seqs[name] = line.strip()
    # Family alignment: residues 21-64 of every protein (ungapped, so already aligned).
    with open(tmp_path / "fam.sto", "w") as fh:
        fh.write("# STOCKHOLM 1.0\n")
        for n, sq in seqs.items():
            fh.write(f"{n} {sq[20:64]}\n")
        fh.write("//\n")
    subprocess.run([str(HMMER_BIN / "hmmbuild"), "-n", "FAKE", str(tmp_path / "fam.hmm"), str(tmp_path / "fam.sto")],
                   check=True, capture_output=True)
    dom = tmp_path / "fake.domtblout"
    with open(dom, "w") as fh:
        for n in seqs:
            f = [n, "-", "84", "FAKE", "PF99999.1", "44"] + ["0"] * 6 + ["1e-10"] + ["0"] * 6 + ["21", "64", "0.9", "-"]
            fh.write(" ".join(f) + "\n")
    csp.main(["pfam-overlap", "--occurrences", *occ_out, "--domtblout", str(dom),
             "--occ-out", str(tmp_path / "po"), "--out", str(tmp_path / "pa.parquet"),
             "--significant", str(tmp_path / "sig.parquet"), "--top-out", str(tmp_path / "top.parquet")])
    csp.main(["align-input", "--pfam", str(tmp_path / "pa.parquet"), "--occurrences", *occ_out,
             "--domtblout", str(dom), "--fastas", *fa_out, "--outdir", str(tmp_path / "ain"),
             "--manifest", str(tmp_path / "man.tsv"), "--parts-dir", str(tmp_path / "parts")])
    man = pl.read_csv(tmp_path / "man.tsv", separator="\t")
    assert man.height == 1
    assert pl.read_csv(tmp_path / "parts" / "manifest_0000.tsv", separator="\t").equals(man)
    tag = man["tag"][0]
    (tmp_path / "sto").mkdir()
    with open(tmp_path / "sto" / f"{tag}.sto", "w") as fh:
        subprocess.run([str(HMMER_BIN / "hmmalign"), "--outformat", "Stockholm", str(tmp_path / "fam.hmm"),
                        str(tmp_path / "ain" / f"{tag}.fa")], check=True, stdout=fh)
    csp.main(["align-control", "--manifest", str(tmp_path / "man.tsv"), "--sto-dir", str(tmp_path / "sto"),
             "--significant", str(tmp_path / "sig.parquet"), "--species", str(tmp_path / "species.tsv"),
             "--oasis", str(OASIS), "--seed", "357", "--out", str(tmp_path / "ac.parquet")])
    ac = pl.read_parquet(tmp_path / "ac.parquet")
    assert ac.height == 10  # the 10 target residues map to 10 match columns
    assert ac["n_species"].min() == 60
    assert (ac["q_column"] < 0.05).sum() >= 8  # C0 differs from the rest at every target column


@needs_oasis
def test_streaming_test_matches_the_original(tmp_path):
    """clade_splash_test.py's test gives the same rows as the original cmd_test in
    clade_splash.py on the planted data, for every labelling."""
    test_end_to_end_planted_clade_target(tmp_path)
    old, new = [], []
    for b in range(cs.N_TARGET_BUCKETS):
        args = ["test", "--setting", "dayhoff:14", "--bucket", str(b), "--dirs", str(tmp_path / "t"),
                "--species", str(tmp_path / "species.tsv"), "--oasis", str(OASIS), "--seed", "357"]
        cs.main(args + ["--out", str(tmp_path / f"old{b}.parquet")])
        old.append(pl.read_parquet(tmp_path / f"old{b}.parquet"))
        new.append(pl.read_parquet(tmp_path / f"test{b}.parquet"))
    key = ["labelling", "gap", "anchor"]
    o, n = pl.concat(old).sort(key), pl.concat(new).sort(key)
    assert o.height == n.height > 0
    assert o.columns == n.columns
    assert o.equals(n)


@needs_oasis
def test_collect_matches_the_original(tmp_path):
    """clade_splash_collect.py gives the same significant rows and counts as the original
    collect in clade_splash.py."""
    test_end_to_end_planted_clade_target(tmp_path)
    outs = [str(tmp_path / f"test{b}.parquet") for b in range(cs.N_TARGET_BUCKETS)]
    cs.main(["collect", "--inputs", *outs, "--all-out", str(tmp_path / "old_all.parquet"),
             "--sig-out", str(tmp_path / "old_sig.parquet"), "--counts-out", str(tmp_path / "old_counts.tsv")])
    key = ["setting", "gap", "labelling", "anchor"]
    old = pl.read_parquet(tmp_path / "old_sig.parquet").sort(key)
    new = pl.read_parquet(tmp_path / "sig.parquet").sort(key)
    assert old.height == new.height > 0
    assert new.select(old.columns).equals(old)
    oc = pl.read_csv(tmp_path / "old_counts.tsv", separator="\t").sort("setting", "gap", "labelling")
    nc = pl.read_csv(tmp_path / "counts.tsv", separator="\t").sort("setting", "gap", "labelling")
    assert oc.equals(nc)
