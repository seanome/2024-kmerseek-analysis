"""Tests for notebooks/invisible_mutations_utils.py on small hand-made cases.

Run: /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 -m pytest -q \
     tests/test_invisible_mutations_utils.py
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "notebooks"))
import invisible_mutations_utils as im  # noqa: E402

REF = "MKLAVD"  # 6 residues; HP (Lehninger): h p h h h p


def scores_table(rows):
    """MaveDB-shaped table: accession, hgvs_pro, score, replicate columns."""
    out = []
    for i, (hgvs, reps) in enumerate(rows):
        reps = list(reps) + [None] * (3 - len(reps))
        vals = [r for r in reps if r is not None]
        out.append(
            {"accession": f"v{i}", "hgvs_pro": hgvs,
             "score": float(np.mean(vals)) if vals else None,
             "score1": reps[0], "score2": reps[1], "score3": reps[2],
             "abundance_class": None}
        )  # fmt: skip
    return pl.DataFrame(
        out, schema_overrides={f"score{j}": pl.Float64 for j in (1, 2, 3)}
    )


def build(rows, ref=REF, assay="A", background="B"):
    return im.build_variants(
        scores_table(rows), ref, assay=assay, condition="c", background=background,
        replicate_columns=["score1", "score2", "score3"],
    )  # fmt: skip


def floor_for(variants, encoding, extra_key=()):
    keys = [
        (
            r["assay"],
            r["condition"],
            r["background"],
            im.encode(r["sequence"], encoding),
        )
        + tuple(r[k] for k in extra_key)
        for r in variants.iter_rows(named=True)
    ]
    g = im.group_ids(keys)
    y = variants["y"].to_numpy()
    v = variants["se"].to_numpy() ** 2
    return im.floor_summary(y, g, v), g


# --- mappings -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,flag", [("hp_lehninger2", "hp"), ("dayhoff6", "dayhoff")]
)
def test_mapping_matches_sourmash_hashes(name, flag):
    """Two residues share a class in our table exactly when sourmash (the encoder kmerseek
    calls for these alphabets) hashes their homopolymer k-mers the same."""
    sourmash = pytest.importorskip("sourmash")
    mh = sourmash.MinHash(n=0, ksize=4, scaled=1, **{flag: True})
    mapping = im.alphabet_mapping(name)
    for a, b in combinations(im.STANDARD_AA, 2):
        ha = mh.seq_to_hashes(a * 4, is_protein=True)
        hb = mh.seq_to_hashes(b * 4, is_protein=True)
        assert (ha == hb) == (mapping[a] == mapping[b]), (a, b)


def test_hp_lehninger_puts_glycine_and_proline_with_hydrophobic():
    m = im.alphabet_mapping("hp_lehninger2")
    assert m["G"] == m["P"] == m["L"] and m["C"] == m["K"]


def test_unknown_residue_is_never_encoded():
    with pytest.raises(ValueError):
        im.encode("MKXA", "hp_lehninger2")


def test_translate():
    assert im.translate("ATGAAATAA") == "MK"


# --- parsing and reconstruction -------------------------------------------------------


def test_parse_kinds():
    assert im.parse_hgvs_pro("p.Lys2Arg", REF).kind == "missense"
    assert im.parse_hgvs_pro("p.Lys2=", REF).kind == "synonymous"
    assert im.parse_hgvs_pro("p.Lys2Ter", REF).kind == "stop"
    assert im.parse_hgvs_pro("p.Ala2Arg", REF).kind == "reference_mismatch"
    assert im.parse_hgvs_pro("p.Lys9Arg", REF).kind == "reference_mismatch"
    assert im.parse_hgvs_pro("p.Lys2Xaa", REF).kind == "nonstandard_residue"
    assert im.parse_hgvs_pro("p.Lys2_Leu3del", REF).kind == "malformed"
    assert im.parse_hgvs_pro("p.[Lys2Arg;Leu3Ile]", REF).kind == "malformed"
    assert im.parse_hgvs_pro(None, REF).kind == "malformed"
    assert im.parse_hgvs_pro("_wt", REF).kind == "wild_type_anchor"


def test_invalid_notation_is_excluded_with_reason():
    variants, excl, syn = build(
        [("p.Lys2Arg", [1.0, 1.2]), ("p.Ala2Arg", [0.5, 0.6]), ("p.Lys2Ter", [0.0, 0.1]),
         ("p.Lys2=", [1.0, 0.9]), ("junk", [1.0, 1.0])]
    )  # fmt: skip
    assert variants.height == 1
    assert variants["sequence"][0] == "MRLAVD"
    assert sorted(excl["reason"]) == ["malformed", "reference_mismatch", "stop"]
    assert syn.height == 1


def test_replicates_of_one_sequence_count_once():
    variants, excl, _ = build([("p.Lys2Arg", [1.0, 2.0]), ("p.Lys2Arg", [3.0])])
    assert variants.height == 1
    assert variants["n_replicates"][0] == 3
    assert variants["y"][0] == pytest.approx(2.0)
    assert excl["reason"][0].startswith("duplicate_sequence")
    # identity control: no two distinct sequences collide, floor is zero
    s, _ = floor_for(variants, "identity")
    assert s["floor_mse"] == 0 and s["n_groups_collided"] == 0


# --- collisions and floors -------------------------------------------------------------


def test_same_reduced_sequence_different_outcomes():
    # K2R and K2H: K, R, H are all polar in HP, so all three sequences encode the same.
    variants, _, _ = build([("p.Lys2Arg", [1.0, 1.0]), ("p.Lys2His", [3.0, 3.0])])
    s, _ = floor_for(variants, "hp_lehninger2")
    assert s["n_groups_collided"] == 1 and s["fraction_collided"] == 1
    assert s["floor_mse"] == pytest.approx(1.0)  # (1-2)^2 + (3-2)^2 over 2
    assert s["total_variance"] == pytest.approx(1.0)
    assert s["floor_fraction"] == pytest.approx(1.0)
    ident, _ = floor_for(variants, "identity")
    assert ident["floor_mse"] == 0


def test_same_reduced_sequence_same_outcome():
    variants, _, _ = build([("p.Lys2Arg", [1.0, 1.0]), ("p.Lys2His", [1.0, 1.0])])
    s, _ = floor_for(variants, "hp_lehninger2")
    assert s["n_groups_collided"] == 1 and s["floor_mse"] == 0


def test_no_collisions():
    # K2L (p -> h) and L3D (h -> p) give different HP strings.
    variants, _, _ = build([("p.Lys2Leu", [1.0, 1.1]), ("p.Leu3Asp", [5.0, 5.1])])
    s, g = floor_for(variants, "hp_lehninger2")
    assert len(set(g)) == 2 and s["n_groups_collided"] == 0 and s["floor_mse"] == 0
    assert np.isnan(s["collided_mse"])


def test_same_string_in_different_assays_or_backgrounds_stays_apart():
    a, _, _ = build([("p.Lys2Arg", [1.0, 1.0])], assay="A1")
    b, _, _ = build([("p.Lys2His", [3.0, 3.0])], assay="A2")
    c, _, _ = build([("p.Lys2His", [3.0, 3.0])], background="B2")
    s, g = floor_for(pl.concat([a, b, c]), "hp_lehninger2")
    assert len(set(g)) == 3 and s["n_groups_collided"] == 0


def test_composition_is_neither_coarser_nor_finer():
    # L2V and L4V change the counts the same way (one L fewer, one V more), so composition
    # groups them although they sit at different positions; L2I does not join them.
    ref = "MLALAD"
    variants, _, _ = build(
        [("p.Leu2Val", [1.0, 1.0]), ("p.Leu4Val", [2.0, 2.0]), ("p.Leu2Ile", [3.0, 3.0])],
        ref=ref,
    )  # fmt: skip
    comp, gc = floor_for(variants, "composition")
    hp, gh = floor_for(variants, "hp_lehninger2")
    # composition: L2V and L4V collide (same counts); L2I alone.
    assert comp["n_groups_collided"] == 1 and comp["n_variants_collided"] == 2
    # HP: all three are h -> h, so all equal the reference's HP string.
    assert hp["n_groups_collided"] == 1 and hp["n_variants_collided"] == 3


def test_same_position_split():
    ref = "MLALAD"
    variants, _, _ = build(
        [("p.Leu2Val", [1.0, 1.0]), ("p.Leu4Val", [2.0, 2.0]), ("p.Leu2Ile", [3.0, 3.0])],
        ref=ref,
    )  # fmt: skip
    s, _ = floor_for(variants, "hp_lehninger2", extra_key=("position",))
    assert s["n_groups_collided"] == 1 and s["n_variants_collided"] == 2
    assert s["floor_mse"] == pytest.approx(((1 - 2) ** 2 + (3 - 2) ** 2) / 3)


# --- noise term and weights -------------------------------------------------------------


def test_expected_noise_term_matches_simulation():
    """If all members share one true value, E[sse] equals the noise term."""
    rng = np.random.default_rng(0)
    v = np.array([0.04, 0.01, 0.09, 0.25])
    groups = np.array([0, 0, 0, 1])
    sims = []
    for _ in range(20000):
        y = rng.normal(0, np.sqrt(v))
        sims.append(im.floor_sums(y, groups, v)["sse"][0])
    expected = im.floor_sums(np.zeros(4), groups, v)["noise"][0]
    assert expected == pytest.approx((1 - 1 / 3) * (0.04 + 0.01 + 0.09))
    assert np.mean(sims) == pytest.approx(expected, rel=0.03)


def test_weights_equal_duplicated_rows():
    y = np.array([1.0, 2.0, 4.0])
    v = np.array([0.1, 0.2, 0.3])
    g = np.array([0, 0, 1])
    w = np.array([2.0, 1.0, 1.0])
    s_w = im.floor_summary(y, g, v, w)
    s_d = im.floor_summary(np.r_[y, 1.0], np.r_[g, 0], np.r_[v, 0.1])
    for k in ("floor_mse", "total_variance", "N"):
        assert s_w[k] == pytest.approx(s_d[k])
    # A duplicated copy shares its error, so the noise term differs from an independent
    # fourth variant: sum w v - sum w^2 v / W = (0.2+0.2) - (0.4+0.2)/3
    assert im.floor_sums(y, g, v, w)["noise"][0] == pytest.approx(0.4 - 0.6 / 3)


def test_zero_weight_member_does_not_make_a_collision():
    y = np.array([1.0, 5.0])
    s = im.floor_summary(
        y, np.array([0, 0]), np.array([0.1, 0.1]), np.array([1.0, 0.0])
    )
    assert s["n_groups_collided"] == 0 and s["floor_mse"] == 0


def test_kmer_sets_identical_for_colliding_sequences():
    a, b = "MRLAVD", "MHLAVD"
    ea, eb = im.encode(a, "hp_lehninger2"), im.encode(b, "hp_lehninger2")
    assert ea == eb and im.kmer_set(ea, 3) == im.kmer_set(eb, 3)
    assert im.kmer_set(a, 3) != im.kmer_set(b, 3)


# --- all kmerseek alphabets and k-mer set keys ---------------------------------------

KMERSEEK_RS = Path(__file__).parents[2] / "kmerseek" / "src" / "rust" / "alphabets.rs"


@pytest.mark.parametrize("name", im.KMERSEEK_ALPHABETS)
def test_every_alphabet_covers_the_20_residues_once(name):
    mapping = im.alphabet_mapping(name)
    assert sorted(mapping) == sorted(im.STANDARD_AA)
    # the size is the number kmerseek's name ends in (20 for protein20)
    import re

    size = int(re.search(r"(\d+)$", name).group(1)) if name != "identity" else 20
    assert im.alphabet_size(name) == size


def test_clusters_match_kmerseek_source():
    """Each cluster table equals the one in kmerseek's alphabets.rs (needs a kmerseek
    checkout next to this repository)."""
    import re

    if not KMERSEEK_RS.exists():
        pytest.skip(f"no kmerseek checkout at {KMERSEEK_RS}")
    src = KMERSEEK_RS.read_text()
    found = {}
    for m in re.finditer(r"const (\w+)_CLUSTERS: &\[&str\] =\s*&\[(.*?)\];", src, re.S):
        found[m.group(1).lower()] = sorted(re.findall(r'"(\w+)"', m.group(2)))
    hp = {
        "THOMAS_DILL_HP": "hp_thomas_dill2", "KYTE_DOOLITTLE_HP": "hp_kyte_doolittle2",
        "THOMAS_DILL_NO_C_HP": "hp_thomas_dill_no_c2",
        "LEHNINGER_C_NONPOLAR_HP": "hp_lehninger_c_nonpolar2",
        "LEHNINGER_HPC": "hp_lehninger_hpc3", "PBOTC_1ST_ED_HP": "hp_pbotc_1st_ed2",
    }  # fmt: skip
    for m in re.finditer(
        r"static (\w+): LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| build_hpc?\((.*?)\)\);",
        src, re.S,
    ):  # fmt: skip
        if m.group(1) in hp:
            found[hp[m.group(1)]] = sorted(re.findall(r'b"(\w+)"', m.group(2)))
    for name, clusters in im.KMERSEEK_CLUSTERS.items():
        assert found[name] == sorted(clusters.values()), name


@pytest.mark.parametrize("name", ["hp_lehninger2", "dayhoff6", "sdm12", "identity"])
@pytest.mark.parametrize("k", [1, 2, 3, 5, 9])
def test_kmer_set_keys_equal_exactly_when_kmer_sets_equal(name, k):
    """Against brute force: every single substitution of a short repetitive sequence."""
    ref = "MKLAVDLLAKMAVDKLLA"
    pos, alts = [], []
    for i, r in enumerate(ref, 1):
        for a in im.STANDARD_AA:
            if a != r:
                pos.append(i)
                alts.append(a)
    keys = im.kmer_set_keys(ref, np.array(pos), alts, name, k)
    sets = [
        frozenset(im.kmer_set(im.encode(im.apply_substitution(ref, p, a), name), k))
        for p, a in zip(pos, alts)
    ]
    g_keys, g_sets = im.group_ids(keys), im.group_ids([(s,) for s in sets])
    for i, j in combinations(range(0, len(pos), 7), 2):
        assert (g_keys[i] == g_keys[j]) == (g_sets[i] == g_sets[j]), (
            pos[i],
            alts[i],
            pos[j],
            alts[j],
        )


def test_kmer_set_groups_are_never_finer_than_sequence_groups():
    """Same encoded sequence means same k-mer set, so a k-mer set group is a union of
    sequence groups."""
    ref = "MKLAVDLLAKMAVDKLLA"
    pos = [2, 2, 5, 9, 9, 12]
    alts = ["R", "H", "I", "L", "V", "D"]
    keys = im.kmer_set_keys(ref, np.array(pos), alts, "hp_lehninger2", 4)
    seqs = [
        im.encode(im.apply_substitution(ref, p, a), "hp_lehninger2")
        for p, a in zip(pos, alts)
    ]
    for i, j in combinations(range(len(pos)), 2):
        if seqs[i] == seqs[j]:
            assert keys[i] == keys[j]
