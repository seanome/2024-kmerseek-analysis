"""Notebook 284, part 4: the spot tables the interactive explainer page filters and sums.

The page lets the reader pick an E-value cut, a counting rule and which alphabets count, and
redraws where BCL2 (query Ced9) and CD47 (query P66) land among the human proteins. Every
count it shows is a sum over the tables written here. The spots come from
alphabet_agreement_utils.assign_spots, the same code behind every other 284 number, run once
per E-value cut in alphabet_agreement_utils.E_CUTS. So the page never joins regions itself.

For each query and cut, one binary file holds:
  - every spot that two or more alphabets back: target protein, query start and end,
    diagonal, and the set of alphabets as a 19-bit mask (bit order = ALPHABETS below);
  - for each alphabet in such a spot: its query start and end in the spot, its lowest
    E-value, and the k-sizes that found it (bit k set);
  - for every target protein, the mask of alphabets with any region on it at this cut, so a
    protein whose spots each have one alphabet is still counted.
Spots one alphabet backs are not stored one by one; the masks above stand in for them.

Removing an alphabet on the page clears its bit. A spot that only that alphabet's region
joined stays one spot there, whereas a fresh run without the alphabet would split it. The
script measures how often that changes a number (see CHECKS in the manifest).

Also written: the human residues (for the aligned-residue view), how many human proteins each
alphabet finds at each query position (for the window view), and the residue classes of the
19 alphabets, read from seanome/kmerseek src/rust/alphabets.rs at 00395d7 and, for dayhoff6,
from sourmash 0.20.0's encodings.rs, which kmerseek uses for it.

Inputs: notebook 241's regions.parquet, queries.fa, GENCODE v49 canonical translations, a
kmerseek checkout (KMERSEEK_REPO, default ~/code/kmerseek) and the sourmash 0.20.0 crate source.
Output: notebooks/284_explainer_files/ (binary tables and their base64 parts for the page, not committed) and
notebooks/284_explainer_manifest.json (committed: sizes, sha256 of every file, checks).
"""
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
import alphabet_agreement_utils as au  # noqa: E402

DATA = Path.home() / "data/botryllus/alphabet-ranking-three-cases"
GENCODE = Path.home() / "data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
KMERSEEK = Path(os.environ.get("KMERSEEK_REPO", Path.home() / "code/kmerseek"))
KMERSEEK_SHA = "00395d7"
SOURMASH = next(Path.home().glob(".cargo/registry/src/*/sourmash-0.20.0/src/encodings.rs"))
OUT = ROOT / "notebooks/284_explainer_files"
MANIFEST = ROOT / "notebooks/284_explainer_manifest.json"
# Bit order of the alphabet mask; also the row order of every figure (grouped by size).
ALPHABETS = au.HP + ["gbmr4", "polarity4", "wwmj5", "dayhoff6", "gbmr7", "funcgroups8",
                     "sdm12", "mmseqs12", "wass14", "hsdm17", "uniprot18", "protein20"]
CASES = {"Ced9": "BCL2", "P66": "CD47"}
RES_CHUNK = 1000
B64_PART = 8_000_000  # characters per base64 part; the artifact limit is 16 MB a file
# UniProt BH motifs of Ced9 and five human BCL-2 family members, drawn as boxes on the page.
MOTIFS = ROOT / "notebooks/284_uniprot_bh_motifs.json"


def uniprot_motifs():
    """Fetched once and committed, so the page does not depend on UniProt being up."""
    if not MOTIFS.exists():
        import datetime
        import urllib.request
        out = {"source": "https://rest.uniprot.org/uniprotkb/<acc>.json?fields=ft_motif",
               "fetched": datetime.date.today().isoformat(), "proteins": {}}
        for acc, name in (("P41958", "Ced9"), ("P10415", "BCL2"), ("Q07817", "BCL2L1"), ("Q07820", "MCL1"),
                          ("Q16611", "BAK1"), ("Q07812", "BAX")):
            d = json.load(urllib.request.urlopen(f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields=ft_motif"))
            out["proteins"][name] = {"accession": acc, "motifs": [
                {"name": f["description"], "start": f["location"]["start"]["value"], "end": f["location"]["end"]["value"]}
                for f in d["features"] if f["type"] == "Motif"]}
        assert out["proteins"]["Ced9"]["motifs"], "UniProt returned no motifs"
        MOTIFS.write_text(json.dumps(out, indent=1) + "\n")
    return json.loads(MOTIFS.read_text())


def read_fasta(path, key=lambda h: h):
    seqs, name = {}, None
    for line in open(path):
        if line.startswith(">"):
            name = key(line[1:].strip())
            seqs[name] = []
        else:
            seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}


def residue_classes():
    """Classes of each alphabet, parsed from the source, not typed in by hand."""
    src = subprocess.run(["git", "-C", str(KMERSEEK), "show", f"{KMERSEEK_SHA}:src/rust/alphabets.rs"],
                         check=True, capture_output=True, text=True).stdout
    out = {}
    static_name = {}  # e.g. HpLehninger2 => &LEHNINGER_HP
    for variant, static in re.findall(r"Self::(\w+) => &(\w+),", src):
        static_name[static] = variant
    moltype = dict(re.findall(r"Self::(\w+) => \"(\w+)\",", src))
    for static, body in re.findall(r"static (\w+): LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| (build_\w+\([^)]*\))\);", src, re.S):
        a = moltype[static_name[static]]
        if body.startswith("build_reduced"):
            const = re.search(r"build_reduced\((\w+)\)", body).group(1)
            lst = re.search(const + r": &\[&str\] =\s*&\[([^\]]*)\]", src, re.S).group(1)
            out[a] = re.findall(r'"([A-Z]+)"', lst)
        else:
            out[a] = re.findall(r'b"([A-Z]+)"', body)
    sm = SOURMASH.read_text()
    block = re.search(r"static DAYHOFFTABLE.*?\]", sm, re.S).group(0)
    groups = {}
    for aa, letter in re.findall(r"\(b'([A-Z])', b'([a-z])'\)", block):
        groups.setdefault(letter, []).append(aa)
    out["dayhoff6"] = ["".join(v) for _, v in sorted(groups.items())]
    out["protein20"] = list("ACDEFGHIKLMNPQRSTVWY")
    for a in ALPHABETS:  # each alphabet's classes cover the 20 residues once
        assert sorted("".join(out[a])) == sorted("ACDEFGHIKLMNPQRSTVWY"), a
    return out


def popcount(x):
    return np.unpackbits(x.astype("<u4").view(np.uint8).reshape(-1, 4), axis=1).sum(axis=1)


def counts_from_mask(mask, rule):
    """The three counts of alphabet_agreement_utils, from a mask. The page uses the same rule."""
    mask = mask.astype(np.uint32)
    hp = (1 << len(au.HP)) - 1
    if rule == "n_alphabets":
        return popcount(mask)
    fam = popcount(mask & ~np.uint32(hp)) + ((mask & hp) != 0)
    if rule == "n_families":
        return fam
    return np.where(mask & (1 << ALPHABETS.index("protein20")), 1, fam)


def build(regions, q, cut, tindex, kbits):
    d = au.prepare(regions.filter(pl.col("query_name") == q))
    r = au.assign_spots(d, cut).with_columns(
        bit=pl.col("alphabet").replace_strict(ALPHABETS, list(range(len(ALPHABETS))), return_dtype=pl.Int64))
    r = r.join(kbits.lazy(), on=["alphabet", "ksize_arm"])
    pa = (r.group_by(*au.SPOT_KEYS, "bit")
          .agg(a_qs=pl.col("qs").min(), a_qe=pl.col("qe").max(), a_e=pl.col("region_evalue").min(),
               a_k=pl.lit(2, pl.Int64).pow(pl.col("kbit")).unique().sum())
          .collect())
    sp = (pa.group_by(au.SPOT_KEYS)
          .agg(mask=pl.lit(2, pl.Int64).pow(pl.col("bit")).sum(), n=pl.len(),
               qs=pl.col("a_qs").min(), qe=pl.col("a_qe").max()))
    # the same spots and counts as au.spots, which made the CSVs
    ref = au.spots(d, cut).select(*au.SPOT_KEYS, "n_alphabets", "n_families", "n_families_identity_as_one", "qs", "qe")
    chk = sp.join(ref, on=au.SPOT_KEYS, how="full", coalesce=True)
    assert chk.height == ref.height == sp.height, (q, cut)
    for c in au.COUNTS:
        assert (counts_from_mask(chk["mask"].to_numpy(), c) == chk[c].to_numpy()).all(), (q, cut, c)
    assert (chk["qs"] == chk["qs_right"]).all() and (chk["qe"] == chk["qe_right"]).all()

    sp = sp.with_columns(t=pl.col("target_name").replace_strict(tindex, return_dtype=pl.Int64))
    union = (pa.with_columns(t=pl.col("target_name").replace_strict(tindex, return_dtype=pl.Int64))
             .group_by("t").agg(m=pl.lit(2, pl.Int64).pow(pl.col("bit")).unique().sum()))
    umask = np.zeros(len(tindex), np.uint32)
    umask[union["t"].to_numpy()] = union["m"].to_numpy()
    multi = sp.filter(pl.col("n") >= 2).sort("t", "qs", "diag", "spot").with_row_index("order")
    # one entry per alphabet in each stored spot, in the spots' order, then by bit
    ent = pa.join(multi.select(*au.SPOT_KEYS, "order"), on=au.SPOT_KEYS).sort("order", "bit")
    assert ent.height == int(multi["n"].sum())
    assert multi.height == 0 or (multi["qe"].max() < 65536 and (multi["qe"] - multi["qs"]).min() > 0)
    arrays = [
        ("t", multi["t"].to_numpy().astype("<u2")),
        ("qs", multi["qs"].to_numpy().astype("<u2")),
        ("qe", multi["qe"].to_numpy().astype("<u2")),
        ("diag", multi["diag"].to_numpy().astype("<i4")),
        ("mask", multi["mask"].to_numpy().astype("<u4")),
        ("a_qs", ent["a_qs"].to_numpy().astype("<u2")),
        ("a_qe", ent["a_qe"].to_numpy().astype("<u2")),
        ("a_e", ent["a_e"].to_numpy().astype("<f4")),
        ("a_k", ent["a_k"].to_numpy().astype("u1")),  # bit i set: the alphabet's i-th k-size found it
        ("union", umask.astype("<u4")),
    ]
    # how many human proteins each alphabet finds at each query position (window view)
    L = regions.filter(pl.col("query_name") == q).select(pl.col("region_end").max()).collect().item()
    cov = (r.select("bit", "target_name", pos=pl.int_ranges("qs", "qe")).explode("pos")
           .unique().group_by("bit", "pos").len().collect())
    return arrays, cov, int(multi.height), int(ent.height), int(L)


def main():
    OUT.mkdir(exist_ok=True)
    regions = pl.scan_parquet(DATA / "regions.parquet").filter(pl.col("query_name").is_in(list(CASES)))
    targets = (regions.select("target_name", "gene").unique().collect().sort("gene", "target_name"))
    assert targets["target_name"].n_unique() == targets.height
    tindex = {t: i for i, t in enumerate(targets["target_name"])}
    ks = regions.select("alphabet", "ksize_arm").unique().collect().sort("alphabet", "ksize_arm")
    kbits = ks.with_columns(kbit=pl.col("ksize_arm").rank("ordinal").over("alphabet").cast(pl.Int64) - 1)
    assert kbits["kbit"].max() < 8
    assert len(tindex) < 65536
    queries = read_fasta(DATA / "queries.fa")
    human = read_fasta(GENCODE, key=lambda h: h.split("|")[0])
    seqs = [human[t.split("|")[0]] for t in targets["target_name"]]
    for f in OUT.glob("human_residues*.txt"):
        f.unlink()
    for i in range(0, len(seqs), RES_CHUNK):  # loaded one chunk at a time by the page
        (OUT / f"human_residues.{i // RES_CHUNK}.txt").write_text("\n".join(seqs[i:i + RES_CHUNK]) + "\n")
    for q, p in CASES.items():
        assert targets.filter(pl.col("gene") == p).height == 1, p

    man = {"alphabets": ALPHABETS, "hp": au.HP, "counts": au.COUNTS,
           "cuts": [None if c is None else c for c in au.E_CUTS],
           "genes": targets["gene"].to_list(), "partners": CASES, "n_proteome": len(human),
           "n_regions": {q: regions.filter(pl.col("query_name") == q).select(pl.len()).collect().item() for q in CASES},
           "queries": {q: queries[q] for q in CASES}, "classes": residue_classes(),
           "kmerseek_sha": KMERSEEK_SHA, "motifs": uniprot_motifs(), "res_chunk": RES_CHUNK,
           "ksizes": {a: ks.filter(pl.col("alphabet") == a)["ksize_arm"].to_list() for a in ALPHABETS}, "files": {}, "coverage": {}}
    for q in CASES:
        man["coverage"][q] = []
        for ci, cut in enumerate(au.E_CUTS):
            arrays, cov, n_spots, n_ent, L = build(regions, q, cut, tindex, kbits)
            name = f"spots.{q}.{ci}.bin"
            buf, layout = bytearray(), {}
            for key, a in arrays:
                layout[key] = [len(buf), int(a.size)]
                buf += a.tobytes()
                buf += b"\0" * (-len(buf) % 8)
            (OUT / name).write_bytes(bytes(buf))
            # artifacts serve no binary files, so the page gets the same bytes as base64 text parts
            b64 = base64.b64encode(bytes(buf)).decode()
            for f in OUT.glob(f"{name}.b64.*.txt"):
                f.unlink()
            parts = [b64[i:i + B64_PART] for i in range(0, len(b64), B64_PART)]
            for i, part in enumerate(parts):
                (OUT / f"{name}.b64.{i}.txt").write_text(part)
            man["files"][name] = {"query": q, "cut": ci, "n_spots": n_spots, "n_entries": n_ent, "layout": layout,
                                  "b64_parts": len(parts)}
            grid = np.zeros((len(ALPHABETS), len(queries[q])), np.int32)
            grid[cov["bit"].to_numpy(), cov["pos"].to_numpy()] = cov["len"].to_numpy()
            man["coverage"][q].append(grid.tolist())
            print(q, cut, n_spots, "spots with 2+ alphabets", n_ent, "entries", len(buf), "bytes", flush=True)
    for f in sorted(OUT.iterdir()):
        man["files"].setdefault(f.name, {})
        man["files"][f.name].update(bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
    json.dump(man, open(MANIFEST, "w"), separators=(",", ":"))
    print("manifest", MANIFEST.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
