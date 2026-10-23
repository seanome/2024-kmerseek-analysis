"""Freeze the query and reference panels, labels and splits of the three datasets.

Run once (`stage_diagnostic.py freeze`). Everything later reads only what this writes to
data/350_stage_diagnostic/<dataset>/:

  queries.fasta, references.fasta   header = the id, nothing else
  queries.tsv                       id, split (dev / heldout), group (the bootstrap unit),
                                    label, and the origin columns
  references.tsv                    id, role, label, group, length, sha1 of the sequence,
                                    Swiss-Prot sequence version where there is one

The three datasets are never pooled. Each has its own positive rule (`relation` in
evaluate.py):

  ice_binding     positive = an ice-binding protein of a different origin. One origin is one
                  independent invention of ice binding: the protein's UniProt family when it
                  has one, else its lineage in the notebook 295 truth set. So the bacterial
                  and fungal members of the Ice-binding protein family are one origin, and a
                  hit between them is homology, not convergence (the notebook 298 trap).
                  Same-origin pairs are kept apart and reported as a sanity check.
  toxin_scaffold  positive = same cysteine framework (pattern of cysteines), from the
                  cys-frameworks peptide table.
  toxin_target    positive = same molecular target class (UniProt keyword: Nav, Kv, Cav,
                  nAChR), mature chain only.

Negatives are of three kinds and stay labelled: annotated (another framework or target
class: annotated, never experimentally shown not to bind), hard (composition-matched or
same-family non-ice-binding proteins; cysteine-rich non-toxins), and unannotated
background (random Swiss-Prot). No dataset has experimentally confirmed negatives.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import polars as pl

from alphabets import CANONICAL

CANON = set(CANONICAL)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


def canonical(s: str) -> bool:
    return len(s) > 0 and set(s) <= CANON


def write_fasta(path: Path, rows: list[tuple[str, str]]) -> None:
    with open(path, "w") as fh:
        for i, s in rows:
            fh.write(f">{i}\n")
            for j in range(0, len(s), 60):
                fh.write(s[j : j + 60] + "\n")


def read_fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:].split()[0]
            out[name] = ""
        elif name:
            out[name] += line
    return out


def split_groups(groups: list[str], frac: float, seed: int, salt: str) -> dict[str, str]:
    """Shuffle the sorted groups with a seeded RNG; the first round(frac*n) are dev."""
    g = sorted(set(groups))
    random.Random(f"{seed}-{salt}").shuffle(g)
    n_dev = max(1, round(frac * len(g)))
    return {x: ("dev" if i < n_dev else "heldout") for i, x in enumerate(g)}


def background(sp: pl.DataFrame, random_ids: list[str], exclude: set[str], n: int):
    seq = dict(zip(sp["accession"], sp["sequence"]))
    out = []
    for a in random_ids:
        if a in exclude or a not in seq or not canonical(seq[a]):
            continue
        out.append(a)
        if len(out) == n:
            break
    return out


def _ref_rows(sp_idx: dict, ids: list[str], role: str, **cols) -> list[dict]:
    rows = []
    for a in ids:
        r = sp_idx[a]
        rows.append(
            dict(
                id=a,
                role=role,
                length=len(r["sequence"]),
                sha1=sha1(r["sequence"]),
                seq_version=r["seq_version"],
                family=r["family"],
                organism=r["organism"],
                **cols,
            )
        )
    return rows


def composition(seqs: list[str]) -> np.ndarray:
    idx = {r: i for i, r in enumerate(CANONICAL)}
    m = np.zeros((len(seqs), 20), dtype=np.float32)
    for k, s in enumerate(seqs):
        for r in s:
            j = idx.get(r)
            if j is not None:
                m[k, j] += 1
        m[k] /= max(1, len(s))
    return m


def is_toxin(kw: str) -> bool:
    return "Toxin" in kw.split("; ")


def freeze_ice(cfg, sp: pl.DataFrame, sp_idx: dict, random_ids, outdir: Path) -> None:
    c = cfg["panels"]["ice_binding"]
    truth = pl.read_csv(cfg["inputs"]["ice_truth"], separator="\t", infer_schema_length=0)
    truth = truth.filter(pl.col("Entry").is_in(list(sp_idx)))
    # same rule as every other panel member: only the 20 canonical residues. Those left out
    # are listed, and are neither positive nor negative (they stay out of every pool).
    keep = [canonical(sp_idx[a]["sequence"]) for a in truth["Entry"]]
    dropped = truth.filter(~pl.Series(keep)).select("Entry", "lineage")
    outdir.mkdir(parents=True, exist_ok=True)
    dropped.with_columns(reason=pl.lit("non-canonical residue")).write_csv(
        outdir / "dropped.tsv", separator="\t"
    )
    all_ibp = set(truth["Entry"])
    truth = truth.filter(pl.Series(keep))
    ibp = truth["Entry"].to_list()
    fam = {a: sp_idx[a]["family"] for a in ibp}
    origin = {a: (fam[a] if fam[a] else lin) for a, lin in zip(truth["Entry"], truth["lineage"])}
    # anything else with the Antifreeze keyword is neither positive nor negative: left out
    af_kw = set(sp.filter(pl.col("keywords").str.contains("Antifreeze protein"))["accession"])
    excluded = all_ibp | af_kw
    pool = sp.filter(
        ~pl.col("accession").is_in(list(excluded))
        & pl.col("sequence").map_elements(canonical, return_dtype=pl.Boolean)
    )
    comp_pool = composition(pool["sequence"].to_list())
    lens = pool["length"].to_numpy()
    acc = pool["accession"].to_list()
    hard_comp, hard_fam = [], []
    for a in ibp:
        q = composition([sp_idx[a]["sequence"]])[0]
        L = len(sp_idx[a]["sequence"])
        d = np.abs(comp_pool - q).sum(1)
        d[(lens < 0.5 * L) | (lens > 2 * L)] = np.inf
        for j in np.argsort(d, kind="stable")[: c["n_composition_matched"]]:
            if np.isfinite(d[j]) and acc[j] not in hard_comp:
                hard_comp.append(acc[j])
    for f in sorted({f for f in fam.values() if f}):
        same = pool.filter(pl.col("family") == f).sort("accession")["accession"].to_list()
        hard_fam += [x for x in same[: c["n_same_family"]] if x not in hard_comp]
    bg = background(
        sp, random_ids, excluded | set(hard_comp) | set(hard_fam), cfg["panels"]["n_background"]
    )
    split = split_groups(list(origin.values()), c["dev_fraction"], cfg["seed"], "ice")
    refs = (
        _ref_rows(sp_idx, ibp, "ice_binding")
        + _ref_rows(sp_idx, hard_comp, "hard_negative_composition")
        + _ref_rows(sp_idx, hard_fam, "hard_negative_same_family")
        + _ref_rows(sp_idx, bg, "background")
    )
    for r in refs:
        r["label"] = origin.get(r["id"])
        r["group"] = origin.get(r["id"])
    qrows = [
        dict(
            id=a,
            split=split[origin[a]],
            group=origin[a],
            label=origin[a],
            lineage=lin,
            length=len(sp_idx[a]["sequence"]),
            seq_version=sp_idx[a]["seq_version"],
            sha1=sha1(sp_idx[a]["sequence"]),
        )
        for a, lin in zip(truth["Entry"], truth["lineage"])
    ]
    _write(outdir, qrows, refs, {a: sp_idx[a]["sequence"] for a in sp_idx_ids(refs)}, sp_idx)


def sp_idx_ids(refs):
    return [r["id"] for r in refs]


def _write(outdir: Path, qrows, refs, seqs: dict, sp_idx=None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    ids = [r["id"] for r in refs]
    if len(ids) != len(set(ids)):
        raise ValueError("a reference is listed twice")
    pl.DataFrame(qrows).write_csv(outdir / "queries.tsv", separator="\t")
    pl.DataFrame(refs, infer_schema_length=None).write_csv(
        outdir / "references.tsv", separator="\t"
    )
    write_fasta(outdir / "references.fasta", [(i, seqs[i]) for i in ids])
    write_fasta(outdir / "queries.fasta", [(q["id"], seqs[q["id"]]) for q in qrows])


def cys_rich_nontoxins(cfg, sp: pl.DataFrame, n: int) -> list[str]:
    pool = sp.filter(
        (pl.col("length") <= 120)
        & (pl.col("sequence").str.count_matches("C") >= 6)
        & ~pl.col("keywords").map_elements(is_toxin, return_dtype=pl.Boolean)
        & pl.col("sequence").map_elements(canonical, return_dtype=pl.Boolean)
    ).sort("accession")
    ids = pool["accession"].to_list()
    random.Random(f"{cfg['seed']}-cysrich").shuffle(ids)
    return sorted(ids[:n])


def freeze_scaffold(cfg, sp, sp_idx, random_ids, cysrich, outdir: Path) -> None:
    c = cfg["panels"]["toxin_scaffold"]
    pep = pl.read_csv(cfg["inputs"]["cys_peptides"], infer_schema_length=0)
    pep = pep.filter(pl.col("peptide").map_elements(canonical, return_dtype=pl.Boolean))
    pep = pep.unique("peptide", keep="first", maintain_order=True)
    counts = (
        pep.group_by("framework").len().filter(pl.col("len") >= c["min_peptides_per_framework"])
    )
    frameworks = sorted(counts["framework"].to_list())
    rng = random.Random(f"{cfg['seed']}-scaffold")
    rng.shuffle(frameworks)
    frameworks = sorted(frameworks[: c["n_frameworks"]])
    refs, qrows, seqs = [], [], {}
    split = split_groups(frameworks, c["dev_fraction"], cfg["seed"], "scaffold")
    for f in frameworks:
        sub = pep.filter(pl.col("framework") == f).sort("query_id")
        rows = sub.to_dicts()
        rng.shuffle(rows)
        rows = rows[: c["peptides_per_framework"]]
        for k, r in enumerate(rows):
            rid = r["query_id"].replace(":", "_")
            seqs[rid] = r["peptide"]
            refs.append(
                dict(
                    id=rid, role="toxin", label=f, group=f, length=len(r["peptide"]),
                    sha1=sha1(r["peptide"]), seq_version=None, family=r["framework_label"],
                    organism=r["organism"], source=r["source"], phylum=r["phylum"],
                    genus=r["genus"],
                )
            )  # fmt: skip
            if k < c["queries_per_framework"]:
                qrows.append(
                    dict(
                        id=rid, split=split[f], group=f, label=f, source=r["source"],
                        phylum=r["phylum"], genus=r["genus"], length=len(r["peptide"]),
                        sha1=sha1(r["peptide"]),
                    )
                )  # fmt: skip
    bg_exclude = set()
    refs += _ref_rows(sp_idx, cysrich, "cys_rich_nontoxin")
    bg = background(
        sp, random_ids, bg_exclude | set(cysrich) | toxin_ids(sp), cfg["panels"]["n_background"]
    )
    refs += _ref_rows(sp_idx, bg, "background")
    for a in cysrich + bg:
        seqs[a] = sp_idx[a]["sequence"]
    _write(outdir, qrows, refs, seqs)


def toxin_ids(sp) -> set[str]:
    return set(
        sp.filter(pl.col("keywords").map_elements(is_toxin, return_dtype=pl.Boolean))["accession"]
    )


def mature(seq: str, features: str, min_len: int) -> str | None:
    """Longest PEPTIDE or CHAIN feature. None when there is none of at least min_len."""
    best = None
    for f in features.split("; ") if features else []:
        kind, span = f.split(":")
        a, b = (int(x) for x in span.split("-"))
        s = seq[a - 1 : b]
        if len(s) >= min_len and (best is None or len(s) > len(best)):
            best = s
    return best


def freeze_target(cfg, sp, sp_idx, random_ids, cysrich, outdir: Path) -> None:
    c = cfg["panels"]["toxin_target"]
    kw2cls = {v: k for k, v in c["classes"].items()}
    tox = sp.filter(pl.col("keywords").map_elements(is_toxin, return_dtype=pl.Boolean))
    rows = []
    for r in tox.iter_rows(named=True):
        cls = {kw2cls[k] for k in r["keywords"].split("; ") if k in kw2cls}
        if len(cls) != 1:
            continue
        m = mature(r["sequence"], r["features"], c["min_mature_length"])
        if m is None or not canonical(m):
            continue
        rows.append(
            dict(
                id=r["accession"], cls=cls.pop(), mature=m, family=r["family"] or f"unassigned:{r['accession']}",
                experimental=r["function_experimental"], organism=r["organism"],
                seq_version=r["seq_version"],
            )
        )  # fmt: skip
    df = pl.DataFrame(rows).sort("id").unique("mature", keep="first", maintain_order=True)
    fam_split = split_groups(df["family"].to_list(), c["dev_fraction"], cfg["seed"], "target")
    rng = random.Random(f"{cfg['seed']}-target")
    refs, qrows, seqs = [], [], {}
    for cls in sorted(c["classes"]):
        sub = df.filter(pl.col("cls") == cls).to_dicts()
        exp = [r for r in sub if r["experimental"]]
        other = [r for r in sub if not r["experimental"]]
        rng.shuffle(exp)
        rng.shuffle(other)
        chosen = (exp + other)[: c["references_per_class"]]
        # queries: experimentally supported only, half from dev families, half held-out
        for split in ("dev", "heldout"):
            pick = [r for r in chosen if r["experimental"] and fam_split[r["family"]] == split]
            for r in pick[: c["queries_per_class"] // 2]:
                qrows.append(
                    dict(
                        id=r["id"], split=split, group=r["family"], label=cls,
                        family=r["family"], organism=r["organism"], length=len(r["mature"]),
                        sha1=sha1(r["mature"]),
                    )
                )  # fmt: skip
        for r in chosen:
            seqs[r["id"]] = r["mature"]
            refs.append(
                dict(
                    id=r["id"], role="toxin", label=cls, group=r["family"],
                    length=len(r["mature"]), sha1=sha1(r["mature"]),
                    seq_version=r["seq_version"], family=r["family"], organism=r["organism"],
                    experimental=r["experimental"],
                )
            )  # fmt: skip
    taken = {r["id"] for r in refs}
    cys = [a for a in cysrich if a not in taken]
    refs += _ref_rows(sp_idx, cys, "cys_rich_nontoxin")
    bg = background(sp, random_ids, taken | set(cys) | toxin_ids(sp), cfg["panels"]["n_background"])
    refs += _ref_rows(sp_idx, bg, "background")
    for a in cys + bg:
        seqs[a] = sp_idx[a]["sequence"]
    _write(outdir, qrows, refs, seqs)
