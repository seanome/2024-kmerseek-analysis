#!/usr/bin/env python3
"""Sequences, structure covariates and pipeline inputs for the ELM motif-transfer benchmark.

Reads elm_instances_tp.parquet (scripts/fetch_elm.py) and writes into --elm-dir:

  elm_proteins.fasta
      Every protein with a true-positive instance, from the UniProt REST `accessions`
      endpoint, which also serves isoform accessions (Q9Y6K9-2). Headers are UniProt's own
      (`sp|ACC|NAME OS=...`), the form every tool output in the pipeline is parsed from.

  elm_instance_checks.parquet
      One row per instance: whether it lies inside the current UniProt sequence and whether
      the class regex matches a stretch overlapping it. ELM records coordinates against the
      sequence it curated from, and UniProt sequences change between releases, so an
      instance whose regex no longer matches there may no longer point at the motif.
      `usable` = both checks pass; only usable instances are scored.

  elm_instance_covariates.parquet
      Per instance: mean pLDDT over the motif's residues, from the AlphaFold model
      (bin/build_query_covariates.read_plddt), and the fraction of motif residues
      metapredict 3.0.1 calls disordered at threshold 0.5
      (bin/predict_disorder_metapredict.py). An
      isoform has no AlphaFold model, and a model whose sequence differs from the UniProt
      sequence is not sliced; both get a null pLDDT, not a dropped row.

  pipeline/
      The inputs qfo-pfam-region-benchmark reads, laid out the way midi-plus lays them out:
        qfo/Eukaryota/UP000005640_9606.fasta   human ELM proteins: the queries
        qfo/ELM/ELMPOOL_0.fasta                every non-human ELM protein: the targets
        qfo/ELM/ELMCHICKEN_9031.fasta          the chicken ones only, for the dry run
        elm_species.tsv                        species registry naming the two targets
        annotations/human_pfam_domains.parquet the human Pfam table, cut to the queries;
                                               buildDomainTruth needs it, and the ELM
                                               targets get none, so the pipeline searches
                                               them and scores nothing
        annotations/no_target_pfam_domains.parquet  an empty table for a label that is never
                                               searched (see the comment where it is written)
        structures/<label>/AF-<acc>-F1.cif     links into the flat AlphaFold cache

AlphaFold models missing from the cache are resolved through the AFDB API
(https://alphafold.ebi.ac.uk/api/prediction/<acc>), which names the current model file;
the model version is not assumed.

Usage:
    python scripts/prep_elm_inputs.py --elm-dir /Users/olga/data/elm-motif-transfer
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PIPELINE = REPO / "nextflow-runs" / "qfo-pfam-region-benchmark"
sys.path.insert(0, str(PIPELINE / "bin"))
from build_query_covariates import read_plddt  # noqa: E402

UNIPROT_ACCESSIONS_URL = (
    "https://rest.uniprot.org/uniprotkb/accessions?accessions={}&format=fasta"
)
AFDB_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{}"
DISORDER_THRESHOLD = 0.5
HUMAN = "Homo sapiens"
CHICKEN = "Gallus gallus"

DEFAULT_ELM_DIR = Path("/Users/olga/data/elm-motif-transfer")
DEFAULT_AF_CACHE = Path("/Users/olga/data/alphafold_structures")
DEFAULT_HUMAN_PFAM = (
    REPO.parent
    / "2024-kmerseek-analysis"
    / "results"
    / "pfam_benchmark"
    / "annotations"
    / "human_pfam_domains.parquet"
)


def read_fasta(path: Path) -> dict[str, tuple[str, str]]:
    """accession -> (header, sequence)."""
    out, header, buf = {}, None, []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header:
                out[accession_of(header)] = (header, "".join(buf))
            header, buf = line[1:], []
        else:
            buf.append(line.strip())
    if header:
        out[accession_of(header)] = (header, "".join(buf))
    return out


def accession_of(header: str) -> str:
    parts = header.split()[0].split("|")
    return parts[1] if len(parts) >= 2 else parts[0]


def write_fasta(records: dict[str, tuple[str, str]], keep, out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as fh:
        for acc in sorted(keep):
            if acc in records:
                h, s = records[acc]
                fh.write(f">{h}\n")
                for i in range(0, len(s), 60):
                    fh.write(s[i : i + 60] + "\n")
                n += 1
    return n


def fetch_sequences(accessions: list[str], out: Path) -> dict[str, tuple[str, str]]:
    have = read_fasta(out) if out.exists() else {}
    missing = [a for a in accessions if a not in have]
    if missing:
        print(f"fetching {len(missing):_} sequences from UniProt")
        chunks = []
        for i in range(0, len(missing), 100):
            batch = missing[i : i + 100]
            chunks.append(
                urlread(
                    UNIPROT_ACCESSIONS_URL.format(",".join(batch)), timeout=120
                ).decode()
            )
        tmp = out.with_suffix(".new.fasta")
        tmp.write_text("".join(chunks))
        have.update(read_fasta(tmp))
        tmp.unlink()
        write_fasta(have, have.keys(), out)
    absent = sorted(set(accessions) - set(have))
    print(
        f"sequences: {len(have):_} of {len(accessions):_} accessions"
        + (
            f"; UniProt returned nothing for {len(absent)}: {absent[:10]}"
            if absent
            else ""
        )
    )
    return have


def regex_at_instance(regex: str | None, seq: str, start: int, end: int) -> bool:
    """True if the class regex matches a stretch of `seq` overlapping [start, end)."""
    if not regex:
        return False
    pat = re.compile(regex)
    for i in range(max(0, start - 60), min(len(seq), end)):
        m = pat.match(seq, i)
        if m and m.end() > start and m.start() < end and m.end() > m.start():
            return True
    return False


def urlread(url: str, timeout: int, tries: int = 5) -> bytes:
    """GET with retries on timeouts and 5xx; a 404 is raised at once (it is an answer)."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def resolve_model(acc: str, cache: Path) -> tuple[Path | None, str | None]:
    """(cif path, AFDB's uniprotSequence) for a canonical accession; (None, None) if AFDB has none."""
    cached = sorted(cache.glob(f"AF-{acc}-F1-model_v*.cif"))
    meta = cache / "_afdb_meta" / f"{acc}.json"
    if cached and meta.exists():
        return cached[-1], json.loads(meta.read_text()).get("uniprotSequence")
    try:
        entries = json.loads(urlread(AFDB_API_URL.format(acc), timeout=60))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None, None
        raise
    entry = next(
        (e for e in entries if e.get("uniprotAccession") == acc),
        entries[0] if entries else None,
    )
    if not entry:
        return None, None
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(
        json.dumps(
            {k: entry.get(k) for k in ("cifUrl", "uniprotSequence", "latestVersion")}
        )
    )
    if cached:
        return cached[-1], entry.get("uniprotSequence")
    url = entry["cifUrl"]
    dest = cache / url.rsplit("/", 1)[1]
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(urlread(url, timeout=300))
    tmp.rename(dest)
    return dest, entry.get("uniprotSequence")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--elm-dir", type=Path, default=DEFAULT_ELM_DIR)
    ap.add_argument("--af-cache", type=Path, default=DEFAULT_AF_CACHE)
    ap.add_argument(
        "--human-pfam",
        type=Path,
        default=DEFAULT_HUMAN_PFAM,
        help="human_pfam_domains.parquet from the Pfam benchmark annotations",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    d = args.elm_dir

    inst = pl.read_parquet(d / "elm_instances_tp.parquet")
    accs = sorted(inst["accession"].unique().to_list())
    seqs = fetch_sequences(accs, d / "elm_proteins.fasta")

    # ---- instance checks ----
    rows = []
    for r in inst.iter_rows(named=True):
        seq = seqs.get(r["accession"], (None, ""))[1]
        in_range = bool(seq) and r["end"] <= len(seq)
        rows.append(
            {
                "elm_instance": r["elm_instance"],
                "have_sequence": bool(seq),
                "in_range": in_range,
                "regex_at_instance": in_range
                and regex_at_instance(r["regex"], seq, r["start"], r["end"]),
                "protein_length": len(seq) if seq else None,
            }
        )
    checks = pl.DataFrame(rows).with_columns(
        usable=pl.col("in_range") & pl.col("regex_at_instance")
    )
    checks.write_parquet(d / "elm_instance_checks.parquet")
    print(
        f"instance checks: {checks.height:_} instances; "
        f"no sequence {(~checks['have_sequence']).sum()}, "
        f"outside the sequence {(checks['have_sequence'] & ~checks['in_range']).sum()}, "
        f"regex does not match there {(checks['in_range'] & ~checks['regex_at_instance']).sum()}, "
        f"usable {checks['usable'].sum():_}"
    )

    # ---- AlphaFold models + pLDDT ----
    canonical = [a for a in accs if "-" not in a and a in seqs]
    with ThreadPoolExecutor(args.workers) as ex:
        models = dict(
            zip(canonical, ex.map(lambda a: resolve_model(a, args.af_cache), canonical))
        )
    tracks, model_ok = {}, {}
    for acc, (path, afdb_seq) in models.items():
        ok = path is not None and afdb_seq == seqs[acc][1]
        model_ok[acc] = ok
        if ok:
            tracks[acc] = read_plddt(path)
    print(
        f"AlphaFold: {sum(p is not None for p, _ in models.values()):_} of {len(canonical):_} canonical "
        f"accessions have a model; {sum(model_ok.values()):_} match the UniProt sequence "
        f"({len(accs) - len(canonical)} isoform accessions have none)"
    )

    plddt_rows = []
    for r in inst.iter_rows(named=True):
        t = tracks.get(r["accession"])
        w = t[r["start"] : r["end"]] if t and r["end"] <= len(t) else []
        plddt_rows.append(
            {
                "elm_instance": r["elm_instance"],
                "mean_plddt_motif": sum(w) / len(w) if w else None,
            }
        )

    # ---- metapredict over each motif ----
    dom = inst.select(
        "accession",
        (pl.col("start") + 1)
        .cast(pl.Int32)
        .alias("domain_start"),  # the script's 1-based inclusive
        pl.col("end").cast(pl.Int32).alias("domain_end"),
    ).unique()
    work = d / "metapredict"
    work.mkdir(exist_ok=True)
    dom.write_parquet(work / "motifs.parquet")
    (work / "elm_proteins.fasta").write_text((d / "elm_proteins.fasta").read_text())
    out_dom = work / "motif_disorder.parquet"
    if not out_dom.exists():
        # Run with this interpreter, which must have metapredict 3.0.1 (the
        # 2025-kmerseek-analysis env does). The olgabot/metapredict image is amd64 and its
        # polars needs AVX, which Rosetta does not provide, so on Apple Silicon it crashes.
        subprocess.run(
            [
                sys.executable,
                str(PIPELINE / "bin" / "predict_disorder_metapredict.py"),
                "--fasta",
                str(work / "elm_proteins.fasta"),
                "--out",
                str(work / "protein_disorder.parquet"),
                "--domains",
                str(work / "motifs.parquet"),
                "--domains-out",
                str(out_dom),
                "--threshold",
                str(DISORDER_THRESHOLD),
            ],
            check=True,
        )
    dis = pl.read_parquet(out_dom)
    dis = dis.select(
        "accession",
        (pl.col("domain_start") - 1).cast(pl.Int64).alias("start"),
        pl.col("domain_end").cast(pl.Int64).alias("end"),
        pl.col("disorder_fraction_region").alias("frac_disordered_motif"),
        pl.col("mean_disorder_region").alias("mean_disorder_score_motif"),
    )

    cov = (
        inst.select(
            "elm_instance",
            "elm_class",
            "elm_type",
            "accession",
            "start",
            "end",
            "motif_length",
            "organism",
        )
        .join(pl.DataFrame(plddt_rows), on="elm_instance", how="left")
        .join(dis, on=["accession", "start", "end"], how="left")
        .join(checks.select("elm_instance", "usable"), on="elm_instance", how="left")
    )
    cov.write_parquet(d / "elm_instance_covariates.parquet")
    u = cov.filter("usable")
    print("\nusable instances, per-motif structure covariates:")
    print(
        u.select(
            n=pl.len(),
            median_length=pl.col("motif_length").median(),
            n_with_plddt=pl.col("mean_plddt_motif").is_not_null().sum(),
            median_plddt=pl.col("mean_plddt_motif").median(),
            frac_plddt_below_50=(pl.col("mean_plddt_motif") < 50).mean(),
            frac_plddt_below_70=(pl.col("mean_plddt_motif") < 70).mean(),
            median_frac_disordered=pl.col("frac_disordered_motif").median(),
            frac_mostly_disordered=(pl.col("frac_disordered_motif") >= 0.5).mean(),
        )
    )

    # ---- pipeline inputs ----
    pdir = d / "pipeline"
    orgs = inst.select("accession", "organism").unique()
    human_accs = set(orgs.filter(pl.col("organism") == HUMAN)["accession"])
    other_accs = set(orgs.filter(pl.col("organism") != HUMAN)["accession"])
    chicken_accs = set(orgs.filter(pl.col("organism") == CHICKEN)["accession"])
    both = human_accs & other_accs
    if both:
        print(
            f"note: {len(both)} accessions are labelled both human and non-human in ELM: {sorted(both)[:5]}"
        )
    n_h = write_fasta(seqs, human_accs, pdir / "qfo/Eukaryota/UP000005640_9606.fasta")
    n_p = write_fasta(seqs, other_accs - human_accs, pdir / "qfo/ELM/ELMPOOL_0.fasta")
    n_c = write_fasta(
        seqs, chicken_accs - human_accs, pdir / "qfo/ELM/ELMCHICKEN_9031.fasta"
    )
    (pdir / "elm_species.tsv").write_text(
        "label\ttaxon\tproteome\tsubdir\tmya\tn_proteins\tscientific_name\n"
        "human\t9606\tUP000005640\tEukaryota\t0\t{}\tHomo sapiens\n"
        "elm_pooled\t0\tELMPOOL\tELM\t\t{}\tevery non-human ELM protein\n"
        "elm_chicken\t9031\tELMCHICKEN\tELM\t\t{}\tGallus gallus ELM proteins\n".format(
            n_h, n_p, n_c
        )
    )
    ann = pdir / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    hp = pl.read_parquet(args.human_pfam)
    hp.filter(pl.col("accession").is_in(list(human_accs))).write_parquet(
        ann / "human_pfam_domains.parquet"
    )
    # buildDomainTruth declares `*_domain_map.parquet` as a required output, and writes one
    # map per target table it finds. The ELM targets have no Pfam table, so an empty one
    # for a label that is never searched satisfies the glob and scores nothing.
    hp.clear().write_parquet(ann / "no_target_pfam_domains.parquet")
    for label, keep in (
        ("human", human_accs),
        ("elm_pooled", other_accs - human_accs),
        ("elm_chicken", chicken_accs - human_accs),
    ):
        sd = pdir / "structures" / label
        sd.mkdir(parents=True, exist_ok=True)
        n = 0
        for acc in keep:
            path, _ = models.get(acc, (None, None))
            if path is not None and model_ok.get(acc):
                link = sd / f"AF-{acc}-F1.cif"
                if not link.exists():
                    link.symlink_to(path)
                n += 1
        print(f"structures/{label}: {n} of {len(keep)} proteins")
    print(
        f"pipeline inputs in {pdir}: {n_h} human queries, {n_p} pooled targets, {n_c} chicken targets"
    )


if __name__ == "__main__":
    main()
