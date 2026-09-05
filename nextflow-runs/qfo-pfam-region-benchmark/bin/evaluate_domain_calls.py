#!/usr/bin/env python3
"""Turn aligned regions into Pfam domain calls and score them against the answer key.

The question is domain finding, not orthology: for a human query protein, which stretch of
it is a given Pfam family, and did the tool put the region in the right place?

Pipeline for a search-based tool:

  1. Read the tool's regions: (query, target, qstart, qend, tstart, tend, score).
  2. Transfer. Look up the target protein's Pfam domains. A region claims a family when it
     covers at least --min-overlap of that domain on the *target* side. The region's query
     interval, carrying that family label, is now a domain call.
  3. Score. A call is a true positive when the query protein has that family
     AND the call's interval reciprocally overlaps (IoU) a real instance of it by at least
     --min-overlap. Right family in the wrong place is a false positive, not a hit -- that
     distinction is the whole reason for scoring regions instead of protein pairs.

With --direct-annotation (hmmscan against Pfam-A) step 2 is skipped: the tool already names
the family, so its intervals are query-side calls as they stand.

Two overlap criteria, deliberately different:
  transfer (target side)  overlap / domain_length -- did the region land on that domain
  scoring  (query side)   IoU = overlap / union   -- is the call in the right place
Transfer is the looser of the two on purpose. Being strict there would throw away correct
families over target-side boundary noise, which is not what is being measured.
"""

import argparse
import bz2
import gzip
import json
import lzma
import shutil
import subprocess
import sys
from pathlib import Path

import polars as pl

# bin/ is on PATH under Nextflow but not on PYTHONPATH, so make the sibling module
# importable regardless of the working directory the task runs in.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cafa_metrics as cm  # noqa: E402
# The Swiss-Prot FT vocabulary, imported rather than re-listed. A second copy of those
# twelve-odd type names is a thing that drifts, and the failure would be silent: an FT type
# added to the truth builder but missing here would make the feature_type axis null for the
# whole run rather than raise.
import build_swissprot_truth as sprot  # noqa: E402


# Covariate axes the results get cut by. Continuous ones are binned; HGNC gene group is
# already categorical. Bin edges are fixed rather than data-derived so a stratum means
# the same thing across every tool, species and combo in the sweep.
# Percent-identity bins. 30-40% is the twilight zone where profile methods lose the
# signal and where claim 1 says HP patterning still holds; <20% is the midnight zone
# below it. Bins are fixed rather than quantile-derived so a stratum means the same thing
# in every species, which is the whole point of comparing across a divergence panel.
IDENTITY_BINS = [0.0, 20.0, 30.0, 40.0, 60.0, 100.01]

# Feature-length bins, in residues, half-open [lo, hi) on `domain_end - domain_start` --
# the same length convention boundary_metrics already sums over, so the two agree.
#
# The unit is the ANNOTATION's own length, not the protein's. A 21-residue TRANSMEM helix
# and a 400-residue kinase domain in the same protein are different measurement problems
# for a k-mer method: at k=19 the helix contains three k-mers and the kinase 380, so an
# alphabet that needs a long window to carry information cannot address the short feature
# at all. Averaging the two hides the only gradient the reduced-alphabet question turns on.
#
# build_swissprot_truth widens point features by one residue so an interval exists at all,
# which puts every point feature in the leading `1` bin on its own. That bin is excluded
# from the boundary metrics -- see cafa_metrics.boundary_metrics(exclude_points=).
FEATURE_LENGTH_BINS = [1, 2, 16, 31, 61, 121, 251]

STRATA = {
    "plddt": ("mean_plddt", [0, 50, 70, 90, 100]),
    "disorder": ("disorder_fraction_plddt", [0.0, 0.1, 0.3, 0.6, 1.01]),
    # Same bins as the pLDDT proxy on purpose, so the two axes are read side by side and a
    # disagreement between them is visible rather than buried in different binning.
    "disorder_seq": ("disorder_fraction_metapredict", [0.0, 0.1, 0.3, 0.6, 1.01]),
    "omega": ("omega", [0.0, 0.1, 0.25, 0.5, 10.0]),
}
# Cutting on every one of ~4200 HGNC groups would produce mostly single-protein strata
# where no metric is stable. Only groups with at least this many query proteins are cut.
MIN_STRATUM_PROTEINS = 30

# Axes whose stratum vocabulary is fixed and biologically defined rather than data-derived,
# so the noise floor above must not delete a cut for being rare. The floor exists to stop
# ~4200 HGNC groups from producing mostly single-protein strata; it has nothing to protect
# against here. `mhc` is 7 curated classes, `geneset` 6 curated sets, `identity` 6 fixed
# bins, `feature_length_bin` 7 fixed bins, and `feature_type` the ~12-name Swiss-Prot FT
# vocabulary -- where rarity is a property of the feature type itself, not a sampling
# accident. ACT_SITE and DNA_BIND are small in every proteome that will ever be measured,
# and dropping them would delete the short-feature end of the very gradient being tested.
# Every row reports its own n_stratum_proteins and n_truth_instances either way.
UNFLOORED_AXES = ("mhc", "geneset", "identity", "feature_length_bin", "feature_type")

# The vocabulary attach_feature_type recognises, from the truth builder itself.
FEATURE_TYPES = sprot.RANGE_FEATURES | sprot.POINT_FEATURES

# Boolean covariate columns that each become their own stratum, so the 200-series' curated
# gene sets are cut out of the box rather than reconstructed in a notebook.
GENE_SET_FLAGS = {
    "mhc_class_i_heavy": "is_mhc_class_i_heavy",
    "antiviral_restriction_factor": "is_antiviral_restriction_factor",
    "igsf_decoy": "is_igsf_decoy",
    "fast_evolving_family": "is_fast_evolving_family",
    "olfactory_receptor": "is_olfactory_receptor",
    "cytochrome_p450_2_3": "is_cytochrome_p450_2_3",
    # Kept as measurable strata even though they are excluded from the HGNC sweep: how
    # repeat-driven families behave is a result, and deleting them forfeits it.
    "zinc_finger_c2h2": "is_zinc_finger_c2h2",
    "zinc_finger_other": "is_zinc_finger_other",
}


def extract_accession(col: pl.Expr) -> pl.Expr:
    """UniProt FASTA names are sp|P12345|NAME_SPECIES; annotations key on the accession.
    Names already bare (Foldseek, after its filename cleanup) pass through untouched."""
    return (
        pl.when(col.str.contains(r"\|"))
        .then(col.str.split("|").list.get(1, null_on_oob=True))
        .otherwise(col)
    )


# Compressed inputs already inflated in this process, keyed by source path. Every arm is
# read once per dedup mode, so without this the same file is inflated twice.
_INFLATED: dict[Path, Path] = {}


def inflate_for_scan(path: Path) -> Path:
    """Inflate a compressed CSV to a plain file next to the task's other work.

    polars' CSV reader does NOT stream a compressed source: it decompresses the whole file
    into memory before parsing. Measured on this project's data at up to 97x the compressed
    size (1.9 GB -> 184.7 GB), which is how a scan that looks lazy OOM-kills a task. Every
    baseline arm here arrives as .tsv.gz, and each is read more than once -- the schema
    probe, the scan, then both again for the second dedup mode -- so each read paid that
    cost over again.

    Inflating to disk trades RAM for scratch space: polars memory-maps a plain file and
    reads it in chunks. Returns the original path unchanged when the file is not compressed,
    or when nothing here can decompress it, so a failure here degrades to the old behaviour
    rather than losing the arm.
    """
    openers = {".gz": gzip.open, ".bz2": bz2.open, ".xz": lzma.open}
    if path.suffix not in openers and path.suffix not in (".zst", ".zstd"):
        return path

    key = path.resolve()
    cached = _INFLATED.get(key)
    if cached is not None and cached.exists():
        return cached

    out = Path(f"{path.name}.inflated")
    try:
        if path.suffix in openers:
            with openers[path.suffix](path, "rb") as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 22)
        else:
            with open(out, "wb") as dst:
                subprocess.run(["zstd", "-dc", str(path)], stdout=dst, check=True)
    except (OSError, subprocess.SubprocessError):
        out.unlink(missing_ok=True)
        return path

    print(f"inflated {path.name} ({path.stat().st_size / 1e6:.1f} MB) -> "
          f"{out.name} ({out.stat().st_size / 1e6:.1f} MB) so polars can stream it",
          file=sys.stderr)
    _INFLATED[key] = out
    return out


def release_inflated(path: Path) -> None:
    """Delete an arm's inflated copy once nothing will read it again.

    A batched task scores ~415 arms in one directory. Holding every inflated baseline until
    the process exits would leave tens of GB of scratch standing for files already scored,
    which on a shared $SCRATCH is its own way to kill a long task.
    """
    out = _INFLATED.pop(path.resolve(), None)
    if out is not None:
        out.unlink(missing_ok=True)


def load_regions(path: Path, direct: bool, rank_by: str = "region_enrichment",
                 max_bonferroni_p: float | None = 0.05) -> pl.LazyFrame | None:
    """Normalize any tool's output to one schema. Returns None for an empty result, which
    is a real outcome (a combo that found nothing), not an error.

    An empty gzip or zstd stream is NOT a zero-byte file -- both carry a frame header --
    so the size check alone cannot catch a tool that legitimately found nothing. polars
    raises NoDataError on those, which is caught here rather than at each call site.

    rank_by and max_bonferroni_p apply to kmerseek only; every other tool ranks on the
    score column its own output already carries.
    """
    try:
        return _load_regions(path, direct, rank_by, max_bonferroni_p)
    except pl.exceptions.NoDataError:
        return None


def _load_regions(path: Path, direct: bool, rank_by: str = "region_enrichment",
                  max_bonferroni_p: float | None = 0.05) -> pl.LazyFrame | None:
    if path.stat().st_size == 0:
        return None

    if path.suffix == ".parquet":
        # kmerseek. region_start/region_end are query-side; target_start/target_end are
        # target-side.
        #
        # Ranking is by `region_enrichment`, not by region_poisson_score. kmerseek's source is
        # explicit that the Poisson score is "a heuristic score for ranking candidate
        # regions against each other, not as a calibrated probability", for two reasons the
        # -log10 transform does not fix: n_shared is arithmetic on the region's own length,
        # which is the quantity find_matched_regions chose by keeping the longest gapless
        # run (close to circular), and the k-mers counted overlap by ksize-1 residues, so
        # they are not the independent trials the Poisson model assumes.
        #
        # Enrichment keeps that same numerator but divides by region_expected_shared_kmers
        # instead of pushing it through a Poisson tail, so it drops the independence
        # assumption while keeping the informative part: how many more k-mers matched than
        # the target DB's composition predicts. It is region-scoped, which matters because
        # the unit of this benchmark is the domain interval.
        #
        # jaccard is available and is deliberately NOT the default. It is a whole
        # query-target statistic, so every region of a protein pair carries the same value
        # and kmerseek would rank proteins while the aligners rank regions. The PR curve
        # groups by score so those ties carry no ordering bias, but the discrimination is
        # genuinely gone.
        lf = pl.scan_parquet(path)
        names = lf.collect_schema().names()
        if not names:
            return None

        # What each ranking column actually is, read off kmerseek's own source rather than
        # inferred from the name, because two of them are less independent than they look:
        #
        #   jaccard                 intersection/union over the WHOLE query-target pair.
        #                           Every region of a pair carries the same value, so this
        #                           ranks proteins, not regions.
        #   region_enrichment       fold_enrichment(n_shared, lambda) = n_shared /
        #                           region_expected_shared_kmers. Region-scoped, and the
        #                           denominator carries target-DB composition, so a long
        #                           region in a k-mer-rich neighbourhood is discounted.
        #   region_n_shared_kmers   NOT an independent count. search.rs computes it as
        #                           `region.length - ksize + 1`, pure arithmetic on the
        #                           region's own length, so ranking by it is ranking by
        #                           region length and nothing else.
        #   region_poisson_score    -log10 of the Poisson tail on that same n_shared.
        #
        # region_enrichment and region_poisson_score share a numerator with
        # region_n_shared_kmers and differ only in how they normalise it, so they are not
        # four independent hypotheses; they are one count under three normalisations plus
        # one whole-protein statistic.
        if rank_by not in names:
            raise SystemExit(
                f"{path} has no `{rank_by}` column, so --kmerseek-rank-by {rank_by} cannot "
                f"be applied. Columns present: {sorted(names)}. A file written by an older "
                f"kmerseek may predate the field; re-run the search arm or pick another."
            )
        score_col = rank_by

        # Bonferroni correction, using the recipe kmerseek's own source prescribes: convert
        # the region tail back to a probability and multiply by how many positions the
        # region could have started at (region_search_space) and how many targets were
        # searched (db_n_targets). run_n_queries is deliberately NOT included -- kmerseek
        # reports these counts separately so that no statistic changes depending on batch
        # composition, and a per-query call should not get harder to make because someone
        # searched more queries alongside it.
        #
        # region_tail_probability is preferred over inverting region_poisson_score: it is
        # the same number without a round trip through -log10, and kmerseek documents it as
        # being reported precisely so downstream tools do not have to invert.
        needed = {"region_search_space", "db_n_targets"}
        can_correct = needed.issubset(names) and (
            "region_tail_probability" in names or "region_poisson_score" in names
        )
        if max_bonferroni_p is not None and not can_correct:
            missing = sorted((needed | {"region_tail_probability"}) - set(names))
            raise SystemExit(
                f"{path} is missing {missing}, so the Bonferroni filter cannot be applied. "
                f"Pass --kmerseek-max-bonferroni-p 0 to disable it, or re-run the search "
                f"arm with a kmerseek that reports the search-space counts."
            )

        if max_bonferroni_p is not None:
            raw_p = (pl.col("region_tail_probability").cast(pl.Float64)
                     if "region_tail_probability" in names
                     else (10.0 ** -pl.col("region_poisson_score").cast(pl.Float64)))
            n_tests = (pl.col("region_search_space").cast(pl.Float64)
                       * pl.col("db_n_targets").cast(pl.Float64))
            # Bonferroni caps at 1: a corrected probability above 1 is still just "not
            # significant", and letting it exceed 1 would be meaningless.
            lf = lf.filter(
                pl.min_horizontal(raw_p * n_tests, pl.lit(1.0)) < max_bonferroni_p
            )

        return lf.select(
            extract_accession(pl.col("query_name")).alias("query_acc"),
            extract_accession(pl.col("target_name")).alias("target_acc"),
            pl.col("region_start").cast(pl.Int64).alias("qstart"),
            pl.col("region_end").cast(pl.Int64).alias("qend"),
            pl.col("target_start").cast(pl.Int64).alias("tstart"),
            pl.col("target_end").cast(pl.Int64).alias("tend"),
            pl.col(score_col).cast(pl.Float64).alias("score"),
        )

    # Everything below this point is CSV, so it is the path that has to be inflated first.
    # The size check above deliberately stays on the ORIGINAL file: an empty gzip stream is
    # not a zero-byte file, and its inflated form is, which scan_csv reports as NoDataError
    # exactly as it did before.
    path = inflate_for_scan(path)

    if direct:
        cols = ["query", "pfam_id", "qstart", "qend", "score", "evalue"]
        lf = pl.scan_csv(path, separator="\t", has_header=False, new_columns=cols)
        return lf.select(
            extract_accession(pl.col("query")).alias("query_acc"),
            # hmmscan reports versioned Pfam accessions (PF00001.24); the tables key on
            # the unversioned id.
            pl.col("pfam_id").str.split(".").list.get(0).alias("pfam_id"),
            pl.col("qstart").cast(pl.Int64),
            pl.col("qend").cast(pl.Int64),
            pl.col("score").cast(pl.Float64),
        )

    # 8 columns for the aligners; motif tools (Folddisco) append a 9th holding how many
    # residues matched, so the envelope's density survives into the metrics.
    # Widths are read off the file rather than declared, so one loader serves both.
    cols = ["query", "target", "qstart", "qend", "tstart", "tend", "score", "evalue",
            "n_matched_residues"]
    probe = pl.scan_csv(path, separator="\t", has_header=False, infer_schema_length=0)
    width = len(probe.collect_schema().names())
    # Every column read as text, and cast here instead. polars infers a column's dtype from
    # the first rows and then fails the WHOLE file on the first row that disagrees: one bad
    # line 105 KB into a 1.2 GB hhblits table killed a task 829 arms in, with a message
    # naming a byte offset and no tool. Whether a row is usable is a question about the
    # tool's output, and it belongs here where the answer can name the arm.
    lf = pl.scan_csv(path, separator="\t", has_header=False,
                     new_columns=cols[:width], infer_schema_length=0)
    selection = [
        extract_accession(pl.col("query")).alias("query_acc"),
        extract_accession(pl.col("target")).alias("target_acc"),
        pl.col("qstart").cast(pl.Int64, strict=False),
        pl.col("qend").cast(pl.Int64, strict=False),
        pl.col("tstart").cast(pl.Int64, strict=False),
        pl.col("tend").cast(pl.Int64, strict=False),
        pl.col("score").cast(pl.Float64, strict=False),
    ]
    if width >= 9:
        selection.append(pl.col("n_matched_residues").cast(pl.Int64, strict=False))
    lf = lf.select(selection)

    # A row whose coordinates did not parse carries a number from the wrong column, so it
    # is not a hit that can be placed anywhere -- see the shifted-field note on the awk in
    # hhblitsSearch. Dropping it is the only thing to do with it, but dropping it QUIETLY
    # is not: the count belongs in the log next to the arm it came from.
    #
    # Not an error, deliberately. A batched task scores hundreds of arms, and one tool's
    # malformed rows must not cost the rest of them their metrics. Every row failing is a
    # different thing -- the file's whole layout is wrong rather than some of its rows --
    # and that does raise, for the same reason hhblitsSearch treats zero hits as a failure.
    #
    # A reversed interval is the same corruption wearing a number that parses. The observed
    # hhblits row shifted by four fields and put a FRACTION in qstart, which is obvious; a
    # shift of one or two puts targetLen or mismatch there instead, and those are integers.
    # end < start is the check that catches those, and no aligner here reports a hit
    # backwards -- every one of them counts residues up. Left in, such a row scores as a
    # false positive rather than as the broken row it is: overlap clips to zero and the IoU
    # guard reads the negative union as no overlap, so the tool is quietly penalised for its
    # writer.
    bad = pl.any_horizontal(
        pl.col("query_acc").is_null(), pl.col("target_acc").is_null(),
        pl.col("qstart").is_null(), pl.col("qend").is_null(),
        pl.col("tstart").is_null(), pl.col("tend").is_null(),
        pl.col("score").is_null(),
        pl.col("qend") < pl.col("qstart"),
        pl.col("tend") < pl.col("tstart"),
    )
    counts = lf.select(pl.len().alias("total"),
                       bad.sum().alias("bad")).collect().row(0, named=True)
    if counts["bad"]:
        if counts["bad"] == counts["total"]:
            raise SystemExit(
                f"every one of {counts['total']} rows in {path.name} has an unparseable "
                f"coordinate or score, so the file's column layout is wrong rather than "
                f"some of its rows. Check the writer for this arm before scoring it."
            )
        print(f"WARNING {path.name}: dropped {counts['bad']} of {counts['total']} rows "
              f"({100 * counts['bad'] / counts['total']:.2f}%) whose coordinates or score "
              f"did not parse, or whose interval ran backwards. These carry a value from "
              f"the wrong column.", file=sys.stderr)
        lf = lf.filter(~bad)
    return lf


# Peak memory of a pairwise suppression pass is set by the JOINED frame, whose height is
# the sum of n^2 over the join-key groups -- not by the height of the table, and not by
# anything the input file size can predict. A sensitive baseline against a whole proteome
# puts thousands of calls of one family on one query protein, so a table polars reads in a
# few GB can ask for billions of pair rows. That is what killed scoreDomainCalls on the
# ecoli batch: the arm before it deduped 1.06 million calls fine, the next one was bigger,
# and the task was OOM-killed with no metric written for any of the 818 arms behind it.
#
# The join is therefore run in batches holding roughly this many pair rows. A group is never
# split across batches, so one group whose own n^2 exceeds the budget still goes through
# whole -- the budget bounds the common case, it is not a hard ceiling.
DEDUP_PAIR_BUDGET = 20_000_000


def _suppress_overlapping(df: pl.DataFrame, keys: list[str],
                          intervals: list[tuple[str, str]], iou_min: float,
                          pair_budget: int) -> pl.DataFrame:
    """Drop every row beaten by another row in its group -- the pass both dedups share.

    A row is beaten when another row with the same `keys` overlaps it at IoU >= iou_min on
    EVERY interval in `intervals`, and outranks it by score, ties broken by original row
    order. Suppression is pairwise, not iterative: a row is dropped if any overlapping row
    scores higher, even if that row is itself dropped.

    Eager on purpose. The pass is a self-join, so the table is materialised whatever it is
    handed, and a LAZY frame here was not merely slower -- it was wrong. The row index the
    anti-join removes by is computed inside the plan, the plan is re-executed for each
    collect, and a re-executed scan does not hand back its rows in the same order, so the
    indices no longer name the same rows. On a file-backed arm the effect was visible:
    identical reruns of the same input kept 16, then 6, then 17 of the same 30 surviving
    calls, while the log line printed 30 every time because that count came from a different
    execution than the file did.
    """
    all_cols = [c for pair in intervals for c in pair]
    c = df.with_row_index("_rid")

    # Exact duplicates go first, and this is not a micro-optimisation. Homology transfer
    # emits the same interval once per target carrying the family, so most of a large group
    # is rows identical but for their score. The pairwise rule already drops every one of
    # them -- an interval's IoU with a copy of itself is 1, and one of any pair beats the
    # other -- so keeping only the best row per distinct interval leaves the surviving set
    # exactly as it was while removing the rows that make the join quadratic.
    #
    # "Best" is the order the pairwise rule itself uses: highest score, then lowest row
    # index. So the row kept here is the row that would have survived, and it still beats
    # everything the removed copies would have beaten -- it carries their score or better,
    # and at equal score a lower index.
    kept = (
        c.sort(["score", "_rid"], descending=[True, False])
        .unique(subset=keys + all_cols, keep="first", maintain_order=True)
    )
    left = kept.select("_rid", *keys, *all_cols, "score")

    def iou(lo: str, hi: str) -> pl.Expr:
        inter = (pl.min_horizontal(hi, f"{hi}_b")
                 - pl.max_horizontal(lo, f"{lo}_b")).clip(lower_bound=0)
        union = (pl.max_horizontal(hi, f"{hi}_b")
                 - pl.min_horizontal(lo, f"{lo}_b"))
        return pl.when(union > 0).then(inter / union).otherwise(0.0)

    overlaps = pl.all_horizontal([iou(lo, hi) >= iou_min for lo, hi in intervals])
    outranked = ((pl.col("score_b") > pl.col("score"))
                 | ((pl.col("score_b") == pl.col("score"))
                    & (pl.col("_rid_b") < pl.col("_rid"))))

    # Batch assignment by a running total of each group's pair count, so a batch's join
    # produces about pair_budget rows whatever the shape of the input. A group's rows stay
    # together because the rule compares a row only against rows sharing its key.
    sizes = left.group_by(keys).agg(pl.len().alias("_n"))
    sizes = sizes.with_columns(
        ((pl.col("_n").cast(pl.Int64) ** 2).cum_sum() // max(1, pair_budget)).alias("_batch")
    )
    left = left.join(sizes.select(*keys, "_batch"), on=keys, how="left")

    beaten = []
    for batch in left.partition_by("_batch", include_key=False):
        right = batch.select(
            pl.col("_rid").alias("_rid_b"), *keys,
            *[pl.col(col).alias(f"{col}_b") for col in all_cols],
            pl.col("score").alias("score_b"),
        )
        beaten.append(
            batch.join(right, on=keys, how="inner")
            .filter(pl.col("_rid") != pl.col("_rid_b"))
            .filter(overlaps & outranked)
            .select("_rid")
            .unique()
        )

    beaten_ids = (pl.concat(beaten).unique() if beaten
                  else pl.DataFrame(schema={"_rid": kept.schema["_rid"]}))
    # Sorted back into input order: the anti-join gives no ordering guarantee, and the
    # parquet written downstream is easier to diff against an older run when it does.
    return kept.join(beaten_ids, on="_rid", how="anti").sort("_rid").drop("_rid")


def dedup_fragment_regions(regions: pl.DataFrame, iou_min: float,
                           pair_budget: int = DEDUP_PAIR_BUDGET) -> pl.DataFrame:
    """Collapse regions duplicated by AlphaFold's overlapping structure fragments.

    A protein over 2700 aa is modelled only as 1400-residue fragments on a 200-residue
    stride, so most of each fragment also sits in its neighbour. The same alignment is
    therefore found once per fragment, and after the fragment offset is applied those hits
    land on the same target accession at the same coordinates. One real alignment, several
    rows.

    That is an artifact of how the structures are prepared, NOT tool behaviour, which is why
    this is deliberately narrow. It keys on (query_acc, target_acc) and requires BOTH the
    query and target intervals to overlap, so a tool reporting several genuinely different
    alignments between the same pair keeps all of them, and a repeat protein's tandem
    domains -- adjacent but not coincident -- are untouched.

    It is emphatically NOT a general "drop near-duplicate calls" pass. assign_instances
    penalises a tool that emits redundant copies of one call, on purpose: one becomes a true
    positive and the rest false positives, so "found all twelve fingers" and "emitted twelve
    copies of one" score differently. Removing duplicates for every tool would delete that
    distinction. Only the arms reading fragmented AlphaFold files pass --dedup-fragments.

    Ties are broken by score, then by original row order, so the result does not depend on
    join ordering.
    """
    return _suppress_overlapping(
        regions, keys=["query_acc", "target_acc"],
        intervals=[("qstart", "qend"), ("tstart", "tend")],
        iou_min=iou_min, pair_budget=pair_budget,
    )


def overlap_expr(a_start: str, a_end: str, b_start: str, b_end: str) -> pl.Expr:
    lo = pl.max_horizontal(pl.col(a_start), pl.col(b_start))
    hi = pl.min_horizontal(pl.col(a_end), pl.col(b_end))
    return (hi - lo).clip(lower_bound=0)


def transfer_domains(regions: pl.LazyFrame, domain_map: pl.LazyFrame, min_overlap: float) -> pl.LazyFrame:
    """Label each region with every target-side Pfam domain it covers."""
    joined = regions.join(
        domain_map.select(
            pl.col("accession").alias("target_acc"),
            "pfam_id",
            pl.col("domain_start").alias("t_dom_start"),
            pl.col("domain_end").alias("t_dom_end"),
        ),
        on="target_acc",
        how="inner",
    )
    return (
        joined.with_columns(
            overlap_expr("tstart", "tend", "t_dom_start", "t_dom_end").alias("t_overlap"),
            (pl.col("t_dom_end") - pl.col("t_dom_start")).alias("t_dom_len"),
        )
        .filter(pl.col("t_overlap") >= min_overlap * pl.col("t_dom_len"))
        .select("query_acc", "pfam_id", "qstart", "qend", "score")
    )


def dedup_transferred_calls(calls: pl.DataFrame, iou_min: float,
                            pair_budget: int = DEDUP_PAIR_BUDGET) -> pl.DataFrame:
    """Collapse calls that are the same prediction arrived at via different targets.

    Homology transfer turns one query region into one call per target domain it covers. A
    human kinase hitting 300 mouse kinases yields 300 calls of PF00069 over nearly the same
    residues. assign_instances() then makes one a true positive and the other 299 false
    positives, so per-protein precision collapses to 1/300 for exactly the proteins whose
    families are best conserved. That is a property of the target proteome's redundancy, not
    of whether the tool found the domain, and it is why precision falls monotonically with
    target proteome size (ecoli 0.138 -> mouse 0.018 for phmmer).

    This pass keys on (query_acc, pfam_id) and requires the QUERY intervals to overlap at
    iou_min, so a protein's tandem domains -- same family, adjacent but not coincident, IoU
    near zero -- are each kept. It says "these calls claim the same region of the same
    protein for the same family", which is one annotation.

    Deliberately NOT the default. Penalising redundant output is a real position: a tool that
    reports one clean call and a tool that reports 300 copies are different tools to a user
    reading the output. Both numbers are worth reporting, which is why this is a flag and
    every metrics row is stamped with dedup_transfers. Compare the pair to separate detection
    from redundancy. See dedup_fragment_regions() for the narrower artifact-removal pass.

    Suppression is pairwise, not iterative NMS: a call is dropped if any overlapping call
    scores higher, even if that call is itself dropped. Ties break by score then original row
    order, so the result does not depend on join ordering.

    Takes and returns an eager frame. The pass is a self-join, so the table is materialised
    whatever the caller passes; taking it eager keeps the caller from collecting the same
    source three times over (count, dedup, count) when each collect re-reads the arm's file.
    """
    return _suppress_overlapping(
        calls, keys=["query_acc", "pfam_id"], intervals=[("qstart", "qend")],
        iou_min=iou_min, pair_budget=pair_budget,
    )


def score_calls(calls: pl.LazyFrame, truth: pl.LazyFrame, min_overlap: float,
                semantics: str = "alignment",
                point_semantics: str = "cover") -> pl.DataFrame:
    """Match each call to the best true instance of the same family on the same protein.

    `semantics` picks what counts as correctly placed:

      alignment  IoU >= min_overlap. The call must coincide with the true domain. This is
                 the right test for a tool that reports an alignment, where a predicted
                 interval claims every residue inside it.
      motif      coverage of the true domain >= min_overlap. The right test for Folddisco,
                 whose interval is the envelope of a discontinuous residue set rather than
                 a claim on the residues between them. Judging that envelope by IoU would
                 score the envelope reduction, not the prediction.

    IoU is recorded either way, so the two are always inspectable side by side -- but
    is_tp, and therefore precision and recall, follow the tool's own semantics.

    `point_semantics` does the same thing on the TRUTH side, per instance. A point feature
    asserts a RESIDUE, not an interval: a Swiss-Prot ACT_SITE is one position that
    build_swissprot_truth widens by one, and an M-CSA catalytic residue is widened by
    --window. Scoring those by IoU is a category error, and not a conservative one -- it is
    arithmetically unsatisfiable. IoU against a 1-residue interval is 1/call_length, so at
    min_overlap 0.5 a true positive would need a call of at most 2 residues. Measured on the
    mini run: 97_706 calls across 32 arms, shortest 3 residues, none <= 2, so the best IoU
    any tool could reach on a point feature was 0.333 against a 0.5 cutoff. Every point
    stratum therefore scored exactly 0 on every metric -- n_instances_found 0, best_f1 0,
    and fmax 0 as well, since protein_centric_curve gates on is_tp too.

    That is not a hard benchmark, it is an unanswerable one, and it manufactures exactly
    the short-feature deficit the reduced-alphabet question is trying to test. "cover"
    scores a point instance by containment instead -- did the call cover the annotated
    residue -- which is the only form of the question that has an answer. Pass "iou" to
    restore the old behaviour and reproduce pre-2026-08-27 numbers.

    The cost is stated rather than hidden: containment favours long calls, since a
    400-residue region covering a catalytic residue counts the same as a tight one. Two
    things bound it. assign_instances is one-to-one, so one call claims at most one
    instance; and precision still counts every call, so a tool that carpets the protein
    pays for it. The boundary metrics exclude point features entirely
    (cafa_metrics.boundary_metrics), so none of this reaches DBD or NDO.
    """
    # Pfam and Pfam-N carry no is_point column; their instances are all intervals, so the
    # literal False keeps one code path instead of two.
    has_point = "is_point" in truth.collect_schema().names()
    point_col = (pl.col("is_point") if has_point
                 else pl.lit(False)).fill_null(False).alias("truth_is_point")
    matched = (
        calls.join(
            truth.select(
                pl.col("accession").alias("query_acc"),
                "pfam_id",
                pl.col("domain_start").alias("true_start"),
                pl.col("domain_end").alias("true_end"),
                point_col,
            ),
            on=["query_acc", "pfam_id"],
            how="left",
        )
        .with_columns(overlap_expr("qstart", "qend", "true_start", "true_end").alias("ov"))
        .with_columns(
            # The null guard is load-bearing. A left join leaves true_start/true_end null
            # when the query protein has no instance of the transferred family -- the
            # definitive false positive. polars' max_horizontal/min_horizontal SKIP nulls
            # rather than propagating them, so without this branch the union collapses to
            # the call's own span, IoU comes out 1.0, and every call for a family the
            # protein does not have scores as a true positive. That inverts the metric.
            pl.when(pl.col("true_start").is_null() | pl.col("true_end").is_null())
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("ov")
                / (
                    pl.max_horizontal("qend", "true_end")
                    - pl.min_horizontal("qstart", "true_start")
                )
            )
            .fill_null(0.0)
            .alias("iou")
        )
        .with_columns(
            # Fraction of the true domain the call covers. Same null guard as above:
            # max/min_horizontal skip nulls, so an absent truth row must be branched on
            # explicitly rather than divided through.
            pl.when(pl.col("true_start").is_null() | pl.col("true_end").is_null())
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("ov") / (pl.col("true_end") - pl.col("true_start"))
            )
            .fill_null(0.0)
            .alias("cover")
        )
    )

    # One row per call, carrying its best candidate. Assignment happens after this.
    per_call = (
        matched.group_by(["query_acc", "pfam_id", "qstart", "qend"])
        .agg(
            pl.col("score").max(),
            pl.col("iou").max(),
            pl.col("cover").max(),
            pl.col("true_start").sort_by("iou", descending=True).first(),
            pl.col("true_end").sort_by("iou", descending=True).first(),
            pl.col("truth_is_point").sort_by("iou", descending=True).first(),
        )
    )
    candidates = matched.select(
        "query_acc", "pfam_id", "qstart", "qend", "score", "iou", "cover",
        "true_start", "true_end", "truth_is_point",
    ).filter(pl.col("true_start").is_not_null())

    calls_df = per_call.collect(engine="streaming")
    cand_df = candidates.collect(engine="streaming")
    return assign_instances(calls_df, cand_df, min_overlap, semantics, point_semantics)


def assign_instances(calls: pl.DataFrame, candidates: pl.DataFrame,
                     min_overlap: float, semantics: str,
                     point_semantics: str = "cover") -> pl.DataFrame:
    """One-to-one matching between predicted regions and annotated instances.

    Without this a protein carrying a tandem array is scored incoherently. Twelve
    predictions landing on the SAME zinc finger would each count as a true positive, and a
    single prediction swallowing all twelve fingers would also count as one -- so "merged
    everything into one region" and "found all twelve correctly" become indistinguishable,
    and "emitted twelve redundant copies" scores as a perfect result. Those are three
    different behaviours and the metric has to separate them.

    Greedy in score order, the COCO convention: walk predictions from best-scoring down,
    match each to the best still-unclaimed annotation it overlaps enough, and mark both
    used. Score order rather than IoU order matters for the PR curve -- a prediction's
    TP/FP status must not depend on predictions ranked below it, or the curve is not
    monotone in the threshold.

    Unmatched predictions are false positives; unmatched annotations are not
    recovered, and show up through the recall denominator rather than as rows here.
    """
    key = "cover" if semantics == "motif" else "iou"
    if candidates.height == 0:
        return calls.with_columns(pl.lit(False).alias("is_tp"))

    # Per ROW, not per tool: the criterion follows the tool's semantics for an interval
    # annotation and the truth's for a point one. See score_calls for why IoU against a
    # 1-residue interval is unsatisfiable rather than merely strict.
    use_point_cover = point_semantics == "cover" and "truth_is_point" in candidates.columns
    is_point = (pl.col("truth_is_point").fill_null(False) if use_point_cover
                else pl.lit(False))
    elig_key = (
        pl.when(is_point).then(pl.col("cover")).otherwise(pl.col(key)).alias("elig")
    )

    # Range instances are offered every call BEFORE point instances are offered any.
    # Containment maxes out at 1.0 for a point instance, so on a plain (score, elig) sort a
    # point feature outbid every interval and a call that correctly delineated a DOMAIN was
    # consumed by an incidental ACT_SITE inside it. Ordering is_point last leaves the
    # (call, range-instance) assignment exactly as it was -- the range candidates keep
    # their relative order and every call that could claim one still does -- so point
    # instances take only the calls nothing else claimed.
    #
    # Measured on the MHC set, cover vs iou, over the 24 range-only cells: 0 changed on
    # n_instances_found, recall_reachable, n_tp_calls or fmax. What does move is precision
    # and coverage, on 10 and 20 cells, and that is the intended consequence rather than
    # leakage: a call whose true match is a point feature OUTSIDE a range cut is no longer
    # charged to that cut as a false positive, it becomes gray there. The shift is ~0.0004
    # in precision and it is in the direction of not blaming a tool for being right about
    # something the cut does not measure.
    # The trailing keys are a DETERMINISM fix, not cosmetics. This is a greedy one-to-one
    # walk, so which call claims an instance depends on row order -- and (score, is_point,
    # elig) is not a unique key. Ties are not rare: rank_roc_auc's own docstring notes that
    # HP alphabets at low ksize produce large blocks of identical region scores. polars does
    # not promise a stable sort, so tied rows came back in different orders between runs and
    # the same inputs scored differently: five identical runs of one arm gave
    # n_instances_found 169, 169, 169, 169, 168.
    #
    # Adding the call's identity and its matched instance makes the key unique, so the walk
    # is reproducible. Same class of bug as the one tests/test_dedup_passes.py guards for
    # the pairwise suppression passes. Those trailing keys are cm.CALL_TIEBREAK, shared
    # with sensitivity_to_first_fp rather than repeated, so the two stages that rank calls
    # cannot come to disagree about what "first" means.
    order = ["score", "_is_point", "elig", *cm.CALL_TIEBREAK]
    elig = (
        candidates.with_columns(elig_key, is_point.alias("_is_point"))
        .filter(pl.col("elig") >= min_overlap)
        .sort(order,
              descending=[True, False, True] + [False] * len(cm.CALL_TIEBREAK),
              nulls_last=True)
    )

    used_truth: set[tuple] = set()
    used_call: set[tuple] = set()
    matched_calls: dict[tuple, tuple] = {}

    for qa, pf, qs, qe, _sc, _iou, _cov, ts, te in elig.select(
        "query_acc", "pfam_id", "qstart", "qend", "score", "iou", "cover",
        "true_start", "true_end",
    ).iter_rows():
        ck = (qa, pf, qs, qe)
        tk = (qa, pf, ts, te)
        if ck in used_call or tk in used_truth:
            continue
        used_call.add(ck)
        used_truth.add(tk)
        matched_calls[ck] = (ts, te)

    if not matched_calls:
        return calls.with_columns(pl.lit(False).alias("is_tp"))

    assigned = pl.DataFrame(
        [(k[0], k[1], k[2], k[3], v[0], v[1]) for k, v in matched_calls.items()],
        schema=["query_acc", "pfam_id", "qstart", "qend", "a_start", "a_end"],
        orient="row",
    )
    out = calls.join(assigned, on=["query_acc", "pfam_id", "qstart", "qend"], how="left")
    return out.with_columns(
        pl.col("a_start").is_not_null().alias("is_tp"),
        # Report the ASSIGNED instance, not the nearest one. A call that overlapped a
        # domain another prediction already claimed is a false positive, and leaving its
        # near-miss coordinates in place would make the calls table read as if it matched.
        pl.when(pl.col("a_start").is_not_null()).then(pl.col("a_start"))
          .otherwise(None).alias("true_start"),
        pl.when(pl.col("a_end").is_not_null()).then(pl.col("a_end"))
          .otherwise(None).alias("true_end"),
    ).drop("a_start", "a_end")


def rank_roc_auc(calls: pl.DataFrame) -> float | None:
    """Call-level ROC-AUC: P(a correctly placed call outranks an incorrectly placed one).

    Computed by the Mann-Whitney rank identity rather than by integrating the curve, so
    tied scores are handled without approximation. Ties matter here: HP alphabets at low ksize produce
    large blocks of identical region scores, and trapezoid integration over a coarse
    curve would quietly round them in the tool's favour.

    Returns None, not 0.0, when one class is absent. A tool whose every call is correct
    has no ROC-AUC to report, and writing 0.0 there would rank it below a coin flip.
    """
    n = calls.height
    if n == 0:
        return None
    n_pos = int(calls["is_tp"].sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ranked = calls.with_columns(
        pl.col("score").fill_null(float("-inf")).rank(method="average").alias("rank")
    )
    sum_pos_ranks = float(ranked.filter("is_tp")["rank"].sum())
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def operating_points(calls: pl.DataFrame, n_reachable: int) -> pl.DataFrame:
    """Every score threshold's full operating point, in one descending-score pass.

    Each row is "keep all calls scoring at least this much". Cumulative counts are taken
    at the last row of each distinct score so a threshold never splits a block of ties.

    Two different denominators sit side by side on purpose:
      precision, tpr, fpr    call-level -- of what was reported, how much was right
      recall_reachable       instance-level -- of the domains that could be found, how
                             many were, counting each true instance once no matter how
                             many regions hit it
    """
    if calls.height == 0 or "is_gray" not in calls.columns:
        return pl.DataFrame(
            schema={
                "score_threshold": pl.Float64, "n_calls": pl.Int64, "tp_calls": pl.Int64,
                "fp_calls": pl.Int64, "instances_found": pl.Int64, "precision": pl.Float64,
                "recall_reachable": pl.Float64, "f1": pl.Float64, "tpr": pl.Float64,
                "fpr": pl.Float64,
            }
        )

    ranked = (
        calls.sort("score", descending=True, nulls_last=True)
        .with_columns(
            pl.when("is_tp")
            .then(pl.struct("query_acc", "pfam_id", "true_start", "true_end"))
            .otherwise(None)
            .alias("tp_key")
        )
        .with_columns(
            (pl.col("tp_key").is_not_null() & pl.col("tp_key").is_first_distinct())
            .alias("novel_tp")
        )
        .with_columns(
            pl.col("is_tp").cum_sum().alias("tp_calls"),
            # Gray calls are excluded here too, so the curve and the scalar precision
            # describe the same thing rather than diverging at every threshold.
            (~pl.col("is_tp") & ~pl.col("is_gray")).cum_sum().alias("fp_calls"),
            pl.col("novel_tp").cum_sum().alias("instances_found"),
        )
    )

    total_tp = int(ranked["is_tp"].sum())
    total_fp = ranked.height - total_tp

    pts = (
        ranked.group_by("score", maintain_order=True)
        .agg(
            pl.col("tp_calls").last(),
            pl.col("fp_calls").last(),
            pl.col("instances_found").last(),
        )
        .rename({"score": "score_threshold"})
        .with_columns((pl.col("tp_calls") + pl.col("fp_calls")).alias("n_calls"))
        # A threshold that retains zero SCOREABLE calls is not an operating point. It
        # happens when the top-scoring block is entirely gray: tp_calls and fp_calls are
        # both 0, so precision came out 0/0 = NaN -- and polars sorts NaN as the largest
        # float, so `sort("f1", descending=True).head(1)` handed that row back as best_f1.
        # The row carries no information either way (no TP means recall is 0 there), so it
        # is dropped rather than papered over with a convention.
        .filter(pl.col("n_calls") > 0)
    )

    return pts.with_columns(
        (pl.col("tp_calls") / pl.col("n_calls")).alias("precision"),
        (pl.col("instances_found") / n_reachable if n_reachable else pl.lit(0.0)).alias(
            "recall_reachable"
        ),
        (pl.col("tp_calls") / total_tp if total_tp else pl.lit(0.0)).alias("tpr"),
        (pl.col("fp_calls") / total_fp if total_fp else pl.lit(0.0)).alias("fpr"),
    ).with_columns(
        pl.when(pl.col("precision") + pl.col("recall_reachable") > 0)
        .then(
            2
            * pl.col("precision")
            * pl.col("recall_reachable")
            / (pl.col("precision") + pl.col("recall_reachable"))
        )
        .otherwise(0.0)
        .alias("f1")
    )


def average_precision(points: pl.DataFrame) -> float:
    """Average precision: sum of precision weighted by the recall gained at each step.

    The step-wise sum, not a trapezoid, which is the standard AP definition and does not
    interpolate credit across a gap the tool never covered.
    """
    if points.height == 0:
        return 0.0
    recall = points["recall_reachable"]
    delta = recall - recall.shift(1, fill_value=0.0)
    return float((points["precision"] * delta).sum())


def downsample(points: pl.DataFrame, max_points: int) -> pl.DataFrame:
    """Thin the curve for storage, always keeping both ends.

    A 1017-combo sweep writing one row per distinct score would dwarf the metrics it
    supports. Every scalar metric is computed on the FULL curve before this runs, so
    thinning changes the plot's resolution and nothing else.
    """
    n = points.height
    if n <= max_points:
        return points
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    return points[sorted(set(idx))]


def classify_scoreable(calls: pl.DataFrame, truth: pl.DataFrame,
                       min_annotated_fraction: float) -> pl.DataFrame:
    """Split non-TP calls into confident false positives and unscoreable gray-zone calls.

    The Foldseek/Folddisco SCOPe convention, adapted to regions: confident positive is a
    TP, confidently-different is an FP, and UNKNOWN is excluded from the denominator with
    coverage reported alongside.

    Why this benchmark needs it. Pfam-A annotates a fraction of residues; everywhere else
    it is silent, not negative. Counting a call in silent territory as a false positive
    asserts that Pfam looked there and found nothing, which it did not. That is
    backwards for the claim under test -- a cryptic domain Pfam never annotated is the
    thing the method is supposed to find, and scoring it as an error makes the benchmark
    punish the hypothesis rather than test it.

    The split, for a call that is not a TP:
      confident FP   its residues lie mostly inside annotated territory on that protein.
                     The annotation looked there and named a different family.
      gray           its residues lie mostly outside any annotation. Unknown, excluded.

    `min_annotated_fraction` is the share of the CALL that must sit inside annotated
    territory to be judged confidently wrong. Measured against the call rather than the
    annotation so a long call is not excused by clipping one domain's edge.
    """
    if calls.height == 0:
        return calls.with_columns(pl.lit(False).alias("is_gray"))

    ann = truth.select(
        pl.col("accession").alias("query_acc"),
        pl.col("domain_start").alias("a_start"),
        pl.col("domain_end").alias("a_end"),
    ).unique()

    # Residues of each call that fall inside ANY annotation on that protein, family
    # ignored -- the question here is only whether the annotation had an opinion.
    ov = (
        calls.filter(~pl.col("is_tp"))
        .select("query_acc", "pfam_id", "qstart", "qend")
        .join(ann, on="query_acc", how="left")
        .with_columns(
            (
                pl.min_horizontal("qend", "a_end") - pl.max_horizontal("qstart", "a_start")
            ).clip(lower_bound=0).alias("ov")
        )
        .group_by("query_acc", "pfam_id", "qstart", "qend")
        # Summed, so a call spanning two adjacent domains counts both. Overlapping
        # annotations could double-count, which can only make a call look MORE covered and
        # therefore more likely to be judged a confident FP -- the conservative direction.
        .agg(pl.col("ov").sum().alias("annotated_residues"))
        .with_columns(
            (
                pl.col("annotated_residues")
                / (pl.col("qend") - pl.col("qstart")).clip(lower_bound=1)
            ).alias("annotated_fraction")
        )
    )

    out = calls.join(ov, on=["query_acc", "pfam_id", "qstart", "qend"], how="left")
    return out.with_columns(
        (
            ~pl.col("is_tp")
            & (pl.col("annotated_fraction").fill_null(0.0) < min_annotated_fraction)
        ).alias("is_gray")
    )


def compute_metrics(calls: pl.DataFrame, points: pl.DataFrame, truth: pl.DataFrame,
                    reachable: pl.DataFrame, min_overlap: float) -> dict:
    n_calls = calls.height
    n_tp_calls = int(calls["is_tp"].sum()) if n_calls else 0

    # Counts distinct true instances found, not calls: several regions hitting one domain
    # is one recovery, not many.
    #
    # No intersection against the truth subset here any more. It used to live in this
    # function alone, which left operating_points -- and therefore the curve, auprc and
    # best_f1 -- reading the unrestricted count. restrict_tp_to_cut now clears is_tp on any
    # call whose instance is outside the cut before either consumer sees the table, so both
    # read the same numerator. See its docstring for why recall could otherwise exceed 1.0.
    found = (
        calls.filter("is_tp").select(
            "query_acc", "pfam_id", "true_start", "true_end"
        ).unique().height
        if n_calls else 0
    )
    n_truth = truth.height
    n_reachable = reachable.height

    # Gray-zone accounting. Calls in territory the annotation never covered are excluded
    # from the precision denominator rather than counted against the tool.
    n_gray = int(calls["is_gray"].sum()) if ("is_gray" in calls.columns and n_calls) else 0
    n_scoreable = n_calls - n_gray

    precision = n_tp_calls / n_scoreable if n_scoreable else 0.0
    # The same number under the old convention, kept visible so the gray-zone choice can
    # never be mistaken for a free improvement -- the gap between these two IS the effect.
    precision_strict = n_tp_calls / n_calls if n_calls else 0.0
    recall = found / n_truth if n_truth else 0.0
    recall_reachable = found / n_reachable if n_reachable else 0.0

    def f1(p, r):
        return 2 * p * r / (p + r) if (p + r) else 0.0

    metrics = {
        "n_calls": n_calls,
        "n_tp_calls": n_tp_calls,
        "n_fp_calls": n_scoreable - n_tp_calls,
        "n_gray_calls": n_gray,
        # Fraction of calls that could be judged at all. A great precision on 12% of calls
        # is a different claim from the same precision on 90%, so this travels with it.
        "coverage": n_scoreable / n_calls if n_calls else 0.0,
        "n_truth_instances": n_truth,
        "n_reachable_instances": n_reachable,
        "n_instances_found": found,
        # --- operating point the tool reported at ---
        "precision": precision,
        "precision_strict": precision_strict,
        "recall": recall,
        # Recall against what was transferable at all. A human family absent from this
        # target proteome cannot be recovered by any search, so raw recall above
        # understates every tool by the same species-specific amount. Compare tools on
        # this one.
        "recall_reachable": recall_reachable,
        "f1": f1(precision, recall),
        "f1_reachable": f1(precision, recall_reachable),
        # --- threshold-free ---
        "roc_auc": rank_roc_auc(calls),
        "auprc": average_precision(points),
        "min_overlap": min_overlap,
        "median_iou_tp": float(calls.filter("is_tp")["iou"].median()) if n_tp_calls else 0.0,
    }

    # --- best achievable operating point, and where it sits ---
    # The reported point above depends on each tool's own default cutoff, which differs
    # between tools and is not a property of the method. This is the comparable one.
    if points.height:
        # Threshold breaks the tie, lowest first. Score plateaus put several thresholds at
        # the same best F1, and the one reported has to be the same one every run.
        best = points.sort(
            ["f1", "score_threshold"], descending=[True, False]
        ).head(1).to_dicts()[0]
        metrics.update({
            "best_f1": best["f1"],
            "best_f1_threshold": best["score_threshold"],
            "best_f1_precision": best["precision"],
            "best_f1_recall_reachable": best["recall_reachable"],
        })
    else:
        metrics.update({
            "best_f1": 0.0, "best_f1_threshold": None,
            "best_f1_precision": 0.0, "best_f1_recall_reachable": 0.0,
        })
    return metrics


def attach_identity(truth: pl.DataFrame, identity: pl.DataFrame | None) -> pl.DataFrame:
    """Bin each domain instance by identity to its closest same-family target domain.

    Instances with no same-family match in the target get a distinct `no_homolog` label
    rather than being dropped or lumped into the lowest bin. They are unreachable by any
    transfer-based method, so mixing them into "<20%" would make every tool look worse in
    the bin the hypothesis cares most about.
    """
    if identity is None or identity.height == 0:
        return truth.with_columns(pl.lit(None, dtype=pl.String).alias("stratum_identity"))

    key = ["accession", "pfam_id", "domain_start", "domain_end"]
    # best_target rides along so a covariate of the winning target can be attached to this
    # human instance downstream. It is only present in tables written after 2026-08-27.
    cols = key + ["best_pident"] + (["best_target"] if "best_target" in identity.columns else [])
    joined = truth.join(identity.select(cols), on=key, how="left")

    expr = pl.when(pl.col("best_pident").is_null()).then(pl.lit("no_homolog"))
    for lo, hi in zip(IDENTITY_BINS[:-1], IDENTITY_BINS[1:]):
        expr = expr.when(
            (pl.col("best_pident") >= lo) & (pl.col("best_pident") < hi)
        ).then(pl.lit(f"{int(lo)}-{int(hi)}%"))
    return joined.with_columns(expr.otherwise(None).alias("stratum_identity"))


def attach_feature_length(truth: pl.DataFrame) -> pl.DataFrame:
    """Bin each truth interval by its own residue length, and keep the raw length.

    `feature_length` rides along beside the bin because the quantity the reduced-alphabet
    question is stated on is feature_length / ksize, not the bin label. A bin midpoint would
    be a made-up number; the median of the instances actually in a cell is measured, and
    score_one puts it on every metric row.
    """
    length = pl.col("domain_end") - pl.col("domain_start")
    edges = FEATURE_LENGTH_BINS
    expr = pl.when(length >= edges[-1]).then(pl.lit(f"{edges[-1]}+"))
    for lo, hi in zip(edges[:-1], edges[1:]):
        expr = expr.when((length >= lo) & (length < hi)).then(
            pl.lit(str(lo) if hi - lo == 1 else f"{lo}-{hi - 1}")
        )
    return truth.with_columns(
        length.alias("feature_length"),
        expr.otherwise(None).alias("stratum_feature_length_bin"),
    )


def attach_feature_type(truth: pl.DataFrame) -> pl.DataFrame:
    """One stratum per Swiss-Prot feature type; null for every other truth set.

    build_swissprot_truth puts the FT type (TRANSMEM, ACT_SITE, ...) in `pfam_id`, keeping
    the column name for schema compatibility. For the Pfam and Pfam-N truth sets that same
    column holds a Pfam accession, which carries no type variation at all -- every row is
    the same kind of object -- and for M-CSA it holds an entry id. Cutting on it there would
    reproduce the hgnc axis at ~19k strata of one.

    Detected from the values rather than from the --truth-set name, so a truth set that
    gains FT types later works without a second place to edit. Null, not "", so a
    downstream group_by drops the axis instead of inventing a stratum named after nothing.
    """
    none = pl.lit(None, dtype=pl.String).alias("stratum_feature_type")
    ids = set(truth["pfam_id"].unique().to_list())
    if not ids or not ids <= FEATURE_TYPES:
        return truth.with_columns(none)
    return truth.with_columns(pl.col("pfam_id").alias("stratum_feature_type"))


def attach_target_disorder(truth: pl.DataFrame,
                           target_disorder: pl.DataFrame | None) -> pl.DataFrame:
    """Bin each human instance by the disorder of the TARGET it could best transfer from.

    The query-side disorder axis asks whether a tool copes with a disordered human region.
    This asks the other half, and for a structure-based method it is the half that bites:
    foldseek and reseek align a structure to a structure, so a target with no confident
    structure defeats them however well-ordered the human query is. A sequence-only method
    has no such dependency on either side.

    "The target it could best transfer from" is the same instance-level definition the
    identity axis uses -- the closest same-family domain in that proteome -- so the two
    axes describe the same target and can be read together. That is why this needs
    best_target from parse_domain_identity rather than a per-proteome average, which would
    only tell you the species.

    Bins match the query-side disorder edges so the two are directly comparable.
    """
    none = pl.lit(None, dtype=pl.String)
    if (target_disorder is None or target_disorder.height == 0
            or "best_target" not in truth.columns):
        return truth.with_columns(none.alias("stratum_disorder_target"))

    col = "disorder_fraction_metapredict"
    if col not in target_disorder.columns:
        return truth.with_columns(none.alias("stratum_disorder_target"))

    joined = truth.join(
        target_disorder.select(pl.col("accession").alias("best_target"),
                               pl.col(col).alias("target_disorder")),
        on="best_target", how="left",
    )
    edges = STRATA["disorder"][1]
    expr = pl.when(pl.col("target_disorder").is_null()).then(none)
    for lo, hi in zip(edges[:-1], edges[1:]):
        expr = expr.when(
            (pl.col("target_disorder") >= lo) & (pl.col("target_disorder") < hi)
        ).then(pl.lit(f"{lo}-{hi}"))
    return joined.with_columns(expr.otherwise(None).alias("stratum_disorder_target"))


def attach_strata(truth: pl.DataFrame, covariates: pl.DataFrame | None,
                  keep_zinc_finger: bool = False) -> pl.DataFrame:
    """Add one column per covariate axis, holding that protein's stratum label."""
    if covariates is None:
        return truth.with_columns(pl.lit("all").alias("stratum_hgnc"))

    cov = covariates
    exprs = []
    for axis, (col, edges) in STRATA.items():
        name = f"stratum_{axis}"
        if col not in cov.columns:
            exprs.append(pl.lit(None, dtype=pl.String).alias(name))
            continue
        expr = pl.when(pl.col(col).is_null()).then(pl.lit(None, dtype=pl.String))
        for lo, hi in zip(edges[:-1], edges[1:]):
            expr = expr.when((pl.col(col) >= lo) & (pl.col(col) < hi)).then(
                pl.lit(f"{lo}-{hi}")
            )
        exprs.append(expr.otherwise(None).alias(name))

    hgnc = "hgnc_gene_group"
    if hgnc in cov.columns:
        # Zinc fingers are kept in the HGNC axis by default. Notebook 206 excludes C2H2
        # families because tandem arrays inflate PROTEIN-level k-mer sharing through repeat
        # content, which is a real confound when the scored object is a protein pair. Here
        # the scored object is a domain instance: a twelve-finger protein contains twelve
        # domains and the right answer is twelve correctly-bounded regions. The exclusion
        # belonged to a different unit of analysis. --exclude-zinc-finger-from-hgnc restores
        # it for anyone comparing against the orthology-era numbers.
        excluded = (
            pl.col("hgnc_group_excluded")
            if ("hgnc_group_excluded" in cov.columns and not keep_zinc_finger)
            else pl.lit(False)
        )
        exprs.append(
            pl.when(excluded).then(None).otherwise(pl.col(hgnc)).alias("stratum_hgnc")
        )
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias("stratum_hgnc"))

    # MHC class is categorical and small; every class is worth cutting on its own because
    # notebook 211 found class I and class II answer the k-size question in opposite
    # directions, so a single pooled "MHC" number hides the result.
    exprs.append(
        pl.col("mhc_class").alias("stratum_mhc") if "mhc_class" in cov.columns
        else pl.lit(None, dtype=pl.String).alias("stratum_mhc")
    )

    # Each curated set becomes one stratum value on a shared axis; non-members are null so
    # they do not form a meaningless "everything else" cell.
    present = [(name, col) for name, col in GENE_SET_FLAGS.items() if col in cov.columns]
    if present:
        expr = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
        for name, col in present:
            expr = expr.when(pl.col(col)).then(pl.lit(name))
        exprs.append(expr.otherwise(None).alias("stratum_geneset"))
    else:
        exprs.append(pl.lit(None, dtype=pl.String).alias("stratum_geneset"))

    cov = cov.with_columns(exprs)

    keep = ["accession"] + [c for c in cov.columns if c.startswith("stratum_")]
    return truth.join(cov.select(keep), on="accession", how="left")


def strata_of(truth: pl.DataFrame) -> list[tuple[str, str]]:
    """Enumerate (axis, value) cuts worth reporting, always including the ungrouped one."""
    out = [("all", "all")]
    for col in (c for c in truth.columns if c.startswith("stratum_")):
        axis = col.removeprefix("stratum_")
        floor = 1 if axis in UNFLOORED_AXES else MIN_STRATUM_PROTEINS
        counts = (
            truth.filter(pl.col(col).is_not_null())
            .group_by(col)
            .agg(pl.col("accession").n_unique().alias("n"))
            .filter(pl.col("n") >= floor)
            .sort("n", descending=True)
        )
        out.extend((axis, v) for v in counts[col].to_list())
    return out


def instance_level_axes(truth: pl.DataFrame) -> frozenset[str]:
    """Which stratum axes can put two instances of ONE protein in different cuts.

    Only those axes need restrict_tp_to_cut, and skipping the rest is not a weakening --
    it is exact. subset() cuts calls by protein. If every row of a protein carries the same
    value on an axis, then "the truth rows of these proteins" and "the truth rows of these
    proteins in this stratum" are the same set, so no true positive can be orphaned and the
    restriction is a no-op by construction.

    n_unique counts null as its own value, which is what makes the test right rather than
    nearly right: a protein carrying one DOMAIN and one unlabelled row scores 2 and is
    correctly treated as instance-level, because subset() drops the unlabelled row from the
    truth while keeping the protein's calls.

    Worth the arithmetic. The restriction is a join, subset() runs once per
    split x stratum x arm, and the hgnc axis alone is ~4200 strata -- so paying for it on
    every axis would put a third join into the innermost loop of the whole sweep to fix
    something only three axes can suffer from.
    """
    out = set()
    for col in (c for c in truth.columns if c.startswith("stratum_")):
        spread = truth.group_by("accession").agg(pl.col(col).n_unique().alias("n"))
        if spread.height and int(spread["n"].max()) > 1:
            out.add(col.removeprefix("stratum_"))
    return frozenset(out)


def subset(truth: pl.DataFrame, calls: pl.DataFrame, split: str, axis: str, value: str,
           instance_axes: frozenset[str] = frozenset(),
           ) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Restrict both the answer key and the calls to one split x stratum cell.

    Truth and calls must be cut the same way or the metrics are incoherent: keeping a
    call whose protein is outside the stratum would count against a denominator that
    never included it.
    """
    t = truth
    if split != "all":
        t = t.filter(pl.col("split") == split)
    if axis != "all":
        t = t.filter(pl.col(f"stratum_{axis}") == value)

    if t.height == 0:
        return t, calls.head(0)

    proteins = t.select("accession").unique().rename({"accession": "query_acc"})
    c = calls.join(proteins, on="query_acc", how="inner")
    if split != "all":
        # Splits are grouped by Pfam family, so a call is in the split iff its claimed
        # family is. Filtering calls by protein alone would leak selection-half families
        # into the held-out numbers.
        fams = t.select("pfam_id").unique()
        c = c.join(fams, on="pfam_id", how="inner")
    # Splits cannot orphan a true positive on their own: they cut truth AND calls by
    # family, so a call matching an instance of an in-split family on an in-cut protein
    # matches a row that survived both filters.
    return t, (restrict_tp_to_cut(t, c) if axis in instance_axes else c)


def restrict_tp_to_cut(truth: pl.DataFrame, calls: pl.DataFrame) -> pl.DataFrame:
    """Re-judge each call against the truth instances that are actually inside this cut.

    Strata are applied to calls by PROTEIN, but three axes are properties of the individual
    annotation -- identity bin, feature length bin, feature type -- and one protein can
    carry instances in several of them at once. A 90%-identity domain and a 25%-identity one
    sit in the same protein; so do a 21-residue TRANSMEM and a 400-residue DOMAIN. A call
    that correctly hit an instance OUTSIDE the cut was still counted as a true positive
    inside it, against a denominator that never contained that instance.

    compute_metrics used to intersect its own numerator for exactly this reason -- recall
    above 1.0, observed at 2.77 -- but operating_points did not, so the curve, auprc and
    best_f1 kept the inflated count. best_f1 is the headline number for the feature-length
    axis, so the restriction has to happen once, here, where every consumer reads it.

    An orphaned TP becomes GRAY -- unscoreable in this cut -- rather than a false positive.
    It is not a wrong answer; it is a right answer about something this cut does not measure,
    and charging it to precision would be the same error pointing the other way. `coverage`
    reports the share, exactly as it does for every other gray call.

    Called only for the axes instance_level_axes() identifies. On a protein-level axis it
    would be a no-op -- every instance of an in-cut protein is in the cut, so every TP key
    survives the join -- and proving that lets the innermost loop skip a join it does not
    need on the ~4200-stratum hgnc axis.
    """
    if calls.height == 0 or "is_tp" not in calls.columns:
        return calls
    key = ["query_acc", "pfam_id", "true_start", "true_end"]
    in_cut = (
        truth.select(
            pl.col("accession").alias("query_acc"), "pfam_id",
            pl.col("domain_start").alias("true_start"),
            pl.col("domain_end").alias("true_end"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("in_cut"))
    )
    # A non-TP call carries null true_start/true_end. polars does not match null keys in a
    # join, so those rows come back with in_cut null -- which is why `orphan` is gated on
    # is_tp rather than on in_cut alone.
    out = calls.join(in_cut, on=key, how="left")
    orphan = pl.col("is_tp") & pl.col("in_cut").is_null()
    exprs = [(pl.col("is_tp") & ~orphan).alias("is_tp")]
    if "is_gray" in calls.columns:
        exprs.append((pl.col("is_gray") | orphan).alias("is_gray"))
    return out.with_columns(exprs).drop("in_cut")


def target_map_coverage(map_lf: pl.LazyFrame, target_families: pl.DataFrame,
                        map_path, job: dict) -> dict:
    """How much target-side annotation this arm actually had to transfer from.

    Every sequence-search arm scores by transferring a family label off the target
    interval it aligned to, so the size of the transfer table is a hard ceiling on what
    any of them can produce -- and it is a property of the TARGET ANNOTATION, not of the
    tool or of evolutionary distance. Nothing in the metrics row recorded it, so a target
    species with almost no annotation looked exactly like a species no tool could reach.

    That is not hypothetical. On the Swiss-Prot truth set, Ciona intestinalis has 28
    reviewed entries in UniProtKB/Swiss-Prot against 2_309 - 20_417 for every other target
    species in the benchmark, so its transfer table is ~1/100th the size and every arm's
    call count collapsed by 30-130x at 550 Mya. It read as an evolutionary cliff.

    The existing reachability bar cannot catch it: on that truth set `pfam_id` holds a
    Swiss-Prot FEATURE TYPE from a 15-value vocabulary (DOMAIN, TRANSMEM, ACT_SITE, ...),
    so 28 proteins still cover almost every type and reachability reads 6_991 / 7_000.
    These three counts are vocabulary-independent and do catch it.
    """
    stats = map_lf.select(
        pl.len().alias("n"),
        pl.col("accession").n_unique().alias("n_prot"),
    ).collect().to_dicts()[0]
    if stats["n"] == 0:
        raise SystemExit(
            f"empty domain map: {map_path} has no rows, so {job['tool']}/"
            f"{job['variant']} can transfer nothing and would publish a valid-looking "
            "all-zero result. Rebuild the target annotation before scoring."
        )
    return {
        "n_target_map_instances": int(stats["n"]),
        "n_target_map_proteins": int(stats["n_prot"]),
        "n_target_families": int(target_families.height),
    }


def read_shared_inputs(args):
    """Read the two frames every arm shares, ONCE, and hand back in-memory lazy views.

    Both used to stay as `pl.scan_parquet` handles held across the whole task. A scan is
    re-executed on every collect, so with A arms and M dedup modes a batched task re-read
    the answer key and the target domain map A x M times over -- 24 arms x 2 modes = 48
    passes over the same two files, from inside one task that had already staged them.

    Measured on the 2026-08-31 midi trace, scoreDomainCalls was the largest disk reader in
    the pipeline: 906.1 GB of read_bytes across 756 tasks, more than folddiscoQuery's
    409.5 GB and more than every search process put together. Two facts identify the
    cause. Read volume fits `0.073 GB x arms + 0.114 GB` (r = 0.50), so 820 of the 906 GB
    scale with the ARM COUNT rather than with the number of tasks. And rchar over the same
    756 tasks is only 19.5 GB -- the process asked the kernel for 19.5 GB and the block
    device delivered 906. A 46x gap between the two is not ordinary reading: polars mmaps a
    parquet scan, and re-collecting a mapped file whose pages the task's own working set
    keeps evicting re-faults those pages in from disk, where they count in read_bytes and
    never in rchar.

    Collecting once turns 48 mapped passes into one read, so the amplification cannot
    happen: what is in memory cannot be evicted and re-faulted from a file.

    Why this rather than the alternatives that were weighed:

      node-local scratch  moves the same 906 GB from the shared filesystem to local NVMe.
                          Cheaper per byte, but it treats the symptom and still pays a copy
                          in on every task.
      collect()/broadcast the files are already staged once per task by Nextflow, by
                          symlink. Sharing them harder cannot help, because the re-reads
                          happen INSIDE a task that has one copy already.
      coarser batching    strictly worse. Reads scale with arms per task, so putting more
                          arms in a task multiplies the same per-arm re-read.

    The cost is holding both frames resident. That is what scoreMemory in main.nf already
    sizes for, and the observed peak over the 756 tasks was 30.6 GB against requests of
    8-99 GB.
    """
    truth_lf = pl.scan_parquet(args.truth).collect().lazy()
    domain_map_lf = (pl.scan_parquet(args.domain_map).collect().lazy()
                     if args.domain_map else None)
    return truth_lf, domain_map_lf


def score_one(args, truth, truth_lf, job, instance_axes=frozenset(),
              domain_map_lf=None):
    """Score one tool's regions against the already-loaded truth for one species.

    truth and truth_lf are passed in rather than read here because they are shared across
    every tool for a given (truth_set, species): a batched task scores ~376 of them, and
    re-reading the answer key that many times was most of the wall clock.

    domain_map_lf is passed in for the same reason and is measured, not assumed. See
    read_shared_inputs: scanning it here made the domain map the single largest disk
    reader in the whole pipeline.
    """
    # Folddisco reports the envelope of a discontinuous residue set, not an alignment.
    # Scoring that by interval IoU would measure the envelope reduction rather than the
    # prediction, so this arm is scored on coverage instead.
    semantics = "motif" if job["tool"] == "folddisco" else "alignment"
    # Only the arms that read AlphaFold structure files. Those are the ones whose targets
    # arrive as overlapping 1400-residue fragments, so the same alignment appears once per
    # fragment. Every other arm reads whole sequences and has nothing to collapse -- and
    # deduping them would erase the redundancy penalty assign_instances applies on purpose.
    dedup_fragments = job["tool"] in ("foldseek", "reseek", "folddisco")

    regions = load_regions(
        job["regions"], args.direct_annotation,
        rank_by=args.kmerseek_rank_by,
        max_bonferroni_p=(args.kmerseek_max_bonferroni_p
                          if args.kmerseek_max_bonferroni_p > 0 else None),
    )

    # Before transfer, not after: a fragment duplicate is one alignment reported twice, so
    # it has to go while the target coordinates that identify it as a duplicate are still
    # in the table. transfer_domains drops them.
    if regions is not None and dedup_fragments and not args.direct_annotation:
        # Collected once, for the reason _suppress_overlapping documents: deduping a lazy
        # frame made the survivors depend on which execution of the plan produced them.
        regions_df = regions.collect()
        before = regions_df.height
        regions_df = dedup_fragment_regions(regions_df, args.fragment_iou,
                                            pair_budget=args.dedup_pair_budget)
        after = regions_df.height
        regions = regions_df.lazy()
        if before != after:
            print(f"fragment dedup: {before - after} of {before} regions were the same "
                  f"alignment seen in overlapping AlphaFold fragments "
                  f"({100 * (before - after) / before:.1f}%)", file=sys.stderr)

    if args.direct_annotation:
        target_families = None
        target_coverage = {"n_target_map_instances": None,
                           "n_target_map_proteins": None,
                           "n_target_families": None}
        calls_lf = regions
    else:
        # Never pl.scan_parquet here. Every scan inside this function is paid once per
        # (arm x dedup mode), and a batched task holds up to 24 arms.
        map_lf = (domain_map_lf if domain_map_lf is not None
                  else pl.scan_parquet(args.domain_map))
        target_families = map_lf.select("pfam_id").unique().collect()
        target_coverage = target_map_coverage(map_lf, target_families,
                                              args.domain_map, job)
        calls_lf = (
            transfer_domains(regions, map_lf, args.min_overlap) if regions is not None else None
        )
        if calls_lf is not None and job["dedup"]:
            iou_min = (args.transfer_dedup_iou if args.transfer_dedup_iou is not None
                       else args.min_overlap)
            # Collected once and reused. This used to be three lazy collects over the same
            # plan -- one to count, one to dedup, one to count again -- and each of them
            # re-read the arm's regions from disk and redid the transfer join. On a gzipped
            # baseline that meant inflating the whole file three times (see
            # inflate_for_scan).
            calls_df = calls_lf.collect()
            before = calls_df.height
            calls_df = dedup_transferred_calls(calls_df, iou_min,
                                               pair_budget=args.dedup_pair_budget)
            after = calls_df.height
            calls_lf = calls_df.lazy()
            if before:
                print(f"[{job['tool']}/{job['variant']}] dedup-transfers: "
                      f"{before} -> {after} calls "
                      f"({100 * (before - after) / before:.1f}% were redundant transfers "
                      f"of the same region)", file=sys.stderr)

    if calls_lf is None:
        scored = pl.DataFrame(
            schema={
                "query_acc": pl.String, "pfam_id": pl.String, "qstart": pl.Int64,
                "qend": pl.Int64, "score": pl.Float64, "iou": pl.Float64,
                "cover": pl.Float64, "true_start": pl.Int64, "true_end": pl.Int64,
                "is_tp": pl.Boolean,
            }
        )
    else:
        scored = score_calls(calls_lf, truth_lf, args.min_overlap, semantics,
                             args.point_semantics)

    scored = classify_scoreable(scored, truth, args.gray_min_annotated_fraction)
    scored.write_parquet(job["calls_out"], compression="zstd")

    # IC is estimated once on the whole answer key, not per stratum. A family's rarity is
    # a property of the proteome; re-estimating it inside each cut would make the same
    # family worth different amounts in different strata and break comparability.
    ic = cm.information_content(truth)

    ident = {"truth_set": args.truth_set,
             "tool": job["tool"], "variant": job["variant"], "species": args.species,
             "species_mya": args.species_mya,
             # Stamped on every row so an alignment tool and a motif tool are never
             # silently compared on boundary metrics that mean different things.
             "interval_semantics": semantics,
             # Both settings are meant to be run and reported side by side; without this
             # stamp the two sets of rows are indistinguishable once pooled.
             "dedup_transfers": bool(job["dedup"]),
             # Same reason, for the truth side. A cell scored by containment and one scored
             # by IoU are not the same measurement, and the row has to say which it is.
             "point_semantics": args.point_semantics}
    # Target-side annotation coverage, on every row. See target_map_coverage: without it a
    # species with almost no target annotation is indistinguishable from a species no tool
    # could reach, and the reachability bar cannot tell the two apart on a truth set whose
    # label vocabulary is small.
    ident.update(target_coverage)
    rows, curves = [], []

    for split in ("all", "selection", "heldout"):
        if split != "all" and "split" not in truth.columns:
            continue
        for axis, value in strata_of(truth):
            t_sub, c_sub = subset(truth, scored, split, axis, value, instance_axes)
            if t_sub.height == 0:
                continue

            reachable = (
                t_sub if target_families is None
                else t_sub.join(target_families, on="pfam_id", how="inner")
            )
            points = operating_points(c_sub, reachable.height)
            m = compute_metrics(c_sub, points, t_sub, reachable, args.min_overlap)

            pc = cm.protein_centric_curve(c_sub, t_sub, ic)
            m.update(cm.cafa_scalars(pc))
            # The same machinery on (protein, family) set membership. Emitted beside the
            # interval reading rather than instead of it: fmax alone scores "never
            # recognised this family" and "recognised it, misplaced the boundary"
            # identically at zero, and the reduced-alphabet question is about which of the
            # two an HP alphabet suffers.
            pc_fam = cm.protein_centric_curve(c_sub, t_sub, ic, level="family")
            m.update(cm.cafa_scalars(pc_fam, prefix="family_"))
            m.update(cm.family_level_counts(c_sub, t_sub))
            m.update(cm.boundary_metrics(c_sub, t_sub, args.strict_iou))
            m.update(cm.domain_count_metrics(c_sub, t_sub))
            m.update(cm.sensitivity_to_first_fp(c_sub, t_sub))
            m.update(ident)
            m.update({
                "split": split, "stratum_axis": axis, "stratum": value,
                # The floor is waived for five axes (UNFLOORED_AXES), so the n that would
                # otherwise have been enforced has to be readable per row instead. Proteins
                # AND instances, because the floor counts proteins while every rate on the
                # row is per instance.
                "n_stratum_proteins": t_sub["accession"].n_unique(),
                # How many distinct labels the reachability join has to work with. A
                # reachability ceiling only means something when `pfam_id` is a FAMILY: on
                # the Swiss-Prot truth set it is one of ~15 feature types, every proteome
                # has nearly all of them, and reachable / truth is then ~1.0 for every
                # species by construction -- recall_reachable is plain recall wearing a
                # reachability label. This column is what lets a reader tell which of the
                # two a row is.
                "n_truth_families": t_sub["pfam_id"].n_unique(),
                # Measured, not a bin midpoint. The reduced-alphabet claim is stated on
                # feature_length / ksize, and this is the numerator for this cell.
                # What share of this cell is point features. 1.0 means every number on the
                # row is a containment result, not a placement result -- which is a
                # different question, and the report must not put the two on one axis.
                "point_fraction": (
                    float(t_sub["is_point"].fill_null(False).mean())
                    if "is_point" in t_sub.columns and t_sub.height else 0.0
                ),
                "median_feature_length": (
                    float(t_sub["feature_length"].median())
                    if "feature_length" in t_sub.columns and t_sub.height else None
                ),
            })
            rows.append(m)

            # Curves only for the ungrouped cut. One per split x stratum x combo would
            # dwarf the metrics they support across a 1017-combo sweep.
            if axis == "all" and job["curve_out"] is not None:
                curves.append(
                    downsample(points, args.max_curve_points).with_columns(
                        pl.lit(split).alias("split"),
                        **{k: pl.lit(v) for k, v in ident.items()},
                    )
                )

    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        job["metrics_out"], compression="zstd"
    )
    if job["curve_out"] is not None:
        (pl.concat(curves, how="diagonal_relaxed") if curves
         else pl.DataFrame(schema={"split": pl.String})).write_parquet(
            job["curve_out"], compression="zstd"
        )

    headline = next(
        (r for r in rows if r["split"] == "all" and r["stratum_axis"] == "all"), {}
    )
    print(json.dumps(
        {k: v for k, v in headline.items() if not k.startswith("stratum")}, indent=2
    ))
    print(f"\nemitted {len(rows)} metric rows across splits x strata")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Optional because --manifest supplies them per row instead. Exactly one of the two
    # forms must be given, checked after parsing.
    p.add_argument("--regions", type=Path)
    p.add_argument("--tool")
    p.add_argument("--variant")
    p.add_argument("--manifest", type=Path,
                   help="TSV of tool<TAB>variant<TAB>regions_path, one row per arm, all "
                        "sharing this species and truth set. Scores every row in one "
                        "process so the answer key is read once rather than once per arm, "
                        "and so a full sweep is 27 SLURM jobs instead of ~10_100.")
    p.add_argument("--species", required=True)
    p.add_argument("--truth-set", default="pfam",
                   help="which truth set this row was measured against")
    p.add_argument("--species-mya", type=float, default=None,
                   help="divergence from human in millions of years; the x-axis for "
                        "per-species plots")
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--domain-map", type=Path)
    p.add_argument("--target-disorder", type=Path,
                   help="optional metapredict parquet for THIS species' proteome; bins each "
                        "human instance by the disorder of the target it could best "
                        "transfer from. Requires an --identity table carrying best_target.")
    p.add_argument("--identity", type=Path,
                   help="per-domain-pair percent identity for this species; the "
                        "twilight-zone stratification axis")
    p.add_argument("--covariates", type=Path,
                   help="per-protein HGNC group / omega / pLDDT / disorder table")
    p.add_argument("--direct-annotation", action="store_true")
    p.add_argument("--kmerseek-rank-by", default="region_enrichment",
                   choices=["jaccard", "region_enrichment", "region_n_shared_kmers",
                            "region_poisson_score"],
                   help="Which kmerseek column ranks calls, bigger is better for all four. "
                        "region_enrichment (default) is region-scoped and normalised by "
                        "the target DB's expected shared k-mers. jaccard is whole-protein "
                        "so it cannot separate regions of one "
                        "pair; region_n_shared_kmers is region_length-ksize+1, so it ranks "
                        "by region length alone; region_poisson_score reproduces the "
                        "pre-2026-08-26 behaviour. The Bonferroni filter is independent of "
                        "this choice and always uses the region Poisson tail.")
    p.add_argument("--kmerseek-max-bonferroni-p", type=float, default=0.05,
                   help="Drop kmerseek regions whose Bonferroni-corrected Poisson tail "
                        "(raw p x region_search_space x db_n_targets) is at or above this. "
                        "0 disables the filter. Applies to kmerseek only.")
    p.add_argument("--dedup-fragments", action="store_true",
                   help="collapse regions duplicated by AlphaFold's overlapping structure "
                        "fragments. Only for arms that read AlphaFold files -- see "
                        "dedup_fragment_regions() for why this is not applied globally.")
    p.add_argument("--fragment-iou", type=float, default=0.9,
                   help="how coincident two regions on the same (query, target) pair must "
                        "be, on BOTH sides, to count as the same alignment seen twice")
    p.add_argument("--dedup-transfers", action="store_true",
                   help="collapse calls that are the same prediction reached through "
                        "different targets: one call per (query, family, region) instead of "
                        "one per target domain hit. Separates detection from the redundancy "
                        "of the target proteome -- run both settings and report the pair. "
                        "See dedup_transferred_calls().")
    p.add_argument("--dedup-transfer-modes", default="off,on",
                   help="which dedup settings to score each manifest arm under, as a comma "
                        "list of 'off' and 'on'. Both by default: the un-deduplicated number "
                        "measures redundant output, the deduplicated one measures detection, "
                        "and the paper wants the pair. Ignored on the single --regions path, "
                        "which takes --dedup-transfers instead.")
    p.add_argument("--transfer-dedup-iou", type=float, default=None,
                   help="how much two calls of the same family on the same query protein "
                        "must overlap each other to count as one annotation "
                        "(default: --min-overlap, i.e. they would claim the same instance)")
    p.add_argument("--point-semantics", choices=["cover", "iou"], default="cover",
                   help="how a POINT truth instance is scored. cover (default) asks "
                        "whether the call contains the annotated residue; iou restores "
                        "the pre-2026-08-27 behaviour, under which no point feature could "
                        "ever be a true positive -- IoU against a 1-residue interval is "
                        "1/call_length, so the criterion was unsatisfiable rather than "
                        "strict. Only affects truth sets carrying is_point: Swiss-Prot and "
                        "M-CSA.")
    p.add_argument("--dedup-pair-budget", type=int, default=DEDUP_PAIR_BUDGET,
                   help="how many pair rows the transfer-dedup self-join may materialise "
                        "at once. Bounds peak memory of that pass, which is otherwise "
                        "quadratic in how many calls of one family land on one query "
                        "protein and is not predictable from the input file size "
                        f"(default: {DEDUP_PAIR_BUDGET})")
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--strict-iou", type=float, default=0.8)
    p.add_argument("--gray-min-annotated-fraction", type=float, default=0.5,
                   help="share of a call that must lie in annotated territory before it "
                        "counts as a confident false positive rather than gray zone")
    # Zinc fingers are INCLUDED by default. The exclusion was inherited from an
    # orthology benchmark, where the scored object is a protein pair and a tandem array
    # inflates protein-level k-mer sharing through repeat content. This benchmark scores
    # DOMAINS: a twelve-finger protein contains twelve domains, and the correct
    # answer is twelve correctly-bounded regions. The confound does not transfer, so the
    # exclusion should not either.
    p.add_argument("--exclude-zinc-finger-from-hgnc", action="store_true",
                   help="restore the orthology-era exclusion of zinc-finger groups from "
                        "the per-group HGNC sweep (off by default; see the note above)")
    p.add_argument("--interval-semantics", choices=["alignment", "motif"],
                   default="alignment",
                   help="motif for tools reporting discontinuous residue sets (Folddisco)")
    # Not required under --manifest: output names are derived per row from truth_set,
    # tool, variant and species, because one process writes a trio per arm.
    p.add_argument("--calls-out", type=Path)
    p.add_argument("--metrics-out", type=Path)
    p.add_argument("--curve-out", type=Path)
    p.add_argument("--max-curve-points", type=int, default=2000)
    args = p.parse_args()

    if not args.direct_annotation and args.domain_map is None:
        raise SystemExit("--domain-map is required unless --direct-annotation is set")
    if bool(args.manifest) == bool(args.regions):
        raise SystemExit("pass exactly one of --manifest or --regions")
    if args.regions and not (args.tool and args.variant
                             and args.calls_out and args.metrics_out):
        raise SystemExit("--regions requires --tool, --variant, --calls-out and --metrics-out")

    truth_lf, domain_map_lf = read_shared_inputs(args)
    truth = truth_lf.collect()
    covariates = pl.read_parquet(args.covariates) if args.covariates else None
    identity = None
    if args.identity and args.identity.exists() and args.identity.stat().st_size > 0:
        try:
            identity = pl.read_parquet(args.identity)
        except Exception:
            identity = None   # sentinel file when --skip_identity is set
    truth = attach_strata(truth, covariates,
                          keep_zinc_finger=not args.exclude_zinc_finger_from_hgnc)
    truth = attach_identity(truth, identity)
    truth = attach_feature_length(truth)
    truth = attach_feature_type(truth)

    target_disorder = None
    if (args.target_disorder and args.target_disorder.exists()
            and args.target_disorder.stat().st_size > 0):
        try:
            target_disorder = pl.read_parquet(args.target_disorder)
        except Exception:
            target_disorder = None   # sentinel file when the arm is skipped
    truth = attach_target_disorder(truth, target_disorder)

    # Once, not once per arm: the truth table is shared across every job in this task, so
    # which axes are instance-level is a property of the answer key and not of the tool.
    instance_axes = instance_level_axes(truth)
    print(f"instance-level strata (TP restricted per cut): "
          f"{', '.join(sorted(instance_axes)) or 'none'}", file=sys.stderr)

    # One job per (tool, variant) when batched, or the single --regions when not. Batching
    # exists because SLURM rate-limits submission: one task per (truth_set, species, tool,
    # variant) is ~10_100 sbatch calls for a full sweep, against 27 when grouped by species.
    # Each manifest row is scored once per dedup mode. The two live in the same task because
    # the expensive setup -- reading and IC-weighting the truth table -- is shared, and
    # because the pair is only meaningful compared to itself: same truth, same calls, one
    # difference. The un-deduplicated stem is left exactly as it was so already-published
    # paths and the globs that read them keep working; only the new mode gets a suffix.
    modes = [m.strip() for m in args.dedup_transfer_modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in ("off", "on")]
    if bad:
        raise SystemExit(f"--dedup-transfer-modes takes 'off' and/or 'on', got {bad}")
    if not modes:
        raise SystemExit("--dedup-transfer-modes needs at least one mode")

    if args.manifest:
        jobs = []
        for line in Path(args.manifest).read_text().splitlines():
            if not line.strip():
                continue
            tool, variant, regions = line.split("\t")
            for mode in modes:
                stem = f"{args.truth_set}.{tool}.{variant}.{args.species}"
                if mode == "on":
                    stem += ".dedup"
                jobs.append({"tool": tool, "variant": variant, "regions": Path(regions),
                             "dedup": mode == "on",
                             "calls_out": Path(f"{stem}.calls.parquet"),
                             "metrics_out": Path(f"{stem}.metrics.parquet"),
                             "curve_out": Path(f"{stem}.curve.parquet")})
    else:
        # The single-arm path keeps taking the plain boolean: output names are given
        # explicitly there, so there is nowhere to put a second mode's files.
        jobs = [{"tool": args.tool, "variant": args.variant, "regions": args.regions,
                 "dedup": bool(args.dedup_transfers),
                 "calls_out": args.calls_out, "metrics_out": args.metrics_out,
                 "curve_out": args.curve_out}]

    for i, job in enumerate(jobs, 1):
        tag = f"{job['tool']}/{job['variant']}" + (" [dedup]" if job["dedup"] else "")
        print(f"[{i}/{len(jobs)}] {tag}", file=sys.stderr)
        score_one(args, truth, truth_lf, job, instance_axes,
                  domain_map_lf=domain_map_lf)
        # Jobs for one arm are adjacent (one per dedup mode), so an arm's inflated copy is
        # dead as soon as the next job reads a different file.
        if i == len(jobs) or jobs[i]["regions"] != job["regions"]:
            release_inflated(job["regions"])


if __name__ == "__main__":
    main()
