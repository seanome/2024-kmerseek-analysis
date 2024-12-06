from typing import Literal

import itertools
import warnings

from IPython.display import display
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    DetCurveDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    det_curve,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
    confusion_matrix,
)

CURVES = Literal["precision_recall", "roc", "det"]

from scop_constants import SCOP_LINEAGES

SOURMASH_SCORE_COLS = Literal[
    "containment",
    "tf_idf_score",
    "containment_adjusted_log10",
    "inverse_prob_overlap_adjusted",
]


class MultiSearchResultEvaluator:

    BINARY_CLASSIFICATION_SCORERS = {
        "average_precision": average_precision_score,
        "roc_auc": roc_auc_score,
        # "top_k_accuracy": top_k_accuracy_score,
    }

    MULTICLASS_SCORERS = {
        "recall": lambda x, y: recall_score(x, y, average="macro"),
        "f1": lambda x, y: f1_score(x, y, average="macro"),
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
        # "top_k_accuracy": top_k_accuracy_score,
    }

    CLASSIFICATION_SCORERS = list(BINARY_CLASSIFICATION_SCORERS.keys()) + list(
        MULTICLASS_SCORERS.keys()
    )

    def __init__(
        self,
        multisearch: pd.DataFrame,
        ksize: int,
        moltype: MOLTYPES,
        verbose: bool = False,
    ):
        self.multisearch_original: pd.DataFrame = multisearch
        self.ksize: int = ksize
        self.moltype: MOLTYPES = moltype
        self._add_binary_classification_columns()
        self.multisearch: pd.DataFrame = self._remove_self_and_spurious_matches()
        self.verbose: bool = verbose

    def _remove_self_and_spurious_matches(self) -> pd.DataFrame:
        # Remove self-matches and matches with a single hash (probably spurious)
        # Ignore self matches
        print(f"Before removing self-matches: {multisearch.shape}")
        df_no_self = self.multisearch_original.query("query_md5 != match_md5")
        print(f"After  removing self-matches: {df_no_self.shape}")
        df_no_spurious = df_no_self.query("intersect_hashes > 1")
        print(f"After  removing spurious matches (only 1 hash): {df_no_spurious.shape}")
        return df_no_spurious

    def _add_binary_classification_columns(self):
        # Add whether the query's SCOP lineage is the same as the "match"'s SCOP lineage
        for lineage in SCOP_LINEAGES:
            query = f"query_{lineage}"
            match = f"match_{lineage}"
            self.multisearch[lineage] = (
                self.multisearch[query] == self.multisearch[match]
            )

    def _add_inverse_prob_overlap(self):
        # Take the inverse of probability of overlap, so that bigger is better
        self.multisearch["inverse_prob_overlap_adjusted"] = (
            1 / self.multisearch["prob_overlap_adjusted"]
        )

    @property
    def CURVE_DF_MAKERS(self):
        # This has to be a property vs defined before __init__ since it references "self"
        return {
            "precision_recall": self.make_precision_recall_curve_df,
            "roc": self.make_roc_curve_df,
            "det": self.make_det_curve_df,
        }

    @staticmethod
    def make_precision_recall_curve_df(y_true, y_score):
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        # Add a dummy value to thresholds so all the arrays are the same length and
        # creating a dataframe doesn't cause an error
        thresholds = np.concat([thresholds, [thresholds.max()]])
        df = pd.DataFrame(
            dict(precision=precision, recall=recall, thresholds=thresholds)
        )
        return df

    @staticmethod
    def make_roc_curve_df(y_true, y_score):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        df = pd.DataFrame(dict(fpr=fpr, tpr=tpr, thresholds=thresholds))
        return df

    @staticmethod
    def make_det_curve_df(y_true, y_score):
        fpr, fnr, thresholds = det_curve(y_true, y_score)
        df = pd.DataFrame(dict(fpr=fpr, fnr=fnr, thresholds=thresholds))
        return df

    def make_curve_df_ksize_moltype(
        self,
        y_true,
        y_score,
        lineage,
        sourmash_score: SOURMASH_SCORE_COLS,
        curve: CURVES,
    ):
        df_maker = self.CURVE_DF_MAKERS[curve]
        df = df_maker(y_true, y_score)
        df["lineage"] = lineage
        df["sourmash_score"] = sourmash_score
        df["moltype"] = self.moltype
        df["ksize"] = self.ksize
        return df

    def make_score_line(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        sklearn_score_name: str,
        lineage: SCOP_LINEAGES,
        sourmash_score_name,
    ):
        try:
            scorer = self.BINARY_CLASSIFICATION_SCORERS[sklearn_score_name]
        except KeyError:
            scorer = self.MULTICLASS_SCORERS[sklearn_score_name]
        score_value = scorer(y_true, y_pred)
        line = [
            self.moltype,
            self.ksize,
            lineage,
            sourmash_score_name,
            score_value,
            sklearn_score_name,
        ]
        return line

    def make_precision_recall_fscore_support_df(
        self, y_true, y_pred, lineage, sourmash_score_name
    ):
        p, r, f, s = self.precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )

        df = pd.DataFrame(dict(precision=p, recall=r, fscore=f, support=s))
        df["lineage"] = lineage
        df["sourmash_score"] = sourmash_score_name
        df["moltype"] = self.moltype
        df["ksize"] = self.ksize
        return df

    def maybe_score_binary(
        self,
        y_true_bool: pd.Series[bool],
        y_score: pd.Series[float],
        sourmash_score_name: SOURMASH_SCORE_COLS,
        lineage: SCOP_LINEAGES,
        lines: list,
    ):
        for sklearn_score in self.BINARY_CLASSIFICATION_SCORERS.keys():
            try:
                line = self.make_score_line(
                    y_true_bool,
                    y_score,
                    sklearn_score,
                    lineage,
                    sourmash_score_name,
                )
                lines.append(line)
            except ValueError:
                # skip when ROC isn't defined, when only one class is present
                pass

    def maybe_score_multiclass(
        self,
        y_true: pd.Series[str],
        y_pred: pd.Series[str],
        sourmash_score_name: SOURMASH_SCORE_COLS,
        lineage,
        lines: list,
    ):
        for sklearn_score in self.MULTICLASS_SCORERS.keys():
            # try:
            line = self.make_score_line(
                y_true,
                y_pred,
                sklearn_score,
                lineage,
                sourmash_score_name,
            )
            lines.append(line)

    def maybe_make_curve_dfs(
        self, y_true_bool, y_score, sourmash_score_name, lineage, curve_dfs
    ):
        for curve in self.CURVE_DF_MAKERS.keys():
            try:
                with warnings.catch_warnings(action="ignore"):
                    # Ignore warnings that show up only one class in y_true
                    # UndefinedMetricWarning: No negative samples in y_true,
                    # false positive value should be meaningless
                    df = self.make_curve_df_ksize_moltype(
                        y_true_bool,
                        y_score,
                        lineage,
                        sourmash_score_name,
                        curve,
                    )

                    curve_dfs[curve].append(df)
            except ValueError:
                # skip when ROC or DET aren't defined, when only one class is present
                pass

    def concatenate_curve_dataframes(self, curve_dfs):
        self.precision_recall = self._concatenate_single_curve_dfs(
            curve_dfs, "precision_recall"
        )
        self.det = self._concatenate_single_curve_dfs(curve_dfs, "det")
        self.roc = self._concatenate_single_curve_dfs(curve_dfs, "roc")

    def _concatenate_single_curve_dfs(self, curve_dfs, curve_name):
        df = pd.concat(curve_dfs[curve_name])
        self._maybe_print_df_info(f"self.{curve_name}", df)

    def _maybe_print_df_info(self, df_name, df):
        if self.verbose:
            print(f"--- Created {df_name} ---")
            print(df.shape)
            display(df.head())
            display(df.describe())

    def concatenate_classification_metrics(self, lines):
        self.classification_metrics = pd.DataFrame(
            lines,
            columns=[
                "moltype",
                "ksize",
                "lineage",
                "sourmash_score",
                "metric",
                "metric_name",
            ],
        )
        self._maybe_print_df_info(
            "self.classification_metrics", self.classification_metrics
        )

    def compute_classification_metrics(self):

        curve_dfs = {
            "precision_recall": [],
            "roc": [],
            "det": [],
        }

        lines = []

        confusion_matrices = {}

        for lineage in SCOP_LINEAGES:
            for sourmash_score_name in self.SOURMASH_SCORE_COLS:
                y_true_bool = self.multisearch[lineage]

                y_true = self.multisearch[f"query_{lineage}"]
                y_pred = self.multisearch[f"match_{lineage}"]
                y_score = self.multisearch[sourmash_score_name]

                matrix = confusion_matrix(y_true, y_pred)
                key = (self.moltype, self.ksize, sourmash_score_name, lineage)
                confusion_matrices[key] = matrix

                self.maybe_score_multiclass(y_true, y_pred, sourmash_score_name, lines)
                self.maybe_score_binary(
                    y_true_bool, y_score, sourmash_score_name, lines
                )
                self.maybe_make_curve_dfs(
                    y_true_bool, y_score, sourmash_score_name, lineage, curve_dfs
                )

        self.concatenate_curve_dataframes(curve_dfs)
        self.concatenate_classification_metrics(lines)


def make_multisearch_pq(
    outdir: str,
    ksize: int,
    moltype: MOLTYPES,
) -> str:
    """Creates a path for the processed multisearch results file

    Args:
        outdir (str): S3 base path
        ksize (int): K-mer size
        moltype (MOLTYPES): Molecular alphabet used for amino acids, one of 'protein', 'dayhoff', 'hp'

    Returns:
        str: _description_
        Example:
        's3://seanome-kmerseek/scope-benchmark/analysis-outputs/2024-10-09__protein_k5-20/00_cleaned_multisearch_results/scope40.multisearch.protein.k20.pq'
    """
    basename = f"scope40.multisearch.{moltype}.k{ksize}.pq"
    pq = f"{outdir}/00_cleaned_multisearch_results/{basename}"
    return pq


if __name__ == "__main__":
    ksizes = range(7, 21)
    analysis_outdir = "s3://seanome-kmerseek/scope-benchmark/analysis-outputs/2024-10-09__protein_k5-20"
    for ksize in ksizes:
        print(f"--- ksize: {ksize} ---")
        pq = make_multisearch_pq(ksize, analysis_outdir, "protein")
        multisearch = pd.read_parquet(pq)
