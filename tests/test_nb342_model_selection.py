"""Regression checks for preprocessing inside gene cross-validation in notebook 342."""

import ast
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def notebook_functions():
    path = (
        Path(__file__).resolve().parents[1]
        / "notebooks/342_exon_function_pattern_label_vs_splicing_setA.ipynb"
    )
    cells = json.loads(path.read_text())["cells"]
    nodes = []
    for cell in cells:
        if cell["cell_type"] == "code":
            nodes += [
                node
                for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"design", "fit_and_score"}
            ]
    scope = dict(
        np=np,
        pl=pl,
        StandardScaler=StandardScaler,
        LogisticRegression=LogisticRegression,
        GridSearchCV=GridSearchCV,
        GroupKFold=GroupKFold,
        make_pipeline=make_pipeline,
        N_FOLDS=3,
    )
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def test_scaler_is_fitted_inside_each_cv_training_fold():
    scope = notebook_functions()
    sizes = []

    class TrackingScaler(StandardScaler):
        def fit(self, X, y=None, sample_weight=None):
            sizes.append(len(X))
            return super().fit(X, y, sample_weight=sample_weight)

    scope["StandardScaler"] = TrackingScaler
    scope["data"] = pl.DataFrame(
        [
            dict(
                gene_name=f"g{g}",
                half="tune" if g < 12 else "test",
                outcome="changed" if event else "unchanged",
                x=float(g + event),
            )
            for g in range(18)
            for event in [0, 1]
        ]
    )
    result = scope["fit_and_score"]("changed", ["x"])
    assert len(result["score"]) == 12
    assert sizes.count(24) == 1  # final refit, after the 39 fold-specific fits
    assert len(sizes) == 40
    assert all(n < 24 for n in sizes[:-1])


def test_missing_indicator_schema_is_kept_for_test_without_missing_values():
    design = notebook_functions()["design"]
    train = design(pl.DataFrame({"x": [None, 1.0]}), ["x"])
    test = design(pl.DataFrame({"x": [2.0]}), ["x"], {"x"})
    assert test.columns == train.columns
    assert test["x_missing"].to_list() == [0.0]
