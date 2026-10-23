"""End to end: rewrite a small repo's history, push it to a stand-in for GitHub, and move
an existing clone (with a worktree holding an uncommitted edit) onto the new history."""

import base64
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("git_filter_repo")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import rewrite_history_256_colours as rh


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


def figure_png(seed):
    fig, ax = plt.subplots(figsize=(3, 2))
    for i in range(4):
        ax.plot(range(30), [((j * (i + seed)) % 7) for j in range(30)], lw=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def notebook(png):
    return json.dumps({
        "cells": [{"cell_type": "code", "execution_count": 1, "metadata": {},
                   "source": ["plot()"], "outputs": [{
                       "output_type": "display_data", "metadata": {},
                       "data": {"image/png": base64.b64encode(png).decode(),
                                "text/plain": ["<Figure>"]}}]}],
        "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1)


def mode(path):
    return Image.open(path).mode


def test_prepare_push_migrate(tmp_path, capsys):
    github = tmp_path / "github.git"
    git(tmp_path, "init", "--bare", "-q", "-b", "main", str(github))
    author = tmp_path / "author"
    git(tmp_path, "clone", "-q", str(github), str(author))
    for k, v in {"user.name": "t", "user.email": "t@t"}.items():
        git(author, "config", k, v)
    (author / "fig.png").write_bytes(figure_png(1))
    (author / "nb.ipynb").write_text(notebook(figure_png(2)))
    (author / "notes.txt").write_text("one\n")
    git(author, "add", ".")
    git(author, "commit", "-q", "-m", "first")
    first = git(author, "rev-parse", "HEAD")
    (author / "notes.txt").write_text("two\n")
    git(author, "commit", "-q", "-am", f"second, after {first[:8]}")
    git(author, "switch", "-q", "-c", "feature", first)
    (author / "fig.png").write_bytes(figure_png(3))
    git(author, "commit", "-q", "-am", "feature figure")
    git(author, "push", "-q", "origin", "main", "feature")

    # Someone's clone: main checked out, feature in a worktree with an uncommitted edit.
    clone = tmp_path / "clone"
    git(tmp_path, "clone", "-q", str(github), str(clone))
    git(clone, "branch", "-q", "feature", "origin/feature")
    wt = tmp_path / "wt"
    git(clone, "worktree", "add", "-q", str(wt), "feature")
    (wt / "notes.txt").write_text("my unsaved edit\n")
    # A branch with a commit that was never pushed.
    git(clone, "branch", "-q", "local-only", "main")
    lw = tmp_path / "lw"
    git(clone, "worktree", "add", "-q", str(lw), "local-only")
    for k, v in {"user.name": "t", "user.email": "t@t"}.items():
        git(lw, "config", k, v)
    (lw / "local.txt").write_text("not pushed\n")
    git(lw, "add", "local.txt")
    git(lw, "commit", "-q", "-m", "not pushed")
    git(clone, "worktree", "remove", str(lw))
    old_local = git(clone, "rev-parse", "local-only")
    old_main = git(clone, "rev-parse", "main")
    assert mode(clone / "fig.png") == "RGBA"

    work = tmp_path / "rewrite"
    rh.prepare(work, str(github))
    rh.push(work, dry_run=True)
    assert rh.remote_tips(str(github))["refs/heads/main"] == old_main  # dry run sent nothing
    rh.push(work, dry_run=False)
    new_tips = rh.remote_tips(str(github))
    assert new_tips["refs/heads/main"] != old_main

    capsys.readouterr()
    rh.migrate(work / "commit-map.txt", str(clone))
    out = capsys.readouterr().out
    # The unpushed branch is left alone, and the rebase it suggests moves it over.
    assert git(clone, "rev-parse", "local-only") == old_local
    hint = next(line for line in out.splitlines() if "git rebase --onto" in line)
    subprocess.run(hint.strip().split(), cwd=clone, check=True, capture_output=True)
    assert git(clone, "rev-parse", "local-only~1") == new_tips["refs/heads/main"]
    assert git(clone, "show", "local-only:local.txt") == "not pushed"
    git(clone, "switch", "-q", "main")
    assert git(clone, "rev-parse", "main") == new_tips["refs/heads/main"]
    assert git(clone, "rev-parse", "feature") == new_tips["refs/heads/feature"]
    assert mode(clone / "fig.png") == "P" and mode(wt / "fig.png") == "P"
    nb = json.loads((clone / "nb.ipynb").read_text())
    image = nb["cells"][0]["outputs"][0]["data"]["image/png"]
    assert Image.open(io.BytesIO(base64.b64decode(image))).mode == "P"
    # The uncommitted edit is still there, and it is the only change in the worktree.
    assert (wt / "notes.txt").read_text() == "my unsaved edit\n"
    assert git(wt, "status", "--porcelain") == "M notes.txt"
    assert git(clone, "status", "--porcelain") == ""
    # The commit message's short SHA now names the rewritten first commit.
    pairs = rh.read_commit_map(work / "commit-map.txt")
    assert pairs[first][:8] in git(clone, "log", "-1", "--format=%s", "main")


def test_push_refuses_when_github_moved(tmp_path):
    github = tmp_path / "github.git"
    git(tmp_path, "init", "--bare", "-q", "-b", "main", str(github))
    author = tmp_path / "author"
    git(tmp_path, "clone", "-q", str(github), str(author))
    for k, v in {"user.name": "t", "user.email": "t@t"}.items():
        git(author, "config", k, v)
    (author / "fig.png").write_bytes(figure_png(1))
    git(author, "add", ".")
    git(author, "commit", "-q", "-m", "first")
    git(author, "push", "-q", "origin", "main")
    work = tmp_path / "rewrite"
    rh.prepare(work, str(github))
    (author / "late.txt").write_text("pushed after prepare\n")
    git(author, "add", ".")
    git(author, "commit", "-q", "-m", "late")
    git(author, "push", "-q", "origin", "main")
    with pytest.raises(SystemExit, match="changed on GitHub after"):
        rh.push(work, dry_run=False)
