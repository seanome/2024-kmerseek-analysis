import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_notebooks  # noqa: E402


def code_cell(source, count=None, outputs=(), hidden=True):
    return {
        "cell_type": "code",
        "execution_count": count,
        "metadata": {"jupyter": {"source_hidden": hidden}},
        "outputs": list(outputs),
        "source": source,
    }


def write(tmp_path, cells):
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4}))
    return str(path)


def test_clean_notebook_passes(tmp_path):
    nb = write(tmp_path, [code_cell("x = 14_873", 1), code_cell("fig.savefig('a.png')", 2)])
    assert check_notebooks.check_notebook(nb) == []


def test_unrun_template_passes(tmp_path):
    nb = write(tmp_path, [code_cell("a = 1"), code_cell("b = 2")])
    assert check_notebooks.check_notebook(nb) == []


def test_each_rule_fires(tmp_path):
    error = {"output_type": "error", "ename": "KeyError", "evalue": "", "traceback": []}
    nb = write(
        tmp_path,
        [
            code_cell("n = 14,873", 1, hidden=False),
            code_cell("plt.show()", 3, outputs=[error]),
            code_cell("q = 'TowerForge-3abcdefghijk'", None),
        ],
    )
    text = "\n".join(check_notebooks.check_notebook(nb))
    assert "not collapsed" in text
    assert "is a tuple" in text
    assert "plt.show()" in text
    assert "KeyError" in text
    assert "queue name" in text
    assert "not run top to bottom" in text


def test_comma_inside_a_call_is_not_flagged(tmp_path):
    nb = write(tmp_path, [code_cell("ax.set_xlim(0, 1)\nsizes = f(1,000)", 1)])
    assert check_notebooks.check_notebook(nb) == []


def test_main_exit_status(tmp_path):
    good = write(tmp_path, [code_cell("a = 1", 1)])
    assert check_notebooks.main([good]) == 0
    bad = tmp_path / "bad.ipynb"
    bad.write_text(json.dumps({"cells": [code_cell("plt.close()", 1)]}))
    assert check_notebooks.main([str(bad)]) == 1
