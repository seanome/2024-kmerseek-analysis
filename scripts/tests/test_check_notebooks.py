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
    nb = write(
        tmp_path, [code_cell("x = 14_873", 1), code_cell("fig.savefig('a.png')", 2)]
    )
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
    assert "separate numbers" in text
    assert "plt.show()" in text
    assert "KeyError" in text
    assert "queue name" in text
    assert "not run top to bottom" in text


def test_ordinary_tuples_and_ranges_are_not_flagged(tmp_path):
    src = "ax.set_xlim(0, 1)\nxs = range(1, 100)\npair = (14, 873)\nlo, hi = 0, 500"
    nb = write(tmp_path, [code_cell(src, 1)])
    assert check_notebooks.check_notebook(nb) == []


def test_comma_integers_outside_an_assignment(tmp_path):
    # Each of these runs, and each gives the wrong value.
    for src in [
        "def f():\n    return 14,873",
        "x = [1,000]",
        "ok = n == 1,000",
        "sizes = f(1,000)",
        "n = 14, 873  # black adds the space",
    ]:
        nb = write(tmp_path, [code_cell(src, 1)])
        assert "separate numbers" in "\n".join(check_notebooks.check_notebook(nb)), src


def test_plt_show_in_a_comment_or_string_is_not_flagged(tmp_path):
    src = "# never call plt.show() here\nmsg = 'no plt.close() please'"
    nb = write(tmp_path, [code_cell(src, 1)])
    assert check_notebooks.check_notebook(nb) == []


def test_plt_show_after_a_magic_line_is_flagged(tmp_path):
    nb = write(tmp_path, [code_cell("%matplotlib inline\nplt.show()", 1)])
    assert "plt.show()" in "\n".join(check_notebooks.check_notebook(nb))


def test_traceback_printed_to_stderr_is_flagged(tmp_path):
    stderr = {
        "output_type": "stream",
        "name": "stderr",
        "text": ["Traceback (most recent call last):\n", "KeyError: 'x'\n"],
    }
    nb = write(tmp_path, [code_cell("a = 1", 1, outputs=[stderr])])
    assert "traceback" in "\n".join(check_notebooks.check_notebook(nb))


def test_secret_in_a_markdown_cell_is_flagged(tmp_path):
    md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": "Job ran under arn:aws:batch:us-west-2:123456789012:job-queue/x",
    }
    nb = write(tmp_path, [md, code_cell("a = 1", 1)])
    assert "ARN" in "\n".join(check_notebooks.check_notebook(nb))


def test_main_exit_status(tmp_path):
    good = write(tmp_path, [code_cell("a = 1", 1)])
    assert check_notebooks.main([good]) == 0
    bad = tmp_path / "bad.ipynb"
    bad.write_text(json.dumps({"cells": [code_cell("plt.close()", 1)]}))
    assert check_notebooks.main([str(bad)]) == 1


def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


TABLE = {"output_type": "execute_result", "data": {"text/plain": "shape: (3, 2)\n┌─"}}
IMAGE = {"output_type": "display_data", "data": {"image/png": "iVBOR"}}


def test_section_with_table_and_no_figure_is_flagged(tmp_path):
    nb = write(
        tmp_path,
        [
            md_cell("# 241: title"),
            code_cell("df", 1, outputs=[TABLE]),
            md_cell("## 1. Which arms got an E-value"),
            code_cell("df", 2, outputs=[TABLE]),
            md_cell("## 2. Ranks"),
            code_cell("df", 3, outputs=[TABLE]),
            code_cell("fig.savefig('a.png')", 4),
            md_cell("## 3. Shown as an image"),
            code_cell("df", 5, outputs=[TABLE, IMAGE]),
        ],
    )
    problems = check_notebooks.check_notebook(nb)
    assert len(problems) == 1
    assert "'1. Which arms got an E-value' shows a table but no figure" in problems[0]


def test_shared_notebook_number_is_flagged(tmp_path):
    cells = [code_cell("a = 1")]
    for name in ("241_alphabet_ranking.ipynb", "241_other.ipynb", "242_next.ipynb"):
        (tmp_path / name).write_text(json.dumps({"cells": cells, "metadata": {}}))
    text = "\n".join(check_notebooks.check_notebook(str(tmp_path / "241_other.ipynb")))
    assert "also used by 241_alphabet_ranking.ipynb" in text
    assert check_notebooks.check_notebook(str(tmp_path / "242_next.ipynb")) == []
