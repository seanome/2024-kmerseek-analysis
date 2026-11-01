"""Figure SF1 of notebook 352: per-query wall time against kmerseek index size, log-log.

Measured points are filled circles. The least-squares line is drawn only between the
smallest and largest measured index sizes; nothing is drawn beyond the last point.
"""

from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np

import pubfig as pf

MEASURED_COLOUR = pf.OKABE_ITO["blue"]  # kmerseek's colour in the paper's figures
LINE_COLOUR = pf.OKABE_ITO["black"]


def _plain_log_ticks(axis) -> None:
    """Plain numbers (1, 2, 5, 10, 20 ...) on a log axis, never 10^2."""
    axis.set_major_locator(mticker.LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())


def plot_sf1(index_gb, median_s, p95_s, fit_median: dict, fit_p95: dict, setting: str):
    """Two panels: a, median; b, 95th percentile of per-query wall seconds.

    index_gb: index size on disk in GB (1e9 bytes), one value per measured size.
    fit_*: output of kmerseek_scale_352.loglog_fit for that series.
    Returns the figure; the caller saves it.
    """
    pf.use_style()
    fig, axs = pf.figure(pf.TWO_COLUMN_MM, 70, ncols=2)
    x = np.asarray(index_gb, float)
    xs = np.geomspace(x.min(), x.max(), 50)
    for ax, y, fit, ylabel in (
        (axs[0], median_s, fit_median, "Median query wall time (s)"),
        (axs[1], p95_s, fit_p95, "95th percentile query wall time (s)"),
    ):
        y = np.asarray(y, float)
        ax.plot(
            xs,
            10 ** (fit["intercept"] + fit["slope"] * np.log10(xs)),
            color=LINE_COLOUR,
            label="Least-squares line through the measured points (stops at the largest one)",
            zorder=1,
        )
        ax.scatter(
            x,
            y,
            s=30,
            color=MEASURED_COLOUR,
            edgecolor="none",
            label="Measured, one point per database size",
            zorder=2,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Index size on disk (GB)")
        ax.set_ylabel(ylabel)
        _plain_log_ticks(ax.xaxis)
        _plain_log_ticks(ax.yaxis)
        ax.text(
            0.03,
            0.97,
            f"exponent {fit['slope']:.2f}, 95% interval {fit['ci_low']:.2f} to {fit['ci_high']:.2f}\n"
            f"({fit['n_points']} points, {fit['df']} degree of freedom)",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        # The x range ends at the last measured point: no room is left for a line beyond it.
        ax.set_xlim(x.min() / 1.6, x.max() * 1.6)
    pf.shared_legend(fig, ncol=2)
    fig.text(
        0.5,
        -0.02,
        f"kmerseek {setting}; databases are nested UniParc subsets; 1,000 queries each",
        ha="center",
        va="top",
    )
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    return fig
