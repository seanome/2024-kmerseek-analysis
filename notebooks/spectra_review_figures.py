"""Save spectrum diagnostics at their printed size."""

from pathlib import Path
import math
import matplotlib.pyplot as plt
from matplotlib.text import Text
from IPython.display import Image, display
import pubfig as pf


def save_spectrum(fig, path):
    width, height = fig.get_size_inches()
    factor = min(183 / 25.4 / width, 170 / 25.4 / height, 1)
    fig.set_size_inches(width * factor, height * factor)
    for text in fig.findobj(Text):
        text.set_fontsize(6)
        text.set_fontfamily("Arial")
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(7)
        ax.yaxis.label.set_fontsize(7)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        rows = math.ceil(len(legend.get_texts()) / max(1, legend._ncols))
        position = ax.get_position()
        room = (rows * 0.15 + 0.18) / fig.get_size_inches()[1]
        height = max(position.height - room, position.height * 0.55)
        ax.set_position([position.x0, position.y0, position.width, height])
        legend.set_loc("lower left")
        legend.set_bbox_to_anchor((0, 1.01))
        ax.title.set_position(
            (0.5, 1.03 + (rows * 0.15) / (height * fig.get_size_inches()[1]))
        )
    for legend in fig.legends:
        legend.set_loc("upper center")
        legend.set_bbox_to_anchor((0.5, 0.98))
    path = Path(path)
    pf.save(fig, path.with_suffix(""))
    display(Image(filename=str(path)))
    plt.close(fig)
