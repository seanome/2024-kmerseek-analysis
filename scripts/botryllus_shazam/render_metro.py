"""Render with nf-metro 2.1.0; move legends above geometry for reading order."""

import re
import subprocess
import tempfile
import xml.etree.ElementTree as E
from pathlib import Path
import sys

repo = Path(__file__).resolve().parents[2]
folder = repo / "reports/botryllus-shazam/metro"
S = "http://www.w3.org/2000/svg"
E.register_namespace("", S)
for name in [
    "workflow-overview",
    "search-collections",
    "reads-discovery",
    "protein",
    "context-archive",
    "archive-delivery",
    "metadata-records",
]:
    source = folder / (name + ".mmd")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "map.svg"
        subprocess.run(
            [
                sys.argv[1],
                "render",
                str(source),
                "-o",
                str(raw),
                "--mode",
                "light",
                "--strict",
                "--row-align",
                "top",
                "--section-x-gap",
                "40",
            ],
            check=True,
        )
        root = E.parse(raw).getroot()
        width = float(root.attrib["width"])
        if name == "workflow-overview":
            # Keep the final Results heading above its box, aligned with Search.
            # nf-metro places it below to avoid a route that is well above the box.
            headings = [
                e for e in root.iter() if e.attrib.get("data-section-id") == "retrieval"
            ]
            label = next(
                e for e in headings if e.attrib.get("class") == "nf-metro-section-label"
            )
            badge = next(
                e
                for e in headings
                if e.attrib.get("class") == "nf-metro-section-num-circle"
            )
            for e in root.iter():
                if e.attrib.get("data-section-id") != "matches":
                    continue
                if e.attrib.get("class") == "nf-metro-section-label":
                    e.set("y", label.attrib["y"])
                elif e.attrib.get("class") == "nf-metro-section-num-circle":
                    e.set("cy", badge.attrib["cy"])
                elif e.tag.endswith("text") and (e.text or "").strip() == "8":
                    e.set("y", badge.attrib["cy"])

        for track in root.iter():
            if track.attrib.get("data-line-id") in {
                "reads",
                "discovery",
            } and track.tag.endswith("path"):
                track.set("stroke-width", "5")
                if track.attrib.get("data-line-id") == "discovery":
                    track.set("stroke-dasharray", "14,10")
        title = next(x for x in root if x.attrib.get("class") == "nf-metro-title")
        legend = next(x for x in root if x.attrib.get("class") == "nf-metro-legend-bg")
        height = float(root.attrib["height"]) + 135
        new = E.Element(
            f"{{{S}}}svg",
            dict(
                width=str(width),
                height=str(height),
                viewBox=f"0 0 {width} {height}",
                style="color-scheme: light",
            ),
        )
        E.SubElement(new, f"{{{S}}}title").text = title.text
        E.SubElement(
            new, f"{{{S}}}rect", dict(width="100%", height="100%", fill="white")
        )
        E.SubElement(
            new,
            f"{{{S}}}text",
            dict(
                x="25",
                y="30",
                **{"font-family": "Arial", "font-size": "23", "font-weight": "bold"},
            ),
        ).text = title.text
        x = 25
        for line in source.read_text().splitlines():
            m = re.match(r"%%metro line: (.*?) \| (.*?) \| (#[a-fA-F0-9]+)", line)
            if m:
                legend_track = E.SubElement(
                    new,
                    f"{{{S}}}path",
                    dict(d=f"M{x},60 h30", stroke=m[3], **{"stroke-width": "5"}),
                )
                if line.endswith("| dashed"):
                    legend_track.set("stroke-dasharray", "14,10")
                E.SubElement(
                    new,
                    f"{{{S}}}text",
                    dict(
                        x=str(x + 40),
                        y="65",
                        **{"font-family": "Arial", "font-size": "15"},
                    ),
                ).text = m[2]
                x += len(m[2]) * 9 + 100
        E.SubElement(
            new,
            f"{{{S}}}text",
            dict(x="25", y="95", **{"font-family": "Arial", "font-size": "15"}),
        ).text = "Circle = program. Document = file. Folder = index directory. Read left to right."
        g = E.SubElement(new, f"{{{S}}}g", dict(transform="translate(0,135)"))
        for element in root:
            if element is legend:
                break
            if element is title or element.attrib.get("class") == "nf-metro-bg":
                continue
            g.append(element)
        E.ElementTree(new).write(folder / (name + ".svg"), encoding="unicode")

subprocess.run(
    [sys.executable, str(repo / "scripts/botryllus_shazam/render_search_overview.py")],
    check=True,
)
