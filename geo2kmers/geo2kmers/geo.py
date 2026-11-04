"""Download GEO files and read the three text formats they come in."""

import gzip
import hashlib
import json
import re
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ACC_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
FTP_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series"
# GEO asks for no more than a few requests a second
PAUSE_SECONDS = 0.4


class Geo2KmersError(Exception):
    """A problem the user must fix. The command line prints the message and stops."""


def open_text(path):
    """Open a plain or gzipped text file for reading."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, path, opener=urllib.request.urlopen):
    """Save url to path, unless path is already on disk. Returns True if it downloaded."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return False
    part = path.with_name(path.name + ".part")
    with opener(url, timeout=120) as response, open(part, "wb") as out:
        out.writelines(iter(lambda: response.read(1 << 20), b""))
    part.replace(path)
    time.sleep(PAUSE_SECONDS)
    return True


def series_dir_url(gse):
    """GEO's folder for a series: GSE2183 is under GSE2nnn, GSE120862 under GSE120nnn."""
    number = gse[3:]
    stem = number[:-3] if len(number) > 3 else ""
    return f"{FTP_URL}/GSE{stem}nnn/{gse}"


def acc_url(accession):
    return f"{ACC_URL}?acc={accession}&targ=self&form=text&view=full"


def read_series_matrix(path):
    """Return (metadata, sample_ids, rows). rows maps probe ID to a list of strings."""
    meta, samples, rows = {}, [], {}
    in_table = False
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                fields = [x.strip('"') for x in line.split("\t")]
                if fields[0] == "ID_REF":
                    samples = fields[1:]
                else:
                    rows[fields[0]] = fields[1:]
            elif line.startswith("!"):
                key, *values = line.split("\t")
                meta.setdefault(key[1:], []).append([v.strip('"') for v in values])
    return meta, samples, rows


def read_table(path, begin):
    """Read a GEO SOFT text file. Returns (metadata, columns, rows of dicts).

    begin is "platform" for a GPL table and "sample" for a GSM table.
    """
    meta, columns, rows = {}, None, []
    in_table = False
    with open_text(path) as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line.startswith(f"!{begin}_table_begin"):
                in_table = True
            elif line.startswith(f"!{begin}_table_end"):
                in_table = False
            elif in_table:
                fields = line.split("\t")
                if columns is None:
                    columns = fields
                else:
                    rows.append(dict(zip(columns, fields)))
            elif line.startswith("!") and " = " in line:
                key, value = line[1:].split(" = ", 1)
                meta.setdefault(key, []).append(value)
    if columns is None:
        raise Geo2KmersError(f"{path} holds no {begin} table")
    return meta, columns, rows


def fetch(gse, outdir, gpl=None, sample_tables=False, opener=urllib.request.urlopen):
    """Download what a series needs into outdir and write provenance.json and samples.tsv."""
    if not re.fullmatch(r"GSE\d+", gse):
        raise Geo2KmersError(f"{gse} is not a GEO series accession (GSE followed by digits)")
    outdir = Path(outdir)
    (outdir / "gsm").mkdir(parents=True, exist_ok=True)

    brief = outdir / f"{gse}.txt"
    url_of = {brief: acc_url(gse)}
    download(url_of[brief], brief, opener)
    series = {}
    with open_text(brief) as f:
        for line in f:
            if line.startswith("!Series_") and " = " in line:
                key, value = line.rstrip("\r\n")[1:].split(" = ", 1)
                series.setdefault(key, []).append(value)
    platforms = series.get("Series_platform_id", [])
    gsms = series.get("Series_sample_id", [])
    if not platforms or not gsms:
        raise Geo2KmersError(f"the GEO record for {gse} lists no platform or no samples")
    if gpl is None:
        if len(platforms) > 1:
            raise Geo2KmersError(
                f"{gse} uses {len(platforms)} platforms ({', '.join(platforms)}); choose one with --gpl"
            )
        gpl = platforms[0]
    elif gpl not in platforms:
        raise Geo2KmersError(f"{gse} does not use {gpl}; it uses {', '.join(platforms)}")

    gpl_path = outdir / f"{gpl}.txt"
    url_of[gpl_path] = acc_url(gpl)
    download(url_of[gpl_path], gpl_path, opener)

    name = f"{gse}_series_matrix.txt.gz" if len(platforms) == 1 else f"{gse}-{gpl}_series_matrix.txt.gz"
    matrix_path = outdir / name
    url_of[matrix_path] = f"{series_dir_url(gse)}/matrix/{name}"
    have_matrix = True
    try:
        download(url_of[matrix_path], matrix_path, opener)
    except OSError as e:
        if getattr(e, "code", None) != 404:  # a busy server is not a missing file
            raise Geo2KmersError(f"could not download {url_of[matrix_path]}: {e}") from e
        print(f"no series matrix for {gse} ({e}); using the sample tables")
        del url_of[matrix_path]
        have_matrix = False

    channel_counts = {}
    if have_matrix:
        meta, samples, _ = read_series_matrix(matrix_path)
        counts = meta.get("Sample_channel_count", [[]])[0]
        channel_counts = dict(zip(samples, counts))
        gsms = samples or gsms
    # A two-colour series matrix holds ratios, so the intensities are only in the sample tables
    need_tables = sample_tables or not have_matrix or any(v != "1" for v in channel_counts.values())
    if need_tables:
        for gsm in gsms:
            path = outdir / "gsm" / f"{gsm}.txt"
            url_of[path] = acc_url(gsm)
            download(url_of[path], path, opener)
            sample_meta, _, _ = read_table(path, "sample")
            channel_counts[gsm] = sample_meta.get("Sample_channel_count", ["1"])[0]

    with open(outdir / "samples.tsv", "w") as f:
        f.write("gsm\tchannel_count\n")
        for gsm in gsms:
            f.write(f"{gsm}\t{channel_counts.get(gsm, '1')}\n")

    provenance = {
        "accession": gse,
        "platform": gpl,
        "download_date_utc": datetime.now(UTC).strftime("%Y-%m-%d"),
        "files": [
            {"file": str(p.relative_to(outdir)), "url": u, "sha256": sha256(p), "bytes": p.stat().st_size}
            for p, u in url_of.items()
        ],
    }
    with open(outdir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)
    return provenance
