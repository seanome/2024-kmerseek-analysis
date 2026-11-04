"""Get one sequence per probe from a GEO platform table."""

import http.client
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from geo2kmers.geo import Geo2KmersError, read_table

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
ACCESSION_COLUMNS = ["GB_ACC", "GB_LIST", "GENBANK", "GENBANK_ACC", "GENBANK_ACCESSION", "ACCESSION"]
ACCESSION_RE = re.compile(r"[A-Z]{1,6}_?\d{3,}(\.\d+)?")


MAX_GENBANK_NT = 5000


def find_column(columns, names):
    """Return the first column whose upper-case name is in names, in the order of names."""
    upper = {c.upper(): c for c in columns}
    for name in names:
        if name in upper:
            return upper[name]
    return None


def first_accession(text):
    """The first GenBank accession in a cell, without its version. Empty string if none."""
    for part in re.split(r"[,;\s]+", (text or "").strip()):
        if ACCESSION_RE.fullmatch(part):
            return part.split(".")[0]
    return ""


def affymetrix_package(meta):
    """Bioconductor probe package for an Affymetrix platform, or None for other makers."""
    title = " ".join(meta.get("Platform_title", []))
    maker = " ".join(meta.get("Platform_manufacturer", []))
    if "affymetrix" not in f"{title} {maker}".lower():
        return None
    match = re.search(r"\[([^\]]+)\]", title)
    if not match:
        return "the probe package for this array (https://www.bioconductor.org/packages/release/BiocViews.html#___ProbeData)"
    return re.sub(r"[^a-z0-9]", "", match.group(1).lower().replace("_at", "")) + "probe"


def post_efetch(ids, email, api_key=None):
    """Ask NCBI for FASTA text for a list of accessions."""
    form = {"db": "nuccore", "rettype": "fasta", "retmode": "text", "id": ",".join(ids)}
    form.update(tool="geo2kmers", email=email)
    if api_key:
        form["api_key"] = api_key
    request = urllib.request.Request(EFETCH_URL, data=urllib.parse.urlencode(form).encode())
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode()


def parse_fasta(text):
    """Return {accession without version: sequence} from NCBI FASTA text."""
    records, name = {}, None
    for line in text.splitlines():
        if line.startswith(">"):
            word = line[1:].split()[0]
            if "|" in word:  # old style: gi|123|emb|AJ516347.1|
                word = [p for p in word.split("|") if ACCESSION_RE.fullmatch(p)][-1]
            name = word.split(".")[0]
            records[name] = []
        elif name:
            records[name].append(line.strip())
    return {k: "".join(v).upper() for k, v in records.items()}


def fetch_accessions(accessions, email, api_key=None, post=post_efetch, sleep=time.sleep, batch_size=BATCH_SIZE):
    """Fetch sequences in batches, keeping to NCBI's limit of 3 requests a second (10 with a key).

    A batch that NCBI rejects is split in two, so one bad accession loses only itself.
    """
    pause = 0.11 if api_key else 0.34
    found = {}
    queue = [accessions[i : i + batch_size] for i in range(0, len(accessions), batch_size)]
    failures = 0
    while queue:
        batch = queue.pop(0)
        try:
            found.update(parse_fasta(post(batch, email, api_key)))
            failures = 0
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:  # NCBI is busy: try the same batch again
                failures += 1
                if failures > 5:
                    raise Geo2KmersError(f"NCBI kept refusing requests: {e}") from e
                queue.insert(0, batch)
                sleep(2**failures)
            elif len(batch) > 1:
                middle = len(batch) // 2
                queue[:0] = [batch[:middle], batch[middle:]]
        except (OSError, http.client.HTTPException) as e:  # no answer at all: try the same batch again
            failures += 1
            if failures > 5:
                raise Geo2KmersError(f"could not reach NCBI: {e}") from e
            queue.insert(0, batch)
            sleep(2**failures)
        sleep(pause)
    return found


def make_probe_fasta(
    gpl_table, out_fasta, source_table, cache_fasta=None, post=post_efetch, max_genbank_nt=MAX_GENBANK_NT
):
    """Write out_fasta and source_table. Returns counts per source."""
    meta, columns, rows = read_table(gpl_table, "platform")
    seq_col = find_column(columns, ["SEQUENCE"])
    acc_col = find_column(columns, ACCESSION_COLUMNS)
    package = affymetrix_package(meta) if seq_col is None else None
    if package:
        # a GenBank record on an Affymetrix table is the gene the probes target, not the 25-letter probes
        raise Geo2KmersError(
            f"{gpl_table} is an Affymetrix platform with no SEQUENCE column. "
            f"The 25-letter probe sequences are in a Bioconductor probe package, probably {package}. "
            "That name is worked out from the platform title and not checked, so look it up at "
            "https://bioconductor.org/packages/release/data/annotation/. This tool will not guess the sequences."
        )
    if seq_col is None and acc_col is None:
        raise Geo2KmersError(
            f"{gpl_table} has no SEQUENCE column and no GenBank column. Its columns are: {', '.join(columns)}"
        )

    fetched = {}
    if cache_fasta and os.path.exists(cache_fasta):
        with open(cache_fasta) as f:
            fetched = parse_fasta(f.read())
    wanted = []
    for row in rows:
        has_sequence = seq_col and row.get(seq_col, "").strip()
        accession = first_accession(row.get(acc_col)) if acc_col else ""
        if not has_sequence and accession and accession not in fetched:
            wanted.append(accession)
    wanted = sorted(set(wanted))
    if wanted:
        email = os.environ.get("NCBI_EMAIL")
        if not email:
            raise Geo2KmersError(
                f"{len(wanted)} sequences must come from NCBI, which asks for a contact address. Set NCBI_EMAIL."
            )
        new = fetch_accessions(wanted, email, os.environ.get("NCBI_API_KEY"), post=post)
        fetched.update(new)
        if cache_fasta:
            with open(cache_fasta, "a") as f:
                f.writelines(f">{name}\n{sequence}\n" for name, sequence in new.items())

    counts = {"gpl_sequence": 0, "genbank_fetch": 0, "record_too_long": 0, "missing": 0}
    with open(out_fasta, "w") as fasta, open(source_table, "w") as table:
        table.write("probe_id\tsource\tlength_nt\n")
        for row in rows:
            probe = row[columns[0]]
            sequence = re.sub(r"\s", "", row.get(seq_col, "")).upper() if seq_col else ""
            source = "gpl_sequence"
            if not sequence:
                accession = first_accession(row.get(acc_col)) if acc_col else ""
                sequence = fetched.get(accession, "")
                source = "genbank_fetch"
            if not sequence:
                source = "missing"
            elif source == "genbank_fetch" and len(sequence) > max_genbank_nt:
                # a record this long is a genome or chromosome, not the clone on the slide
                source = "record_too_long"
            else:
                fasta.write(f">{probe}\n{sequence}\n")
            counts[source] += 1
            table.write(f"{probe}\t{source}\t{len(sequence)}\n")
    return counts
