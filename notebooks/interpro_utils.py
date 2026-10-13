"""Pfam domain coordinates for UniProt proteins, from the InterPro API.

Each protein's answer is saved as one JSON file under a cache directory, so a
second run reads from disk and asks InterPro only for proteins it has not seen.
A protein with no Pfam domain (InterPro answers HTTP 204) is saved as an empty
answer too, so it is not asked for again. A request that fails after three tries
is not saved, and is tried again on the next run.

For Pfam coordinates of whole QfO proteomes, use the bulk file that
fetch_pfam_envelope_coords.py filters instead; this module is for a few hundred
proteins at a time.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import polars as pl

PFAM_BY_UNIPROT = (
    "https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{}/?page_size=200"
)
DEFAULT_CACHE = Path.home() / ".cache" / "kmerseek-analysis" / "interpro" / "pfam"


def _fetch_json(url: str, tries: int = 3) -> dict | None:
    """The API's JSON answer, {"results": []} for HTTP 204, or None on failure."""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=45) as fh:
                body = fh.read().decode()
            # urlopen does not raise on 204; it returns an empty body.
            return json.loads(body) if body.strip() else {"results": []}
        except urllib.error.HTTPError as e:
            if e.code == 204:
                return {"results": []}
        except Exception:
            pass
        if attempt < tries - 1:
            time.sleep(2)
    return None


def fetch_pfam_domains(
    accessions: dict[str, str],
    cache_dir: Path = DEFAULT_CACHE,
    pause: float = 0.12,
) -> pl.DataFrame:
    """Pfam domains for each protein, one row per domain interval.

    accessions maps a label (e.g. a gene symbol) to a UniProt accession.
    Coordinates are 1-based and inclusive, as InterPro reports them.
    Columns: symbol, accession, pfam_id, uniprot_length, domain_start, domain_end.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows, failed, n_asked = [], [], 0
    for symbol, acc in accessions.items():
        path = cache_dir / f"{acc}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            payload = _fetch_json(PFAM_BY_UNIPROT.format(acc))
            n_asked += 1
            time.sleep(pause)
            if payload is None:
                failed.append(symbol)
                continue
            path.write_text(json.dumps(payload))
        for res in payload.get("results", []):
            pfam = res["metadata"]["accession"]
            for prot in res.get("proteins", []):
                for loc in prot.get("entry_protein_locations", []):
                    for frag in loc.get("fragments", []):
                        rows.append(
                            {
                                "symbol": symbol,
                                "accession": acc,
                                "pfam_id": pfam,
                                "uniprot_length": prot.get("protein_length"),
                                "domain_start": frag["start"],
                                "domain_end": frag["end"],
                            }
                        )
    print(
        f"  InterPro: {len(accessions) - n_asked} of {len(accessions)} from the cache "
        f"in {cache_dir}, {n_asked} asked"
    )
    if failed:
        print(f"  no Pfam answer for {len(failed)}: {failed}")
    return pl.DataFrame(rows)
