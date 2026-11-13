#!/usr/bin/env python3
"""Download a UniProt search result as TSV, and fail if it comes back short.

UniProt's /stream endpoint returned 7_518 of 7_922 toxin entries and 1_986 of 3_916
query-taxon entries with HTTP 200 and no error at all, so finishing is not evidence a
stream is complete. This pages through /search following the cursor in the Link header and
checks the row count against the X-Total-Results the server states on the first page.

Usage: fetch_uniprot.py <query> <fields> <output.tsv>
"""

import sys
import time
import urllib.parse
import urllib.request

PAGE_SIZE = 500
MAX_TRIES = 5


def get(url):
    """One GET, retried on a transport error, returning (body, headers)."""
    for attempt in range(1, MAX_TRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read().decode("utf-8"), response.headers
        except Exception as error:  # noqa: BLE001 - any failure here is worth one retry
            if attempt == MAX_TRIES:
                raise
            print(f"  attempt {attempt} failed ({error}), retrying", file=sys.stderr)
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def next_url(headers):
    """The cursor URL from the Link header, or None on the last page."""
    link = headers.get("Link")
    if not link or 'rel="next"' not in link:
        return None
    return link[link.index("<") + 1 : link.index(">")]


def main(query, fields, out_path):
    base = "https://rest.uniprot.org/uniprotkb/search"
    params = urllib.parse.urlencode(
        {"query": query, "format": "tsv", "fields": fields, "size": PAGE_SIZE}
    )
    url = f"{base}?{params}"

    header_line = None
    rows = []
    expected = None
    page = 0
    while url:
        body, headers = get(url)
        if expected is None:
            expected = int(headers["X-Total-Results"])
            print(f"expecting {expected} rows", file=sys.stderr)
        lines = body.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if header_line is None:
            header_line = lines[0]
        assert lines[0] == header_line, "column header changed between pages"
        rows.extend(lines[1:])
        page += 1
        print(
            f"  page {page}: {len(lines) - 1} rows, {len(rows)} so far", file=sys.stderr
        )
        url = next_url(headers)

    # UniProt does not quote fields, so one line is one entry and the two counts are
    # comparable. A short result means the download is incomplete, not that the query
    # changed, so it fails rather than writing a file that looks finished.
    assert len(rows) == expected, f"got {len(rows)} rows, server said {expected}"
    with open(out_path, "w") as out:
        out.write(header_line + "\n")
        out.write("\n".join(rows) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
