"""Build and search actual per-run sourmash indexes in four encodings."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
import sourmash

SETTINGS = {"dna": 31, "protein": 7, "dayhoff": 12, "hp": 30}


def main(run, reads, output):
    if importlib.metadata.version("sourmash") != "4.9.4":
        raise ValueError("This execution protocol requires sourmash 4.9.4")
    sources = [
        dict(
            path=str(p.resolve()),
            sha256=hashlib.file_digest(p.open("rb"), "sha256").hexdigest(),
        )
        for p in reads
    ]
    marker = output / "validated.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if old["sources"] != sources:
            raise ValueError("Input sources changed")
        for record in old["settings"]:
            if (
                hashlib.file_digest(
                    Path(record["path"]).open("rb"), "sha256"
                ).hexdigest()
                != record["sha256"]
            ):
                raise ValueError("Completed signature changed")
        print("Reused validated signatures:", run)
        return
    output.mkdir(parents=True, exist_ok=False)
    commands = []
    results = []

    def command(args, label):
        start = time.monotonic()
        with (output / f"{label}.log").open("w") as log:
            subprocess.run(
                ["/usr/bin/time", "-v", *args],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        commands.append(
            dict(argv=args, wall_seconds=time.monotonic() - start, log=f"{label}.log")
        )

    for mode, k in SETTINGS.items():
        folder = output / mode
        folder.mkdir()
        base = folder / "scaled5.sig.zip"
        command(
            [
                sys.executable,
                "-m",
                "sourmash",
                "sketch",
                "dna" if mode == "dna" else "translate",
                *map(str, reads),
                "--merge",
                run,
                "-p",
                f"{mode},k={k},scaled=5",
                "-o",
                str(base),
            ],
            f"{mode}-sketch",
        )
        for scale in [10, 100]:
            target = folder / f"scaled{scale}.sig.zip"
            command(
                [
                    sys.executable,
                    "-m",
                    "sourmash",
                    "sig",
                    "downsample",
                    "--scaled",
                    str(scale),
                    str(base),
                    "-o",
                    str(target),
                ],
                f"{mode}-downsample-{scale}",
            )
        previous = None
        for scale in [5, 10, 100]:
            target = folder / f"scaled{scale}.sig.zip"
            sigs = list(sourmash.load_file_as_signatures(str(target)))
            assert len(sigs) == 1
            sig = sigs[0]
            mh = sig.minhash
            assert mh.moltype.lower() == mode and mh.ksize == k and mh.scaled == scale
            if previous is not None:
                assert set(mh.hashes) <= previous
            previous = set(mh.hashes)
            index = folder / f"scaled{scale}.sbt.zip"
            command(
                [
                    sys.executable,
                    "-m",
                    "sourmash",
                    "index",
                    "-F",
                    "SBT",
                    "-k",
                    str(k),
                    f"--{mode}",
                    "--scaled",
                    str(scale),
                    str(index),
                    str(target),
                ],
                f"{mode}-index-{scale}",
            )
            # A nonempty self query must be found at full containment.
            if len(mh):
                command(
                    [
                        sys.executable,
                        "-m",
                        "sourmash",
                        "search",
                        str(target),
                        str(index),
                        "--containment",
                        "--threshold",
                        "1",
                        "-k",
                        str(k),
                        f"--{mode}",
                        "--output",
                        str(folder / f"scaled{scale}.self-search.csv"),
                    ],
                    f"{mode}-self-search-{scale}",
                )
                import csv

                hits = list(
                    csv.DictReader((folder / f"scaled{scale}.self-search.csv").open())
                )
                assert len(hits) == 1 and hits[0]["name"] == run
            results.append(
                dict(
                    mode=mode,
                    ksize=k,
                    scaled=scale,
                    hashes=len(mh),
                    empty=len(mh) == 0,
                    bytes=target.stat().st_size,
                    path=str(target),
                    sha256=hashlib.file_digest(target.open("rb"), "sha256").hexdigest(),
                    index=str(index),
                    index_bytes=index.stat().st_size,
                )
            )
        del previous, sigs, mh, sig
    report = dict(
        run=run,
        sources=sources,
        settings=results,
        commands=commands,
        input_scope="complete raw public run",
        translation="upstream sourmash six-frame translated reads; not predicted proteins",
        searched_runs=1,
        self_study_match=True,
        biological_replication="not inferred from runs",
    )
    temp = marker.with_suffix(".json.partial")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.rename(marker)
    print("Validated four encodings and three scales:", run, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--reads", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    main(a.run, a.reads, a.output)
