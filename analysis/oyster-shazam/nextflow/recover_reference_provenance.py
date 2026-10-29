"""Recover reference build provenance from the preserved job archive, without rebuilding."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from artifacts import sha, write_new


def recover(archive_path, output, repository):
    with tarfile.open(archive_path) as archive:

        def read(name):
            member = archive.getmember("./" + name)
            if not member.isfile():
                raise ValueError("Expected regular archived file")
            return archive.extractfile(member).read()

        def digest(name):
            return hashlib.sha256(read(name)).hexdigest()

        def load(name):
            return json.loads(read(name))

        pilot = load("public-pilot/corrected-public-pilot.json")
        translation = load("public-pilot/translation/complete.json")
        sketches = load("public-pilot/sketches/validated.json")
        if sketches["protocol"] != "six-frame-standard-stop-only-X-preserved-min30-v1":
            raise ValueError("Wrong corrected protocol")
        if translation["protocol"] != sketches["protocol"]:
            raise ValueError("Translation protocol mismatch")
        if digest("public-pilot/reference.fa") != translation["source_sha256"]:
            raise ValueError("Reference DNA changed")
        for name, expected in translation["outputs"].items():
            if digest("public-pilot/translation/" + name) != expected:
                raise ValueError("Translation changed")
        if (
            digest("public-pilot/translation/complete.json")
            != sketches["translation_manifest_sha256"]
        ):
            raise ValueError("Translation manifest changed")
        for row in sketches["collections"]:
            if digest("public-pilot/sketches/" + row["file"]) != row["sha256"]:
                raise ValueError("Sketch changed")
        for i, source in enumerate(pilot["sources"]):
            raw = read(f"public-pilot/reads-{i}.fq.gz")
            if (
                hashlib.md5(raw).hexdigest() != source["md5"]
                or hashlib.sha256(raw).hexdigest() != source["raw_sha256"]
            ):
                raise ValueError("Raw input changed")
            if (
                digest(f"public-pilot/reads-{i}_trimmed.fq.gz")
                != source["trimmed_sha256"]
            ):
                raise ValueError("Trimmed input changed")
            log = read(f"public-pilot/trim-{i}.log").decode()
            for expected in (
                "Trim Galore version: 0.6.10",
                "Cutadapt version: 5.2",
                "Quality Phred score cutoff: 20",
                "20 bp",
            ):
                if expected not in log:
                    raise ValueError("Missing preprocessing evidence")
        names = [
            "peptide_protocol.py",
            "sketch_corrected_peptides.py",
            "nextflow/corrected_public_pilot.py",
            "nextflow/migration-recovery.sbatch",
        ]
        code = {}
        for name in names:
            actual = read("code/" + name)
            committed = subprocess.check_output(
                ["git", "show", "8d7efccb:analysis/oyster-shazam/" + name],
                cwd=repository,
            )
            if actual != committed:
                raise ValueError("Archived source differs from recorded build commit")
            code[name] = hashlib.sha256(actual).hexdigest()
        version_files = [
            m.name[2:]
            for m in archive.getmembers()
            if m.name.startswith("./migration/work/")
            and m.name.endswith("/versions.json")
        ]
        versions = [load(name) for name in version_files]
        if not versions or any(
            v != {"sourmash": "4.9.4", "sourmash_plugin_branchwater": "0.9.15"}
            for v in versions
        ):
            raise ValueError("Missing or conflicting same-job environment evidence")
        write_new(
            output,
            dict(
                artifact_scope="Corrected SRR5945967 reference only; historical reference builds remain unrecorded",
                provenance_scope="recovered-from-preserved-build-evidence",
                job_id=49521583,
                build_commit="8d7efccb",
                archive_sha256=sha(archive_path),
                protocol=sketches["protocol"],
                source_code_sha256=code,
                sources=pilot["sources"],
                translation=translation,
                sketches=sketches,
                tools=dict(trim_galore="0.6.10", cutadapt="5.2", **versions[0]),
                tool_version_evidence="Trim logs and later checks in the same job and explicit Python/Sourmash environment; Biopython version was not recorded",
                version_files={name: digest(name) for name in version_files},
                production_rollout=False,
            ),
        )


if __name__ == "__main__":
    recover(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
