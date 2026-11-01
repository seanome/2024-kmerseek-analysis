#!/usr/bin/env python3
"""Structural test of the claim that Borrelia burgdorferi P66 mimics human CD47.

The claim under test is quoted in notebooks/016-ced9-bcl2-and-p66-cd47.ipynb: a
predicted extracellular loop of P66 at mature residues 181-187, carrying D184 and
D186, is said to bind the anti-phagocytic receptor SIRPalpha the way CD47 does.

Nothing here runs kmerseek. Every accession, model version and PDB ID is resolved
against the live source at run time; none is written from memory. Outputs go to
OUTDIR as CSV, and the notebook (notebooks/276_*.ipynb) only reads them.

Steps
  1. Resolve the P66 UniProt accession by gene and taxon, and read its signal
     peptide so the paper's mature numbering can be mapped to the model's.
  2. Fetch the AlphaFold model at the version the API reports, and tabulate
     per-residue pLDDT over the loop and its flanks.
  3. Search RCSB for a CD47 ectodomain / SIRPalpha complex, pick the best
     resolution one with both chains wild type, and fetch it.
  4. Define CD47's SIRPalpha-binding face as the CD47 residues with a heavy atom
     within CONTACT_ANGSTROM of a SIRPalpha heavy atom.
  5. Comparison 1, whole fold: USalign P66 against CD47's IgV domain, with 20
     unrelated SCOPe b.1.1 immunoglobulin domains as the background.
  6. Comparison 2: Foldseek P66 against the PDB and against CD47's AlphaFold model.
  7. Comparison 3, face against loop: superpose the loop on every same-length
     window of the binding face, with 10 other P66 loops as the background.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_p66_cd47_structure.py
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl
from Bio.PDB import MMCIFParser, PDBParser, Superimposer
from Bio.PDB.MMCIFParser import MMCIFParser as _MMCIFParser  # noqa: F401
from Bio.PDB.NeighborSearch import NeighborSearch
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Polypeptide import is_aa

# ---------------------------------------------------------------------------
# Constants. Every one of these is either a published value with its source, or
# a parameter of this analysis that the notebook quotes.
# ---------------------------------------------------------------------------

#: The loop named in the claim, in the paper's mature-chain numbering.
LOOP_MATURE_START, LOOP_MATURE_END = 181, 187
#: The two aspartates the paper calls critical, in mature numbering.
LOOP_CRITICAL_MATURE = (184, 186)
#: Gene and NCBI taxon used to resolve P66 without naming its accession.
P66_GENE, P66_TAXON = "p66", 139  # 139 = Borreliella burgdorferi
#: Human CD47, resolved by gene and taxon the same way.
CD47_GENE, CD47_TAXON = "CD47", 9606
#: Heavy-atom distance that defines a contact. 4.5 A is the cutoff the task set.
CONTACT_ANGSTROM = 4.5
#: SCOPe 2.08 superfamily for immunoglobulin-like beta-sandwich domains.
SCOPE_IG_SUPERFAMILY = "b.1.1"
#: Background sizes, from the task.
N_IG_BACKGROUND, N_LOOP_BACKGROUND = 20, 10
#: Seed for every random draw, so the background sets are reproducible.
SEED = 276

OUTDIR = Path("/Users/olga/data/nb276-p66-cd47-structure")
SCOPE_ASTRAL40 = Path(
    "/Users/olga/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
)
SCOPE_PDBSTYLE = Path("/Users/olga/data/scope/pdbstyle-2.08/pdbstyle-2.08")
USALIGN = Path("/Users/olga/anaconda3/envs/usalign/bin/USalign")
FOLDSEEK = Path("/Users/olga/anaconda3/envs/foldseek-10.941cd33/bin/foldseek")
FOLDSEEK_PDB_DB = Path("/Users/olga/data/foldseek-dbs/pdb")


def _get(url: str, cache: Path, binary: bool = False) -> str | bytes:
    """Fetch url once and reuse the cached copy on later runs.

    A cached file that is HTML when the caller wanted data is a server outage
    served as 200, so refuse it rather than parse it.
    """
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as fh:
            cache.write_bytes(fh.read())
    raw = cache.read_bytes()
    if not binary and raw.lstrip()[:5].lower() in (b"<html", b"<!doc"):
        raise RuntimeError(f"{cache} holds an HTML page, not data. Delete it and retry.")
    return raw if binary else raw.decode()


# ---------------------------------------------------------------------------
# Step 1: resolve accessions and the mature-numbering offset
# ---------------------------------------------------------------------------


def resolve_uniprot(gene: str, taxon: int, cache: Path) -> dict:
    """Return the canonical UniProt entry for a gene in a taxon, with its features.

    A reviewed (Swiss-Prot) entry wins over any unreviewed one, because a human
    gene has many TrEMBL isoform entries and the longest of those is not the
    canonical sequence. Among equally reviewed entries the longest wins. The
    paper's numbering is of the mature chain, so the signal peptide length is
    read from the entry rather than assumed.
    """
    url = (
        "https://rest.uniprot.org/uniprotkb/search?"
        f"query=gene:{gene}+AND+taxonomy_id:{taxon}"
        "&fields=accession,id,protein_name,length,sequence,ft_signal,ft_chain,reviewed"
        "&format=json&size=50"
    )
    hits = json.loads(_get(url, cache))["results"]
    if not hits:
        raise RuntimeError(f"UniProt returned no entry for gene {gene} taxon {taxon}")
    best = max(
        hits,
        key=lambda h: (
            h.get("entryType", "").startswith("UniProtKB reviewed"),
            h["sequence"]["length"],
        ),
    )
    signal_end = 0
    for feat in best.get("features", []):
        if feat["type"] == "Signal":
            signal_end = feat["location"]["end"]["value"]
    return {
        "accession": best["primaryAccession"],
        "entry_name": best["uniProtkbId"],
        "protein_name": best["proteinDescription"]["recommendedName"]["fullName"]["value"]
        if "recommendedName" in best["proteinDescription"]
        else best["proteinDescription"]["submissionNames"][0]["fullName"]["value"],
        "length": best["sequence"]["length"],
        "sequence": best["sequence"]["value"],
        "signal_peptide_end": signal_end,
        "reviewed": best.get("entryType", ""),
    }


def check_loop_numbering(p66: dict) -> dict:
    """Map the paper's mature loop numbering onto the full-length sequence.

    Returns the full-length span and the residues there. Raises if the two
    aspartates the paper names are not where the mapping puts them, because then
    the numbering in the claim is not the numbering of this entry.
    """
    offset = p66["signal_peptide_end"]  # mature position p == full-length p + offset
    full_start = LOOP_MATURE_START + offset
    full_end = LOOP_MATURE_END + offset
    seq = p66["sequence"]
    residues = seq[full_start - 1 : full_end]
    for mature_pos in LOOP_CRITICAL_MATURE:
        full_pos = mature_pos + offset
        if seq[full_pos - 1] != "D":
            raise RuntimeError(
                f"Mature D{mature_pos} maps to full-length {full_pos}, which is "
                f"{seq[full_pos - 1]}, not D. The claim's numbering does not fit "
                f"{p66['accession']}."
            )
    return {
        "mature_offset": offset,
        "full_start": full_start,
        "full_end": full_end,
        "residues": residues,
    }


# ---------------------------------------------------------------------------
# Step 2: the AlphaFold model and its confidence over the loop
# ---------------------------------------------------------------------------


def fetch_alphafold(accession: str, outdir: Path) -> dict:
    """Download the AlphaFold model at whatever version the API reports.

    The version is never written from memory: AFDB has served v4 and v6 for the
    same accession, and a hard-coded URL silently 404s or serves the wrong model.
    """
    meta = json.loads(
        _get(
            f"https://alphafold.ebi.ac.uk/api/prediction/{accession}",
            outdir / f"afdb_{accession}.json",
        )
    )[0]
    pdb_path = outdir / f"{meta['entryId']}-model_v{meta['latestVersion']}.pdb"
    _get(meta["pdbUrl"], pdb_path, binary=True)
    return {
        "entry_id": meta["entryId"],
        "mean_plddt": meta.get("globalMetricValue"),
        "version": meta["latestVersion"],
        "all_versions": meta["allVersions"],
        "model_created": meta["modelCreatedDate"][:10],
        "uniprot_start": meta["uniprotStart"],
        "uniprot_end": meta["uniprotEnd"],
        "pdb_url": meta["pdbUrl"],
        "pdb_path": pdb_path,
    }


def plddt_per_residue(pdb_path: Path) -> pl.DataFrame:
    """Per-residue pLDDT, read from the B-factor column of an AlphaFold model."""
    chain = next(iter(PDBParser(QUIET=True).get_structure("af", pdb_path)[0]))
    rows = [
        {
            "full_position": res.id[1],
            "residue": res.get_resname(),
            "plddt": float(np.mean([a.get_bfactor() for a in res])),
        }
        for res in chain
        if is_aa(res, standard=True)
    ]
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 3 and 4: the CD47 / SIRPalpha complex and CD47's binding face
# ---------------------------------------------------------------------------


def search_rcsb_cd47_sirpa(outdir: Path) -> pl.DataFrame:
    """Search RCSB for CD47 / SIRPalpha complexes and describe each hit.

    The PDB ID is never typed from memory. Entries whose own description calls a
    chain engineered, high-affinity or a variant are flagged, because a mutant
    interface is not the wild-type face the claim is about.
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "full_text",
                    "parameters": {"value": "CD47 SIRP alpha"},
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater_or_equal",
                        "value": 2,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 50},
            "results_verbosity": "compact",
        },
    }
    cache = outdir / "rcsb_search.json"
    if not cache.exists():
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as fh:
            cache.write_bytes(fh.read())
    ids = json.loads(cache.read_text())["result_set"]

    engineered = re.compile(r"engineer|high-affinity|variant|FD6|N80A|mutant", re.I)
    rows = []
    for pdb_id in ids:
        entry = json.loads(
            _get(
                f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}",
                outdir / f"rcsb_entry_{pdb_id}.json",
            )
        )
        info = entry.get("rcsb_entry_info", {})
        n_prot = info.get("polymer_entity_count_protein", 0)
        chains = {}
        for i in range(1, n_prot + 1):
            ent = json.loads(
                _get(
                    f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{i}",
                    outdir / f"rcsb_entity_{pdb_id}_{i}.json",
                )
            )
            desc = ent["rcsb_polymer_entity"].get("pdbx_description", "")
            for auth in ent["rcsb_polymer_entity_container_identifiers"].get(
                "auth_asym_ids", []
            ):
                chains[auth] = desc
        descs = " | ".join(sorted(set(chains.values())))
        has_cd47 = bool(re.search(r"CD47", descs, re.I))
        has_sirpa = bool(
            re.search(r"substrate 1|SIRPA|SIRP-alpha|SIRPa\b", descs, re.I)
        )
        rows.append(
            {
                "pdb_id": pdb_id,
                "title": entry["struct"]["title"],
                "method": ", ".join(info.get("experimental_method", []))
                if isinstance(info.get("experimental_method"), list)
                else str(info.get("experimental_method")),
                "resolution_angstrom": (info.get("resolution_combined") or [None])[0],
                "chains": json.dumps(chains),
                "has_cd47": has_cd47,
                "has_sirpalpha": has_sirpa,
                "engineered_partner": bool(engineered.search(entry["struct"]["title"]))
                or bool(engineered.search(descs)),
            }
        )
    return pl.DataFrame(rows).sort("resolution_angstrom", nulls_last=True)


def search_rcsb_free_text(value: str, outdir: Path, tag: str) -> list[str]:
    """RCSB full-text search, returning the entry IDs. An empty body means zero.

    Used to check whether an experimental structure of P66 exists at all, which
    decides whether the AlphaFold model is a convenience or the only option.
    """
    query = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": value},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 25},
            "results_verbosity": "compact",
        },
    }
    cache = outdir / f"rcsb_freetext_{tag}.json"
    if not cache.exists():
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=json.dumps(query).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as fh:
                cache.write_bytes(fh.read())
        except urllib.error.HTTPError:
            cache.write_text("")  # RCSB answers a zero-hit search with no body
    raw = cache.read_text().strip()
    return json.loads(raw)["result_set"] if raw else []


def pick_complex(hits: pl.DataFrame) -> dict:
    """Best-resolution entry holding wild-type CD47 and wild-type SIRPalpha."""
    usable = hits.filter(
        pl.col("has_cd47") & pl.col("has_sirpalpha") & ~pl.col("engineered_partner")
    )
    if usable.is_empty():
        raise RuntimeError("No wild-type CD47/SIRPalpha complex in the RCSB hits.")
    row = usable.row(0, named=True)
    chains = json.loads(row["chains"])
    cd47 = [c for c, d in chains.items() if re.search(r"CD47", d, re.I)]
    sirpa = [c for c, d in chains.items() if c not in cd47]
    return {**row, "cd47_chains": sorted(cd47), "sirpalpha_chains": sorted(sirpa)}


def fetch_cif(pdb_id: str, outdir: Path) -> Path:
    path = outdir / f"{pdb_id}.cif"
    _get(f"https://files.rcsb.org/download/{pdb_id}.cif", path, binary=True)
    return path


def binding_face(
    cif_path: Path, cd47_chains: list[str], sirpa_chains: list[str]
) -> pl.DataFrame:
    """CD47 residues with a heavy atom within CONTACT_ANGSTROM of SIRPalpha.

    Every CD47 chain is paired against every SIRPalpha chain; the pair with the
    most contacts is the biological complex, and the others are crystal packing.
    """
    model = MMCIFParser(QUIET=True).get_structure("cx", cif_path)[0]

    def heavy(chain_id):
        return [
            a
            for res in model[chain_id]
            if is_aa(res, standard=True)
            for a in res
            if a.element != "H"
        ]

    best = None
    for cd47_id in cd47_chains:
        for sirpa_id in sirpa_chains:
            search = NeighborSearch(heavy(sirpa_id))
            found = {}
            for atom in heavy(cd47_id):
                near = search.search(atom.coord, CONTACT_ANGSTROM)
                if not near:
                    continue
                res = atom.get_parent()
                key = res.id[1]
                dist = min(float(np.linalg.norm(atom.coord - n.coord)) for n in near)
                prev = found.get(key)
                if prev is None or dist < prev["min_distance_angstrom"]:
                    found[key] = {
                        "cd47_chain": cd47_id,
                        "sirpalpha_chain": sirpa_id,
                        "cd47_position": key,
                        "residue": res.get_resname(),
                        "min_distance_angstrom": dist,
                    }
            if best is None or len(found) > len(best):
                best = found
    return pl.DataFrame(sorted(best.values(), key=lambda r: r["cd47_position"]))


def write_chain_pdb(cif_path: Path, chain_id: str, out_path: Path) -> dict:
    """Write one chain of a mmCIF out as PDB, and report what it contains."""
    model = MMCIFParser(QUIET=True).get_structure("s", cif_path)[0]
    chain = model[chain_id]
    for res in list(chain):
        if not is_aa(res, standard=True):
            chain.detach_child(res.id)
    io = PDBIO()
    io.set_structure(chain)
    io.save(str(out_path))
    positions = [r.id[1] for r in chain]
    return {
        "path": out_path,
        "chain": chain_id,
        "n_residues": len(positions),
        "first_position": min(positions),
        "last_position": max(positions),
    }


# ---------------------------------------------------------------------------
# Step 5: whole-fold comparison with USalign, and the immunoglobulin background
# ---------------------------------------------------------------------------


def usalign(query: Path, target: Path) -> dict:
    """TM-score, aligned length and RMSD from USalign.

    TM-score depends on which chain it is normalised by, so both are returned.
    Over a short domain against a long one only the short normalisation is
    interpretable, and the notebook says which it quotes.
    """
    out = subprocess.run(
        [str(USALIGN), str(query), str(target)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    scores = [float(m) for m in re.findall(r"TM-score= ([\d.]+)", out)]
    aligned = re.search(r"Aligned length= *(\d+), RMSD= *([\d.]+)", out)
    lengths = [int(m) for m in re.findall(r"Length of Structure_\d+: (\d+)", out)]
    return {
        "tm_score_norm_query": scores[0] if scores else None,
        "tm_score_norm_target": scores[1] if len(scores) > 1 else None,
        "aligned_length": int(aligned.group(1)) if aligned else None,
        "rmsd_angstrom": float(aligned.group(2)) if aligned else None,
        "query_length": lengths[0] if lengths else None,
        "target_length": lengths[1] if len(lengths) > 1 else None,
    }


def scope_ig_domains(exclude_pattern: str = r"CD47|signal.regulatory|substrate 1") -> pl.DataFrame:
    """SCOPe 2.08 ASTRAL-40 domains in the immunoglobulin-like superfamily.

    ASTRAL-40 is one domain per 40%-identity cluster, so these 20 are unrelated
    to each other by sequence as well as to CD47.
    """
    rows = []
    for line in SCOPE_ASTRAL40.read_text().splitlines():
        if not line.startswith(">"):
            continue
        match = re.match(r">(\S+)\s+(\S+)\s+\((\S+)\)\s+(.*)", line)
        if not match or not match.group(2).startswith(SCOPE_IG_SUPERFAMILY + "."):
            continue
        sid, sccs, _chain, desc = match.groups()
        if re.search(exclude_pattern, desc, re.I):
            continue
        ent = SCOPE_PDBSTYLE / sid[2:4] / f"{sid}.ent"
        if ent.exists():
            rows.append({"scope_id": sid, "sccs": sccs, "description": desc,
                         "path": str(ent)})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 7: the loop against the face
# ---------------------------------------------------------------------------


def ca_secondary_structure(pdb_path: Path) -> pl.DataFrame:
    """Helix / strand / coil per residue from CA geometry only (P-SEA criteria).

    DSSP is not installed on this machine, so secondary structure comes from the
    CA-trace distance and angle ranges of Labesse et al. 1997, which is what
    P-SEA uses. Only the coil assignment is used here, to find loops.
    """
    chain = next(iter(PDBParser(QUIET=True).get_structure("af", pdb_path)[0]))
    residues = [r for r in chain if is_aa(r, standard=True) and "CA" in r]
    coords = np.array([r["CA"].coord for r in residues])
    n = len(residues)

    def dist(i, j):
        return float(np.linalg.norm(coords[i] - coords[j])) if 0 <= j < n else np.nan

    labels = []
    for i in range(n):
        d2, d3, d4 = dist(i, i + 2), dist(i, i + 3), dist(i, i + 4)
        helix = (
            5.0 <= d2 <= 6.1 and 4.8 <= d3 <= 6.3 and 5.8 <= d4 <= 7.0
        )
        strand = (
            6.1 <= d2 <= 7.3 and 9.0 <= d3 <= 10.8 and 11.3 <= d4 <= 13.5
        )
        labels.append("helix" if helix else "strand" if strand else "coil")
    # A single residue of helix or strand inside a run of coil is noise.
    smoothed = list(labels)
    for i in range(1, n - 1):
        if labels[i] != labels[i - 1] and labels[i - 1] == labels[i + 1]:
            smoothed[i] = labels[i - 1]
    return pl.DataFrame(
        {
            "full_position": [r.id[1] for r in residues],
            "secondary_structure": smoothed,
        }
    )


def backbone_rmsd_loop_on_windows(
    loop_pdb: Path,
    loop_span: tuple[int, int],
    target_pdb: Path,
    window_positions: list[list[int]],
) -> pl.DataFrame:
    """Superpose a loop's backbone on each target window and report the RMSD.

    The binding face is not one contiguous stretch, so the loop is superposed on
    every contiguous window of the target that is the loop's length and overlaps
    the face. Backbone means N, CA, C and O, in sequence order.
    """
    atoms = ("N", "CA", "C", "O")

    def backbone(pdb_path, positions):
        chain = next(iter(PDBParser(QUIET=True).get_structure("s", pdb_path)[0]))
        by_pos = {r.id[1]: r for r in chain if is_aa(r, standard=True)}
        picked = []
        for pos in positions:
            res = by_pos.get(pos)
            if res is None or any(a not in res for a in atoms):
                return None
            picked.extend(res[a] for a in atoms)
        return picked

    loop_positions = list(range(loop_span[0], loop_span[1] + 1))
    fixed = backbone(loop_pdb, loop_positions)
    if fixed is None:
        raise RuntimeError(f"Loop {loop_span} is incomplete in {loop_pdb}")

    rows = []
    for positions in window_positions:
        moving = backbone(target_pdb, positions)
        if moving is None or len(moving) != len(fixed):
            continue
        sup = Superimposer()
        sup.set_atoms(fixed, moving)
        rows.append(
            {
                "window_start": positions[0],
                "window_end": positions[-1],
                "backbone_rmsd_angstrom": float(sup.rms),
                "n_backbone_atoms": len(fixed),
            }
        )
    return pl.DataFrame(rows).sort("backbone_rmsd_angstrom")


def hp_class_strings(loop_residues: str) -> pl.DataFrame:
    """The loop written out in each two-letter alphabet kmerseek uses.

    A stretch that encodes to one repeated class letter carries no two-letter
    pattern, so no exact k-mer in that alphabet can distinguish it from any other
    stretch of the same class. This is the mechanism behind the kmerseek negative
    in PR 96, and it is read off the alphabet definitions rather than asserted.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
    import hp_conservation_utils as hc

    rows = []
    for name in hc.HP2:
        clusters = hc.ALPHABET_CLUSTERS[name]
        letters = "".join(
            "H" if residue in clusters[0] else "P" for residue in loop_residues
        )
        rows.append(
            {
                "alphabet": name,
                "residues": loop_residues,
                "class_string": letters,
                "n_distinct_classes": len(set(letters)),
                "hydrophobic_cluster": clusters[0],
                "polar_cluster": clusters[1],
            }
        )
    return pl.DataFrame(rows).sort("alphabet")


#: Alphabets kmerseek offers that are not in the repo's hp_conservation_utils table.
#: Clusters copied from the alphabet table in seanome/kmerseek README.md, which cites
#: each partition's paper; class order is the order that table numbers them in.
EXTRA_ALPHABET_CLUSTERS: dict[str, list[str]] = {
    # Jain, Jain & Jain 2014
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    # Steinegger & Soding 2018, the Linclust partition
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    # Ieremie et al. 2024
    "wass14": ["WM", "DI", "P", "C", "AV", "K", "T", "RE", "G", "L", "Y", "SH", "F",
               "NQ"],
    # Prlic et al. 2000
    "hsdm17": ["A", "D", "KE", "R", "N", "T", "S", "Q", "Y", "F", "LIV", "M", "C", "W",
               "H", "G", "P"],
}

#: k values swept per alphabet when asking whether any loop k-mer reaches CD47.
#: 5 is the shortest kmerseek is run at anywhere in this project and 25 is past the
#: longest k at which P66 and CD47 share anything at all (PR 96 tops out at 21).
LOOP_REACH_KSIZES = range(5, 26)


def all_alphabet_clusters() -> dict[str, list[str]]:
    """Every alphabet in one table, the repo's own plus the four from the README."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
    import hp_conservation_utils as hc

    return {**hc.ALPHABET_CLUSTERS, **EXTRA_ALPHABET_CLUSTERS}


def loop_kmer_reach(p66_seq: str, cd47_seq: str, loop_span: tuple[int, int]) -> pl.DataFrame:
    """Does any k-mer covering the P66 loop occur in CD47, in any alphabet at any k?

    This is the question the two-letter class string cannot answer. In a two-letter
    alphabet the loop is one repeated letter, so nothing there is distinctive; in a
    larger alphabet it is a varied string, and then the only way to know whether a
    k-mer there could ever be shared is to encode both proteins and look.

    One row per alphabet and k: how many P66 k-mers cover the loop, how many of
    those encoded k-mers occur anywhere in CD47, and how distinct the loop's own
    encoding is.
    """
    start, end = loop_span

    def encode(seq, clusters):
        table = {}
        for index, cluster in enumerate(clusters):
            for residue in cluster:
                table[residue] = index
        # An unknown residue gets its own class so it can never match anything.
        return [table.get(r, -1 - i) for i, r in enumerate(seq)]

    rows = []
    for name, clusters in sorted(all_alphabet_clusters().items()):
        p66_code = encode(p66_seq, clusters)
        cd47_code = encode(cd47_seq, clusters)
        loop_code = tuple(p66_code[start - 1 : end])
        for k in LOOP_REACH_KSIZES:
            if k > len(cd47_code):
                continue
            cd47_kmers = {
                tuple(cd47_code[i : i + k]) for i in range(len(cd47_code) - k + 1)
            }
            # Every P66 k-mer that covers at least one residue of the loop.
            first = max(0, start - k)
            last = min(len(p66_code) - k, end - 1)
            covering = [
                tuple(p66_code[i : i + k]) for i in range(first, last + 1)
            ]
            rows.append(
                {
                    "alphabet": name,
                    "n_classes": len(clusters),
                    "ksize": k,
                    "bits_per_kmer": round(k * float(np.log2(len(clusters))), 2),
                    "n_p66_kmers_covering_loop": len(covering),
                    "n_covering_kmers_found_in_cd47": sum(
                        kmer in cd47_kmers for kmer in covering
                    ),
                    "loop_class_string": "".join(
                        "ABCDEFGHIJKLMNOPQRST"[c] if 0 <= c < 20 else "?"
                        for c in loop_code
                    ),
                    "n_distinct_classes_in_loop": len(set(loop_code)),
                }
            )
    return pl.DataFrame(rows)


def contiguous_windows(positions: list[int], length: int, must_touch: set[int]) -> list[list[int]]:
    """Every run of `length` consecutive positions that includes a face residue."""
    available = set(positions)
    windows = []
    for start in sorted(available):
        run = list(range(start, start + length))
        if all(p in available for p in run) and must_touch.intersection(run):
            windows.append(run)
    return windows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=OUTDIR)
    parser.add_argument(
        "--skip-foldseek",
        action="store_true",
        help="Skip step 6 when the Foldseek PDB database is still downloading.",
    )
    args = parser.parse_args()
    out = args.outdir
    (out / "cache").mkdir(parents=True, exist_ok=True)
    cache = out / "cache"

    # --- Step 1 ---------------------------------------------------------
    p66 = resolve_uniprot(P66_GENE, P66_TAXON, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot(CD47_GENE, CD47_TAXON, cache / "uniprot_cd47.json")
    loop = check_loop_numbering(p66)
    pl.DataFrame(
        [
            {k: v for k, v in p66.items() if k != "sequence"} | {"role": "P66 query"},
            {k: v for k, v in cd47.items() if k != "sequence"} | {"role": "CD47"},
        ]
    ).write_csv(out / "01_accessions.csv")
    pl.DataFrame([loop]).write_csv(out / "01_loop_numbering.csv")
    print(f"[1] P66 {p66['accession']} ({p66['length']} aa), signal peptide 1-"
          f"{p66['signal_peptide_end']}")
    print(f"    mature {LOOP_MATURE_START}-{LOOP_MATURE_END} = full-length "
          f"{loop['full_start']}-{loop['full_end']} = {loop['residues']}")
    print(f"[1] CD47 {cd47['accession']} ({cd47['length']} aa)")

    # --- Step 2 ---------------------------------------------------------
    af_p66 = fetch_alphafold(p66["accession"], cache)
    af_cd47 = fetch_alphafold(cd47["accession"], cache)
    plddt = plddt_per_residue(af_p66["pdb_path"])
    plddt.write_csv(out / "02_p66_plddt.csv")
    pl.DataFrame(
        [
            {k: str(v) for k, v in af_p66.items()} | {"protein": "P66"},
            {k: str(v) for k, v in af_cd47.items()} | {"protein": "CD47"},
        ]
    ).write_csv(out / "02_alphafold_models.csv")
    in_loop = plddt.filter(
        pl.col("full_position").is_between(loop["full_start"], loop["full_end"])
    )
    experimental_p66 = search_rcsb_free_text(
        "Borrelia burgdorferi P66 outer membrane protein", cache, "p66"
    )
    pl.DataFrame(
        [
            {
                "experimental_p66_entries": ", ".join(experimental_p66) or "none",
                "n_experimental_p66_entries": len(experimental_p66),
                "p66_model_mean_plddt": af_p66["mean_plddt"],
                "cd47_model_mean_plddt": af_cd47["mean_plddt"],
                "p66_fraction_below_50_plddt": float((plddt["plddt"] < 50).mean()),
            }
        ]
    ).write_csv(out / "02_model_confidence_summary.csv")
    print(f"[2] P66 model {af_p66['entry_id']} v{af_p66['version']}; whole-chain mean "
          f"pLDDT {af_p66['mean_plddt']}; loop pLDDT "
          f"mean {in_loop['plddt'].mean():.1f}, min {in_loop['plddt'].min():.1f}, "
          f"max {in_loop['plddt'].max():.1f}")
    print(f"    experimental P66 structures in the PDB: "
          f"{len(experimental_p66)}; CD47 model mean pLDDT {af_cd47['mean_plddt']}")

    # --- Steps 3 and 4 --------------------------------------------------
    hits = search_rcsb_cd47_sirpa(cache)
    hits.write_csv(out / "03_rcsb_search.csv")
    chosen = pick_complex(hits)
    cif = fetch_cif(chosen["pdb_id"], cache)
    pl.DataFrame(
        [{k: str(v) for k, v in chosen.items()}]
    ).write_csv(out / "03_chosen_complex.csv")
    face = binding_face(cif, chosen["cd47_chains"], chosen["sirpalpha_chains"])
    # The crystal numbers the mature chain; UniProt numbers the precursor. Both
    # go in the table, because mixing them is the easiest way to be wrong here.
    face = face.with_columns(
        (pl.col("cd47_position") + cd47["signal_peptide_end"]).alias(
            "cd47_uniprot_position"
        )
    )
    face.write_csv(out / "04_cd47_sirpalpha_face.csv")
    print(f"[3] {chosen['pdb_id']} at {chosen['resolution_angstrom']} A: "
          f"CD47 {chosen['cd47_chains']}, SIRPalpha {chosen['sirpalpha_chains']}")
    print(f"[4] CD47 binding face: {face.height} residues within "
          f"{CONTACT_ANGSTROM} A of SIRPalpha")

    # --- Step 5 ---------------------------------------------------------
    cd47_chain_id = face["cd47_chain"][0]
    cd47_igv = write_chain_pdb(cif, cd47_chain_id, cache / "cd47_igv.pdb")
    pl.DataFrame([{k: str(v) for k, v in cd47_igv.items()}]).write_csv(
        out / "05_cd47_igv_domain.csv"
    )
    claimed = usalign(af_p66["pdb_path"], cd47_igv["path"])
    claimed_row = {"comparison": "P66 AlphaFold model vs CD47 IgV", **claimed}
    ig = scope_ig_domains()
    rng = np.random.default_rng(SEED)
    picked = ig[
        sorted(rng.choice(ig.height, size=min(N_IG_BACKGROUND, ig.height), replace=False))
    ]
    background = []
    for row in picked.iter_rows(named=True):
        result = usalign(Path(row["path"]), cd47_igv["path"])
        background.append(
            {"comparison": row["scope_id"], "sccs": row["sccs"],
             "description": row["description"], **result}
        )
    pl.DataFrame([claimed_row]).write_csv(out / "05_wholefold_claimed.csv")
    pl.DataFrame(background).write_csv(out / "05_wholefold_background.csv")
    ig.write_csv(out / "05_ig_superfamily_pool.csv")
    bg_tm = [b["tm_score_norm_target"] for b in background if b["tm_score_norm_target"]]
    print(f"[5] claimed pair TM-score (normalised by CD47) "
          f"{claimed['tm_score_norm_target']}, aligned {claimed['aligned_length']}, "
          f"RMSD {claimed['rmsd_angstrom']} A")
    print(f"    {len(bg_tm)} immunoglobulin domains vs CD47: median "
          f"{np.median(bg_tm):.3f}, range {min(bg_tm):.3f}-{max(bg_tm):.3f}")

    # --- Step 6 ---------------------------------------------------------
    if not args.skip_foldseek:
        fs_dir = cache / "foldseek"
        fs_dir.mkdir(exist_ok=True)
        pair_tsv = fs_dir / "p66_vs_cd47_model.tsv"
        fmt = "query,target,evalue,bits,alntmscore,qstart,qend,tstart,tend,alnlen,prob"
        subprocess.run(
            [str(FOLDSEEK), "easy-search", str(af_p66["pdb_path"]),
             str(af_cd47["pdb_path"]), str(pair_tsv), str(fs_dir / "tmp1"),
             "-e", "inf", "--exhaustive-search", "1",
             "--format-output", fmt, "--format-mode", "4"],
            check=True, capture_output=True, text=True,
        )
        pl.read_csv(pair_tsv, separator="\t").write_csv(out / "06_foldseek_vs_cd47_model.csv")
        if FOLDSEEK_PDB_DB.with_suffix(".dbtype").exists() or FOLDSEEK_PDB_DB.exists():
            pdb_tsv = fs_dir / "p66_vs_pdb.tsv"
            subprocess.run(
                [str(FOLDSEEK), "easy-search", str(af_p66["pdb_path"]),
                 str(FOLDSEEK_PDB_DB), str(pdb_tsv), str(fs_dir / "tmp2"),
                 "--format-output", fmt, "--format-mode", "4", "--max-seqs", "4000"],
                check=True, capture_output=True, text=True,
            )
            pl.read_csv(pdb_tsv, separator="\t").write_csv(out / "06_foldseek_vs_pdb.csv")
            print("[6] Foldseek against the PDB and the CD47 model: written")
        else:
            print("[6] Foldseek PDB database not present; CD47-model search only")

    # --- Step 7 ---------------------------------------------------------
    face_positions = set(face["cd47_position"].to_list())
    cd47_positions = [
        r.id[1]
        for r in next(iter(PDBParser(QUIET=True).get_structure("c", cd47_igv["path"])[0]))
        if is_aa(r, standard=True)
    ]
    loop_length = LOOP_MATURE_END - LOOP_MATURE_START + 1
    windows = contiguous_windows(cd47_positions, loop_length, face_positions)
    claimed_rmsd = backbone_rmsd_loop_on_windows(
        af_p66["pdb_path"], (loop["full_start"], loop["full_end"]),
        cd47_igv["path"], windows,
    ).with_columns(pl.lit("claimed loop").alias("loop"))
    claimed_rmsd.write_csv(out / "07_claimed_loop_vs_face.csv")

    ss = ca_secondary_structure(af_p66["pdb_path"])
    coil = ss.filter(pl.col("secondary_structure") == "coil")["full_position"].to_list()
    other_loops = [
        run
        for run in contiguous_windows(coil, loop_length, set(coil))
        if not set(run).intersection(range(loop["full_start"], loop["full_end"] + 1))
    ]
    rng2 = np.random.default_rng(SEED)
    chosen_loops = [
        other_loops[i]
        for i in sorted(
            rng2.choice(len(other_loops), size=min(N_LOOP_BACKGROUND, len(other_loops)),
                        replace=False)
        )
    ]
    loop_bg = []
    for run in chosen_loops:
        best = backbone_rmsd_loop_on_windows(
            af_p66["pdb_path"], (run[0], run[-1]), cd47_igv["path"], windows
        )
        if best.is_empty():
            continue
        row = best.row(0, named=True)
        loop_bg.append(
            {"loop": f"P66 {run[0]}-{run[-1]}", "p66_full_start": run[0],
             "p66_full_end": run[-1], **row}
        )
    pl.DataFrame(loop_bg).write_csv(out / "07_background_loops_vs_face.csv")
    ss.write_csv(out / "07_p66_secondary_structure.csv")
    best_claimed = claimed_rmsd.row(0, named=True)
    bg_rmsd = [r["backbone_rmsd_angstrom"] for r in loop_bg]
    print(f"[7] claimed loop best backbone RMSD {best_claimed['backbone_rmsd_angstrom']:.2f} A "
          f"on CD47 {best_claimed['window_start']}-{best_claimed['window_end']}")
    print(f"    {len(bg_rmsd)} other P66 loops: median {np.median(bg_rmsd):.2f} A, "
          f"best {min(bg_rmsd):.2f} A")
    # --- Step 8: what the two-letter alphabets see on the loop ------------
    hp = hp_class_strings(loop["residues"])
    hp.write_csv(out / "08_loop_hp_class_strings.csv")
    uniform = hp.filter(pl.col("n_distinct_classes") == 1)
    print(f"[8] loop {loop['residues']} encodes to one repeated class letter in "
          f"{uniform.height} of {hp.height} two-letter alphabets")

    # --- Step 9: the same question for every other alphabet -------------
    reach = loop_kmer_reach(
        p66["sequence"], cd47["sequence"], (loop["full_start"], loop["full_end"])
    )
    reach.write_csv(out / "09_loop_kmer_reach_all_alphabets.csv")
    per_alphabet = reach.group_by("alphabet", "n_classes").agg(
        pl.col("n_covering_kmers_found_in_cd47").sum().alias("found_over_all_k"),
        pl.col("n_distinct_classes_in_loop").first(),
        pl.col("loop_class_string").first(),
    ).sort("n_classes", "alphabet")
    print(f"[9] loop k-mers reaching CD47, {reach['alphabet'].n_unique()} alphabets "
          f"x k {min(LOOP_REACH_KSIZES)}-{max(LOOP_REACH_KSIZES)}: "
          f"{reach['n_covering_kmers_found_in_cd47'].sum()} in total")
    print(f"    alphabets where the loop is one repeated class: "
          f"{(per_alphabet['n_distinct_classes_in_loop'] == 1).sum()} of "
          f"{per_alphabet.height}")
    highest = reach.filter(pl.col("n_covering_kmers_found_in_cd47") > 0)
    if highest.is_empty():
        print("    no alphabet at any k puts a loop k-mer in CD47")
    else:
        print(f"    highest k that still does: "
              f"{highest['ksize'].max()} ({highest.filter(pl.col('ksize') == highest['ksize'].max())['alphabet'].to_list()})")

    print(f"\nWrote {len(list(out.glob('*.csv')))} tables to {out}")


if __name__ == "__main__":
    main()
