"""Write tables/deep_twilight_pairs.tsv and the pipeline's 16-protein FASTA from UniProt.

Three families of experimentally characterized homologs, chosen by hand: globins,
lysozyme / alpha-lactalbumin, and cystatins. For each protein the table holds the residues
of its functional label, copied from that entry's own Swiss-Prot features. The only
hand-written part is RULES: which Swiss-Prot feature on which protein carries which label.

One row per protein per label, because the cystatins carry two labels (the reactive site
and the contact motif). Positions are 1-based and inclusive, as in UniProt, and refer to
the full UniProt sequence including any signal peptide.

    python scripts/make_deep_twilight_pairs_tsv.py
"""

import json
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TSV = REPO / "tables" / "deep_twilight_pairs.tsv"
FASTA = (
    REPO
    / "nextflow-runs"
    / "deep-twilight-controls"
    / "assets"
    / "deep_twilight_proteins.fasta"
)

# (family, accession, common name)
PROTEINS = [
    ("globin", "P02144", "human myoglobin"),
    ("globin", "P69905", "human hemoglobin alpha"),
    ("globin", "P68871", "human hemoglobin beta"),
    ("globin", "P02185", "sperm whale myoglobin"),
    ("globin", "P02238", "soybean leghemoglobin A"),
    ("globin", "P02229", "midge erythrocruorin III"),
    ("globin", "P02216", "bloodworm monomeric globin"),
    ("globin", "Q42831", "barley non-symbiotic hemoglobin"),
    ("lysozyme_lactalbumin", "P00698", "hen egg-white lysozyme"),
    ("lysozyme_lactalbumin", "P61626", "human lysozyme"),
    ("lysozyme_lactalbumin", "P00709", "human alpha-lactalbumin"),
    ("lysozyme_lactalbumin", "P00711", "bovine alpha-lactalbumin"),
    ("cystatin", "P01038", "chicken egg-white cystatin"),
    ("cystatin", "P01034", "human cystatin C"),
    ("cystatin", "P04080", "human stefin B"),
    ("cystatin", "P01040", "human stefin A"),
]


def heme_iron_histidine(f):
    """The histidine bound to the heme iron. Swiss-Prot calls it the "proximal binding
    residue" on six of the globins and the "axial binding residue" on the midge and
    bloodworm globins. The distal residue and O2 sites are left out."""
    return (
        f["type"] == "Binding site"
        and f.get("ligand", {}).get("name") == "heme b"
        and f.get("description")
        in {"proximal binding residue", "axial binding residue"}
    )


# label -> which Swiss-Prot features carry it
RULES = {
    "heme iron histidine": heme_iron_histidine,
    "catalytic residue": lambda f: f["type"] == "Active site",
    "calcium-binding residue": lambda f: f["type"] == "Binding site"
    and f.get("ligand", {}).get("name") == "Ca(2+)",
    "reactive site": lambda f: f["type"] == "Site"
    and f.get("description") == "Reactive site",
    "secondary area of contact": lambda f: f["type"] == "Motif"
    and f.get("description") == "Secondary area of contact",
}


def fetch(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def positions(features):
    out = []
    for f in features:
        start, end = f["location"]["start"]["value"], f["location"]["end"]["value"]
        out.extend(range(start, end + 1))
    return sorted(set(out))


def main():
    rows, fasta = [], []
    for family, accession, common in PROTEINS:
        entry = fetch(accession)
        assert entry["entryType"].startswith("UniProtKB reviewed"), accession
        seq = entry["sequence"]["value"]
        signal = [f for f in entry.get("features", []) if f["type"] == "Signal"]
        signal_str = (
            f"{signal[0]['location']['start']['value']}-{signal[0]['location']['end']['value']}"
            if signal
            else ""
        )
        for label, rule in RULES.items():
            hits = [f for f in entry.get("features", []) if rule(f)]
            if not hits:
                continue
            pos = positions(hits)
            residues = "".join(seq[p - 1] for p in pos)
            evidence = sorted(
                {
                    e.get("evidenceCode", "")
                    for f in hits
                    for e in f.get("evidences", [])
                }
                - {""}
            )
            described = sorted(
                {
                    f"{f['type']}"
                    + (f": {f['description']}" if f.get("description") else "")
                    + (f" ({f['ligand']['name']})" if f.get("ligand") else "")
                    for f in hits
                }
            )
            rows.append(
                [
                    family,
                    accession,
                    entry["uniProtkbId"],
                    common,
                    entry["organism"]["scientificName"],
                    str(len(seq)),
                    signal_str,
                    label,
                    ",".join(map(str, pos)),
                    residues,
                    "; ".join(described),
                    ",".join(evidence),
                ]
            )
        fasta.append(f">sp|{accession}|{entry['uniProtkbId']} {common}\n{seq}\n")

    header = [
        "family",
        "accession",
        "entry_name",
        "common_name",
        "organism",
        "length",
        "signal_peptide",
        "label",
        "positions",
        "residues",
        "swissprot_features",
        "evidence_codes",
    ]
    TSV.parent.mkdir(parents=True, exist_ok=True)
    TSV.write_text(
        "\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    )
    FASTA.write_text("".join(fasta))
    print(f"wrote {len(rows)} rows for {len(PROTEINS)} proteins to {TSV}")


if __name__ == "__main__":
    main()
