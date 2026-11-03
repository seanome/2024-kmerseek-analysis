"""Six-frame context translation; only stops delimit stretches."""

from Bio.Seq import Seq


def translate_context(dna):
    fragments, membership, coordinates = {}, {}, {}
    for origin, sequence in dna.items():
        for strand, oriented in [
            ("+", sequence),
            ("-", str(Seq(sequence).reverse_complement())),
        ]:
            for frame in range(3):
                coding = oriented[frame:]
                coding = coding[: len(coding) - len(coding) % 3]
                peptide = str(Seq(coding).translate())
                offset = 0
                for stretch in peptide.split("*"):
                    if len(stretch) >= 30:
                        fid = f"FRAG{len(fragments)+1:012d}"
                        fragments[fid] = stretch
                        membership[fid] = origin
                        coordinates[fid] = dict(
                            origin=origin,
                            strand=strand,
                            frame=frame,
                            amino_acid_offset=offset,
                        )
                    offset += len(stretch) + 1
    return fragments, membership, coordinates
