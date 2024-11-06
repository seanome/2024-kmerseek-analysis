# ! /usr/bin/env python3
"""
Given a signature file and a collection of sequences, output all of the
k-mers and sequences that match a hashval in the signature file.

Cribbed from https://github.com/dib-lab/sourmash/pull/724/
"""
import argparse
import csv
import sys

import screed

import sourmash
from sourmash.cli.utils import add_construct_moltype_args, add_ksize_arg
from sourmash.logging import error, notify
from sourmash.minhash import hash_murmur
from sourmash.sourmash_args import calculate_moltype

NOTIFY_EVERY_BP = int(1e5)


def degenerate_protein_chatgpt(sequence, moltype):
    """Convert protein sequence from 20-letter amino acid alphabet to degenerate alphabet"""
    # Slightly faster than other version (3.05 µs/loop for this one vs 3.4 µs/loop for the other one)
    # Pre-determine the alphabet function based on moltype
    if moltype == "hp":
        alphabet_func = sourmash._lowlevel.lib.sourmash_aa_to_hp
    elif moltype == "dayhoff":
        alphabet_func = sourmash._lowlevel.lib.sourmash_aa_to_dayhoff
    elif moltype == "DNA" or moltype == "protein":
        # No transformation on the moltype -> Return None
        return
    else:
        raise ValueError(f"Unknown moltype: {moltype}")

    # Convert the entire sequence to bytes once
    byte_sequence = sequence.encode("utf-8")

    # Apply the alphabet function to each byte in the sequence and join the result
    degenerate = b"".join(
        alphabet_func(letter.to_bytes(1, "big")) for letter in byte_sequence
    ).decode()

    return degenerate


def degenerate_protein(sequence, moltype):
    """Convert protein sequence from 20-letter amino acid alphabet to degenerate alphabet"""
    if moltype == "hp":
        alphabet = sourmash._lowlevel.lib.sourmash_aa_to_hp
    elif moltype == "dayhoff":
        alphabet = sourmash._lowlevel.lib.sourmash_aa_to_dayhoff
    else:
        raise ValueError(f"Unknown moltype: {moltype}")

    # Convert the entire sequence to bytes once
    byte_encoded = (x.encode("utf-8") for x in sequence)

    degenerate = b"".join(alphabet(letter) for letter in byte_encoded).decode()
    return degenerate


def get_kmer_moltype(sequence, start, ksize, moltype, input_is_protein):
    kmer_in_seq = sequence[start : start + ksize]
    if moltype == "DNA":
        # Get reverse complement
        kmer_rc = screed.rc(kmer_in_seq)
        if kmer_in_seq > kmer_rc:  # choose fwd or rc
            kmer_encoded = kmer_rc
        else:
            kmer_encoded = kmer_in_seq
    elif input_is_protein:
        kmer_encoded = degenerate_protein(kmer_in_seq, moltype)
    elif not input_is_protein:
        raise NotImplementedError("Currently cannot translate DNA to protein sequence")
    return kmer_encoded, kmer_in_seq


def revise_ksize(ksize, moltype, input_is_protein):
    """If input is protein, then divide the ksize by three"""
    if moltype == "DNA":
        return ksize
    elif input_is_protein:
        # Ksize includes codons
        return int(ksize / 3)
    else:
        return ksize


def get_kmers_for_hashvals(sequence, hashvals, ksize, moltype, input_is_protein):
    "Return k-mers from 'sequence' that yield hashes in 'hashvals'."
    # uppercase!
    sequence = sequence.upper()

    # Divide ksize by 3 if sequence is protein
    # ksize = revise_ksize(ksize, moltype, input_is_protein)

    for start in range(0, len(sequence) - ksize + 1):

        kmer_encoded, kmer_in_seq = get_kmer_moltype(
            sequence, start, ksize, moltype, input_is_protein
        )

        # NOTE: we do not avoid non-ACGT characters, because those k-mers,
        # when hashed, shouldn't match anything that sourmash outputs.
        hashval = hash_murmur(kmer_encoded)
        if hashval in hashvals:
            yield kmer_encoded, kmer_in_seq, hashval, start


def get_matching_hashes_in_file(
    filename,
    ksize,
    moltype,
    input_is_protein,
    hashes,
    found_kmers,
    m,
    n,
    n_seq,
    seqout_fp,
    kmerout_w,
    watermark,
    first=False,
):
    for record in screed.open(filename):
        n += len(record.sequence)
        n_seq += 1
        while n >= watermark:
            notify(
                "...Searched {:d} residues,\tfound {} kmers in\t{} seqs from\t{}",
                watermark,
                found_kmers,
                n_seq,
                filename,
                end="\r",
            )
            watermark += NOTIFY_EVERY_BP

        # now do the hard work of finding the matching k-mers!
        for kmer_encoded, kmer_in_seq, hashval, i in get_kmers_for_hashvals(
            record.sequence, hashes, ksize, moltype, input_is_protein
        ):
            found_kmers += 1
            # found_kmers.append([kmer_in_seq, kmer_encoded, hashval, record["name"]])

            # write out sequence
            if seqout_fp:
                seqout_fp.write(
                    ">{}|hashval:{}|kmer:{}|kmer_encoded:{}\n{}\n".format(
                        record.name, hashval, kmer_in_seq, kmer_encoded, record.sequence
                    )
                )
                m += len(record.sequence)
            if kmerout_w:
                kmerout_w.writerow(
                    [
                        kmer_in_seq,
                        kmer_encoded,
                        str(hashval),
                        i,
                        record["name"],
                        filename,
                    ]
                )

            if first:
                return m, n
    return m, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")  # signature file
    p.add_argument(
        "seqfiles", nargs="+"
    )  # sequence files from which to look for matches
    p.add_argument(
        "--output-sequences",
        type=str,
        default=None,
        help="save matching sequences to this file.",
    )
    p.add_argument(
        "--output-kmers",
        type=str,
        default=None,
        help="save matching kmers to this file.",
    )
    p.add_argument(
        "--input-is-protein",
        action="store_true",
        help="Consume protein sequences - no translation needed.",
    )
    add_ksize_arg(p)
    add_construct_moltype_args(p)
    args = p.parse_args()

    # set up the outputs.
    seqout_fp = None
    if args.output_sequences:
        seqout_fp = open(args.output_sequences, "wt")

    kmerout_fp = None
    if args.output_kmers:
        kmerout_fp = open(args.output_kmers, "wt")
        kmerout_w = csv.writer(kmerout_fp)
        kmerout_w.writerow(
            [
                "kmer_in_sequence",
                "kmer_in_alphabet",
                "hashval",
                "start",
                "read_name",
                "filename",
            ]
        )

    # Ensure that protein ksizes are divisible by 3
    if (args.protein or args.dayhoff or args.hp) and not args.input_is_protein:
        if args.ksize % 3 != 0:
            error("protein ksizes must be divisible by 3, sorry!")
            error("bad ksizes: {}", ", ".join(args.ksize))
            sys.exit(-1)

    if not (seqout_fp or kmerout_fp):
        error("No output options given!")
        return -1

    # first, load the signature and extract the hashvals
    moltype = calculate_moltype(args)
    sigobj = sourmash.load_one_signature(
        args.query, ksize=args.ksize, select_moltype=moltype
    )
    query_hashvals = set(sigobj.minhash.hashes.keys())
    query_ksize = sigobj.minhash.ksize

    # now, iterate over the input sequences and output those that overlap
    # with hashes!
    n_seq = 0
    n = 0  # bp loaded
    m = 0  # bp in found sequences
    p = 0  # number of k-mers found
    found_kmers = 0
    watermark = int(NOTIFY_EVERY_BP)
    for filename in args.seqfiles:
        m, n = get_matching_hashes_in_file(
            filename,
            query_ksize,
            moltype,
            args.input_is_protein,
            query_hashvals,
            found_kmers,
            m,
            n,
            n_seq,
            seqout_fp,
            kmerout_w,
            watermark,
        )

    if seqout_fp:
        notify("read {} bp, wrote {} bp in matching sequences", n, m)

    if kmerout_fp and found_kmers:
        # for kmer_in_seq, kmer_encoded, hashval, read_id in found_kmers:
        #     kmerout_w.writerow([kmer_in_seq, kmer_encoded, str(hashval), read_id])
        notify("read {} bp, found {} kmers matching hashvals", n, found_kmers)


if __name__ == "__main__":
    sys.exit(main())
