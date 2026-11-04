# The k-mer table format

Every k-mer table is one tab-separated file with one setting in it. A setting is one alphabet at one
k. `geo2kmers/schema.py` checks a table against this file, and `sketch` checks each table it writes.

| column | required | meaning |
|---|---|---|
| `kmer` | yes | the k-mer, once per table. DNA k-mers are in canonical form: the smaller of the k-mer and its reverse complement |
| `alphabet` | yes | `dna`, `protein`, `dayhoff` or `hp` |
| `k` | yes | k-mer length: bases for `dna`, residues for the others |
| `setting` | yes | the alphabet's name, then k: `dna_k31`, or with kmerseek's names `protein20_k10`, `dayhoff6_k19`, `hp_lehninger2_k30`. kmerseek has no DNA alphabet, so `dna` is this format's own name. Tables with different settings are never joined |
| value column | no | see below |
| `n_probes` | no | array tables only, so never beside `count`: how many probes hold the k-mer, a whole number above 0 |

Letters: `dna` uses A C G T. `protein` uses the 20 amino acids. `dayhoff` uses a to f and `hp` uses h
and p, with the groups in the main README. Columns come in the order above.

The value column is one of two, and a table never has both:

- `count`: how many times the k-mer was counted, in reads or in sequence. A whole number.
- `intensity_weight`: array intensity above background, summed over the probes that hold the k-mer.
  It has no unit shared between arrays and is not a count.

A table with no value column is a plain set of k-mers. `sketch --mode present` writes this.

Read sketches from kmerseek are expected to follow this file, with `count` as the value column.
