# Regions-index pilot: build report (step 2, before any search)

Built 2026-10-06 on the laptop with `make build-local` (`-profile local`). Swiss-Prot release
2026_03; checksums in `build_summary/inputs_and_outputs.md5`. Every number below is printed
by the pipeline into `build_summary/` unless it is marked as an estimate.

## Queries

975 reviewed human entries on chromosome 6 (UniProt 2026_03), all present in the flat file.
866 of them carry at least one feature of the nine types: 7_174 features in all.

| chromosome 6 group | proteins | features |
|---|---|---|
| MHC (HLA-*) | 18 | 107 |
| histone (H1-*, H2AC*, H2BC*, H3C*, H4C*) | 27 | 86 (26 proteins carry features) |
| olfactory receptor (OR + digits) | 17 | 119 |
| everything else | 913 | 6_862 |

HLA-DQB1, HLA-DRB3 and HLA-DRB4 are not in the list: UniProt puts them in the reference
proteome's "Unplaced" component, not "Chromosome 6".

## Truth features, by kind and evidence

| kind | rule | all evidence | experimental (ECO:0000269) | proteins | median length (aa) |
|---|---|---|---|---|---|
| folded domain | DOMAIN, REPEAT, ZN_FING of 60 aa or more | 907 | 6 | 396 | 103 |
| motif | MOTIF | 179 | 37 | 134 | 5 |
| disordered | REGION "Disordered", experimental or at least half inside DisProt disorder | 45 | 0 | 30 | 43 |
| composition-driven | TRANSMEM, INTRAMEM, COILED, COMPBIAS | 3_185 | 80 | 686 | 20 |
| other | other REGIONs, unconfirmed "Disordered", DOMAIN/REPEAT/ZN_FING under 60 aa | 2_858 | 185 | 675 | 37 |
| short (under 60 aa, any kind) | overlaps the rows above | 5_460 | 210 | 818 | 21 |

1_241 REGION "Disordered" features fail the disordered rule and sit in "other". 1_240 of
them carry ECO:0000256 alone, UniProt's code for an automatic annotation (BCL2's, for
example, cites SAM:MobiDB-lite, a disorder predictor).
62 chromosome 6 proteins have DisProt entries (96 consensus disorder regions).

Experimental evidence is rare across all of Swiss-Prot, not only on chromosome 6. The
share of features with ECO:0000269: DOMAIN 0.1%, ZN_FING 0.4%, COILED 0.6%, TRANSMEM 1.2%,
REPEAT 1.7%, REGION 2.2%, MOTIF 3.4%, INTRAMEM 6.0%, COMPBIAS 0%.

## The two indexes

| | whole-protein index | regions index |
|---|---|---|
| Swiss-Prot entries used | 507_740 (68_008 mammals removed) | the same 507_740 |
| index entries (targets) | 507_740 | 637_655 |
| residues, targets | 175_144_459 | 43_379_517 |
| residues, targets + decoys (what is searched) | 350_288_918 | 86_759_034 |

The regions index starts from 991_113 cut-outs. 358_042 are of features of 30 aa or more;
MMseqs2 18.8cc5c clusters them into 143_348 representatives. 633_071 are shorter, and
494_307 of those have distinct sequences. 7_476 clusters mix feature types (5_371 of them
COMPBIAS).

| feature type | cut-outs | index entries | multi-member clusters, one type | mixed-type clusters |
|---|---|---|---|---|
| TRANSMEM | 301_037 | 220_287 | 26_766 | 4 |
| REGION | 225_452 | 136_648 | 20_822 | 1_310 |
| COMPBIAS | 172_620 | 150_236 | 9_773 | 5_371 |
| DOMAIN | 159_554 | 41_330 | 15_263 | 445 |
| REPEAT | 66_135 | 43_497 | 6_880 | 31 |
| MOTIF | 36_946 | 26_951 | 3_848 | 75 |
| COILED | 14_301 | 9_217 | 1_343 | 209 |
| ZN_FING | 13_587 | 8_596 | 1_326 | 15 |
| INTRAMEM | 1_481 | 893 | 146 | 16 |

48% of the regions index's residues are flank: 20_891_916 of 43_379_517. A 5-residue motif
becomes a 41-residue entry.

## Bits needed to reach E = 1

log2(K·m·n), with m = 409 aa (the median query length) and K = 0.0180. K is borrowed from
kmerseek `docs/evalue.md` (a 15_000-sequence Swiss-Prot sample, hp_thomas_dill2 k = 12,
C = 2), not fitted on these indexes. The difference between the indexes does not depend
on K.

| index | n counted | bits needed | aligned positions needed at 0.157 bits each (assumed) |
|---|---|---|---|
| whole-protein | targets, 175.1 M | 30.26 | 193 |
| regions | targets, 43.4 M | 28.25 | 180 |
| whole-protein | targets + decoys, 350.3 M | 31.26 | 199 |
| regions | targets + decoys, 86.8 M | 29.25 | 186 |

The regions index lowers the bar by 2.01 bits, not the 3.3 a tenfold smaller index would
give: it is 4.04 times smaller. The 0.157 bits per aligned position is the value given in
the pilot's design for the H/P alphabet at 20–30% identity. It is not measured here. At
that value an H/P hit needs about 180 aligned positions against either index, so a feature
under 60 aa cannot reach E = 1 in either one.

Other designs, estimated from the same representatives without re-clustering:

| design | entries | residues (targets) | bits saved against the whole-protein index |
|---|---|---|---|
| as built: nine types, flank 18 | 637_655 | 43_379_517 (measured) | 2.01 |
| nine types, flank 10 (k_max 11, polarity4 alone) | 637_655 | 34_280_215 | 2.35 |
| nine types, no flank | 637_655 | 22_487_601 | 2.96 |
| without TRANSMEM, INTRAMEM, COILED, COMPBIAS, flank 18 | 257_022 | 22_667_335 | 2.95 |
| also without REGION "Disordered", flank 18 | 158_923 | 15_092_992 | 3.54 |
| also without REGION "Disordered", flank 10 | 158_923 | 12_800_825 | 3.77 |

## Proposed kmerseek settings

From olgabot/dark-set-kmerseek-0.4 `main.nf` (7913404): image
`docker.io/olgabot/kmerseek:0.4.0-rc5`, `--threshold 0.0 --min-shared-kmers 2
--max-query-pvalue 0.05 --min-region-score 1.3`, extension with the alphabet's own mismatch
penalty (`opt`, from `assets/kappa_by_alphabet.tsv` there), X-drop = 4 × penalty, `--chain-max-gap 30 --chain-max-shift 10`.

| setting | alphabet | k | scaled | mask | penalty C | X-drop | source |
|---|---|---|---|---|---|---|---|
| 1 | polarity4 | 11 | 10 | off | 0.54 | 2.16 | the dark-set run (given) |
| 2 (proposed) | hp_pbotc_1st_ed2 | 19 | 1 | off | 1.59 | 6.36 | the designated H/P setting (alphabet ranking, nb 200); on the 0.4 ladder |
| 3 (proposed) | protein20 | 5 | 1 | off | 0.14 | 0.56 | the 2026-10-01 mini-set decoy run: at 1 false region per 10 shuffled queries, `region_evalue` kept 40% of true regions for protein20 k5, 18% for mmseqs12, 3–7% for H/P alphabets |

MMseqs2: `mmseqs search -s 7 --num-iterations 3 -e 10`, as in the dark set's mmseqs2Search,
same image.

## Open decisions before any search

1. **Go or change the index.** As built, the regions index saves 2.0 bits. Shrinking it
   further means dropping feature types, which changes the question.
2. **The experimental-only headline cannot be measured as specified.** It would rest on 0
   disordered features, 6 folded domains and 37 motifs. Options: headline on all
   evidence, with experimental as a secondary split; or experimental-or-DisProt for every
   kind.
3. **Settings 2 and 3, and scaled for setting 1.** At scaled 10, a k-mer survives with
   probability 1/10. A 20-aa feature holds 10 k-mers of k = 11, so it keeps about one on
   average, below `--min-shared-kmers 2`. Setting 1 may miss short features for that reason alone.
4. **MMseqs2 version.** The local clustering used MMseqs2 18.8cc5c. The pinned container's
   version has not been checked; the Sherlock build may cluster slightly differently.
