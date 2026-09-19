/*
 * The kmerseek alphabet x ksize matrix, shared by every pipeline that sweeps it.
 *
 * One copy. qfo-pfam-region-benchmark and invertebrate-dark-set both expand the same
 * matrix, and a second literal of this table in either main.nf is a copy nothing checks
 * against the original -- the exact drift that already happened once on this project
 * between the pipeline, make_mini_testset.py and build_pfam_architectures.py.
 *
 *   include { allEncodings; extraEncodings; knownEncodings; expandEncodings } \
 *       from '../shared/kmerseek_encodings'
 *
 * Every row is [cli_flag, label, kmin, kmax]. In kmerseek v0.4.0 the CLI name and the
 * moltype written into the CSV are the same string, so one column serves both. The LABEL
 * is a cache key: kmerseekIndex stores <target>.<label>.k<k>.lc<lc>.kmerseek.rocksdb, so
 * a renamed label orphans every index built under the old one.
 *
 * Every alphabet was renamed in kmerseek PR #43 to state its class count: protein is
 * protein20, dayhoff is dayhoff6, hp_lehninger is hp_lehninger2, hp_lehninger_plus_c is
 * hp_lehninger_c_nonpolar2, hp_shuffled_control is hp_random_control2. Results produced
 * under the old names will not join these labels.
 *
 * Twelve ksizes for the HP family, ten for everything else, from a bit-matched floor. The
 * HP alphabets get the wider range because they are what the paper is testing and the k
 * optimum is least constrained there.
 *
 * The floor uses real entropy, not log2(classes). log2(n) assumes every class is equally
 * likely, which overstates every coarse alphabet. The bits/symbol below come from
 * amino-acid background frequencies grouped as kmerseek groups them
 * (notebooks/ortholog_analysis_utils.entropy_per_symbol). HP carries 0.994 bits/symbol, so
 * its k18 floor is 17.9 bits, and every kmin is round(17.9 / bits). Below that floor a
 * coarse alphabet produces prohibitive output volume: a 2-letter alphabet at k=15 has
 * 32768 possible k-mers against ~20k x 20k proteins, and the measured output at k=18 was
 * already 838 MB compressed for the smallest QfO species.
 *
 * Two entries contradict class count, which is why entropy is measured rather than
 * assumed. hp_lehninger_hpc3 has three classes but 1.128 bits/symbol against HP's 0.994,
 * because cysteine is ~1.4% of residues. gbmr7 carries less information than wwmj5, 1.976
 * against 2.197, despite two more classes, because its classes are unbalanced.
 */
def allEncodings() {
    [
        // k=4 dropped: 20^4 = 160_000 keys against ~11.3M proteome k-mers means the entire
        // keyspace is occupied ~70 times over, so every query 4-mer matches a large share
        // of the proteome. It OOM-killed its task and the result would be noise either way.
        ['protein20', 'protein20', 5, 13],                      // 4.176 bits/sym, 9 ksizes
        ['uniprot18', 'uniprot18', 5, 14],                      // 3.951 bits/sym, 10 ksizes
        ['hsdm17', 'hsdm17', 5, 14],                            // 3.742 bits/sym, 10 ksizes
        ['wass14', 'wass14', 5, 14],                            // 3.626 bits/sym, 10 ksizes
        ['mmseqs12', 'mmseqs12', 5, 14],                        // 3.293 bits/sym, 10 ksizes
        ['sdm12', 'sdm12', 6, 15],                              // 3.127 bits/sym, 10 ksizes
        ['dayhoff6', 'dayhoff6', 8, 17],                        // 2.278 bits/sym, 10 ksizes
        ['wwmj5', 'wwmj5', 8, 17],                              // 2.197 bits/sym, 10 ksizes
        ['gbmr7', 'gbmr7', 9, 18],                              // 1.976 bits/sym, 10 ksizes
        ['gbmr4', 'gbmr4', 12, 21],                             // 1.522 bits/sym, 10 ksizes
        ['hp_lehninger_hpc3', 'hp_lehninger_hpc3', 16, 27],     // 1.128 bits/sym, 12 ksizes
        ['hp_lehninger2', 'hp_lehninger2', 18, 29],             // 1.000 bits/sym, 12 ksizes
        ['hp_lehninger_c_nonpolar2', 'hp_lehninger_c_nonpolar2', 18, 29],// 0.999 bits/sym, 12 ksizes
        ['hp_pbotc_1st_ed2', 'hp_pbotc_1st_ed2', 18, 29],       // 0.994 bits/sym, 12 ksizes
        ['hp_thomas_dill2', 'hp_thomas_dill2', 19, 30],         // 0.966 bits/sym, 12 ksizes
        ['hp_thomas_dill_no_c2', 'hp_thomas_dill_no_c2', 19, 30],// 0.951 bits/sym, 12 ksizes
        ['hp_kyte_doolittle2', 'hp_kyte_doolittle2', 19, 30],   // 0.937 bits/sym, 12 ksizes
    ]
}

/*
 * Alphabets that exist in kmerseek but are NOT in the default matrix.
 *
 * polarity4 and funcgroups8 arrived in kmerseek dd630a8 (2026-08-25), from Rannon &
 * Burstein (2026), which credits funcgroups8 to Jain, Jain & Jain (2014) and polarity4 to
 * Ball, Hill & Scott (2014). The region benchmark's sweep was built from 3fdfd51a
 * (2026-08-24), whose binary does not have them, and a bare `nextflow run` there must keep
 * expanding to exactly allEncodings() or an in-flight sweep widens by 40 combos on its
 * next -resume, every one of them dying under the old image on an unrecognised flag. So
 * they are reachable only by asking: --kmerseek_encodings polarity4,funcgroups8 in the
 * region benchmark, --kmerseek_extra_encodings true in the dark set.
 *
 * Both ksize ranges come from the same measurement as every row above -- amino-acid
 * background frequencies grouped as kmerseek groups them, kmin = round(17.9 / bits) --
 * and both are cases where class count would have given the wrong answer:
 *
 *   polarity4   GAVLIFWMP / STCYNQ / DE / HKR         1.787 bits/sym, not log2(4) = 2.0
 *   funcgroups8 GVALI / ST / CM / FY / WHP / NQ / DE / KR
 *                                                     2.727 bits/sym, not log2(8) = 3.0
 *
 * polarity4 carries 1.787 against gbmr4's 1.522 at the same four classes, because it puts
 * its split on charge with G and P folded into the hydrophobic class, where gbmr4 keeps G
 * and P each alone and so spends two of its four classes on 12.7% of residues. That 0.265
 * bits/symbol is the difference between k=10 and k=12 at the floor.
 *
 * funcgroups8 lands ABOVE dayhoff6 (2.278) and gbmr7 (1.976), which is what eight
 * reasonably balanced classes buys, and its 6.56 rounds up to 7 rather than down to 6:
 * k=6 would be 16.4 bits, under the 17.9-bit floor, which is the regime that OOM-killed
 * protein20 at k=4.
 */
def extraEncodings() {
    [
        ['polarity4', 'polarity4', 10, 19],                     // 1.787 bits/sym, 10 ksizes
        ['funcgroups8', 'funcgroups8', 7, 16],                  // 2.727 bits/sym, 10 ksizes
    ]
}

// Everything either flag may name. The default matrix stays allEncodings(); this is the
// lookup table.
def knownEncodings() {
    allEncodings() + extraEncodings()
}

// Rows to (cli_flag, label, ksize) triples, one per ksize in the row's range.
def expandEncodings(rows) {
    rows.collectMany { cli, label, kmin, kmax ->
        (kmin..kmax).collect { k -> [cli, label, k] }
    }
}

// Class count is the trailing number in every encoding name: protein20, gbmr4,
// hp_lehninger_hpc3, hp_thomas_dill_no_c2.
def alphabetClasses(label) {
    def m = label =~ /(\d+)$/
    m ? (m[0][1] as int) : 20
}

// How big the k-mer keyspace is, in bits: ksize x log2(alphabet cardinality). The one
// number that predicts kmerseek's memory on both sides: a small keyspace saturates, every
// query k-mer matches a large share of the target, and the SEARCH explodes; a large one
// leaves every target k-mer distinct and the INDEX carries them all.
def keyspaceBits(label, ksize) {
    ksize * (Math.log(alphabetClasses(label)) / Math.log(2.0d))
}
