//! k-mer hashes of a peptide under one kmerseek alphabet.
//!
//! Each residue is encoded the way kmerseek's `ProteinSketch::add_protein` encodes it before
//! hashing (its private `hash_encoder`, rebuilt here from kmerseek's public
//! `alphabet_table` and `get_encoding_fn_from_moltype`), and each window is hashed with
//! sourmash's `_hash_murmur` and kmerseek's seed. The test below checks the hashes against
//! `ProteinSketch` itself. A window holding an X is skipped: the peptide is split at X first.
//! With scaled = s, only hashes at or below sourmash's `max_hash_for_scaled(s)` are kept,
//! about 1 in s, the same rule sourmash's FracMinHash uses.

use anyhow::Result;
use kmerseek::alphabets::alphabet_table;
use kmerseek::hash_functions::get_encoding_fn_from_moltype;
use kmerseek::SEED;
use sourmash::_hash_murmur;
use sourmash::sketch::minhash::max_hash_for_scaled;

/// One alphabet at one k, with its residue encoder and the largest hash kept.
pub struct Alphabet {
    pub name: String,
    pub k: usize,
    pub scaled: u32,
    pub max_hash: u64,
    encode: [u8; 256],
}

impl Alphabet {
    pub fn new(name: &str, k: usize, scaled: u32) -> Result<Self> {
        let table = alphabet_table(name);
        let encoding_fn = get_encoding_fn_from_moltype(name)?;
        let mut encode = [0u8; 256];
        for b in 0..=255u8 {
            // Table alphabets: the class symbol, uppercased, since sourmash uppercases
            // protein input before hashing. protein20, dayhoff6, hp_lehninger2: sourmash's
            // own encoder. Same branches as kmerseek's hash_encoder.
            encode[b as usize] = match table {
                Some(t) => t.get(&b.to_ascii_uppercase()).copied().unwrap_or(b).to_ascii_uppercase(),
                None => encoding_fn(b),
            };
        }
        Ok(Self { name: name.to_string(), k, scaled, max_hash: max_hash_for_scaled(scaled), encode })
    }

    /// Call `f` with the hash of every window of `peptide` that holds no X and is kept at
    /// this scaled.
    pub fn for_each_hash(&self, peptide: &[u8], buf: &mut Vec<u8>, mut f: impl FnMut(u64)) {
        for run in peptide.split(|&c| c == b'X') {
            if run.len() < self.k {
                continue;
            }
            buf.clear();
            buf.extend(run.iter().map(|&b| self.encode[b as usize]));
            for w in buf.windows(self.k) {
                let h = _hash_murmur(w, SEED);
                if h <= self.max_hash {
                    f(h);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kmerseek::sketch::ProteinSketch;
    use std::collections::HashSet;

    // Human BCL2 residues 1-60 (UniProt P10415).
    const SEQ: &str = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAA";
    // Human BCL2, full length (UniProt P10415), for the scaled test: 60 residues keep too
    // few hashes at scaled 5 to show that the right ones are left out.
    const BCL2: &str = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLHLTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEYLNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK";

    #[test]
    fn hashes_match_kmerseek_protein_sketch() {
        let names = [
            "protein20", "dayhoff6", "hp_lehninger2", "hp_thomas_dill2", "hp_kyte_doolittle2",
            "hp_thomas_dill_no_c2", "hp_lehninger_c_nonpolar2", "hp_lehninger_hpc3",
            "hp_pbotc_1st_ed2", "gbmr4", "polarity4", "wwmj5", "gbmr7", "funcgroups8",
            "sdm12", "mmseqs12", "wass14", "hsdm17", "uniprot18",
        ];
        for name in names {
            for k in [5usize, 7, 13] {
                let a = Alphabet::new(name, k, 1).unwrap();
                let mut ours = HashSet::new();
                a.for_each_hash(SEQ.as_bytes(), &mut Vec::new(), |h| {
                    ours.insert(h);
                });
                let mut s = ProteinSketch::new("t", k as u32, 1, name).unwrap();
                s.add_protein(SEQ, false).unwrap();
                assert_eq!(ours, s.mins_as_set(), "{name} k={k}");
            }
        }
    }

    #[test]
    fn scaled_keeps_the_hashes_kmerseek_keeps() {
        for name in ["protein20", "hp_lehninger2", "gbmr4", "sdm12"] {
            for scaled in [5u32, 10] {
                let a = Alphabet::new(name, 7, scaled).unwrap();
                let mut ours = HashSet::new();
                a.for_each_hash(BCL2.as_bytes(), &mut Vec::new(), |h| {
                    ours.insert(h);
                });
                let mut s = ProteinSketch::new("t", 7, scaled, name).unwrap();
                s.add_protein(BCL2, false).unwrap();
                assert_eq!(ours, s.mins_as_set(), "{name} scaled={scaled}");
                // 233 windows; about 1 in `scaled` kept, and some, but not all
                assert!(!ours.is_empty() && ours.len() < 100, "{name} scaled={scaled}: {}", ours.len());
            }
        }
    }

    #[test]
    fn x_windows_are_skipped() {
        let a = Alphabet::new("protein20", 3, 1).unwrap();
        let mut n = 0;
        a.for_each_hash(b"ACDXEFGH", &mut Vec::new(), |_| n += 1);
        // ACD has one window; EFGH has two; none spans the X.
        assert_eq!(n, 3);
    }
}
