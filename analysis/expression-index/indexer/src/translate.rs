//! Six-frame translation of reads, with the rule for codons that contain N.
//!
//! The same rule as `translate.py`, which writes `codon_table_with_N.tsv`: standard genetic
//! code (NCBI table 1); a codon with one N stands for four codons and gets their amino acid
//! when all four agree, and X when any of the four is a stop or they disagree; a codon with
//! two or more Ns is X. X is never a stop. Any base other than A, C, G, T is read as N.

/// Fewest residues a piece of translation needs, between stops or the read's ends, to be kept.
pub const MIN_AA: usize = 30;

const BASES: &[u8; 4] = b"TCAG";
const AMINO: &[u8; 64] = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";

/// How a codon was translated. The names match the `rule` column of `codon_table_with_N.tsv`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    NoN = 0,
    OneNResolved = 1,
    OneNCouldBeStop = 2,
    OneNAmbiguous = 3,
    TwoOrMoreN = 4,
}

pub const RULE_NAMES: [&str; 5] =
    ["no_N", "one_N_resolved", "one_N_could_be_stop", "one_N_ambiguous", "two_or_more_N"];

/// Base to index: T C A G as 0..4 (the order of `AMINO`), anything else 4 (N).
fn base_index(b: u8) -> usize {
    match b {
        b'T' | b't' => 0,
        b'C' | b'c' => 1,
        b'A' | b'a' => 2,
        b'G' | b'g' => 3,
        _ => 4,
    }
}

fn standard(i: usize, j: usize, k: usize) -> u8 {
    AMINO[16 * i + 4 * j + k]
}

/// All 125 codons over T, C, A, G, N, indexed by `25 * i + 5 * j + k`.
pub struct CodonTable {
    aa: [u8; 125],
    rule: [Rule; 125],
}

impl CodonTable {
    pub fn new() -> Self {
        let mut aa = [b'X'; 125];
        let mut rule = [Rule::TwoOrMoreN; 125];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let idx = 25 * i + 5 * j + k;
                    let ns = [i, j, k].iter().filter(|&&x| x == 4).count();
                    let (a, r) = match ns {
                        0 => (standard(i, j, k), Rule::NoN),
                        1 => {
                            let four: Vec<u8> = (0..4)
                                .map(|b| {
                                    let p = |x: usize| if x == 4 { b } else { x };
                                    standard(p(i), p(j), p(k))
                                })
                                .collect();
                            if four.contains(&b'*') {
                                (b'X', Rule::OneNCouldBeStop)
                            } else if four.iter().all(|&x| x == four[0]) {
                                (four[0], Rule::OneNResolved)
                            } else {
                                (b'X', Rule::OneNAmbiguous)
                            }
                        }
                        _ => (b'X', Rule::TwoOrMoreN),
                    };
                    aa[idx] = a;
                    rule[idx] = r;
                }
            }
        }
        Self { aa, rule }
    }

    /// The amino acid and rule for a codon given as three bases.
    pub fn get(&self, codon: &[u8]) -> (u8, Rule) {
        let idx = 25 * base_index(codon[0]) + 5 * base_index(codon[1]) + base_index(codon[2]);
        (self.aa[idx], self.rule[idx])
    }
}

impl Default for CodonTable {
    fn default() -> Self {
        Self::new()
    }
}

fn complement(b: u8) -> u8 {
    match b {
        b'A' | b'a' => b'T',
        b'C' | b'c' => b'G',
        b'G' | b'g' => b'C',
        b'T' | b't' => b'A',
        _ => b'N',
    }
}

/// One kept piece of translation.
#[derive(Debug, PartialEq, Eq)]
pub struct Piece {
    /// `b'+'` for the read as given, `b'-'` for its reverse complement.
    pub strand: u8,
    pub frame: u8,
    /// Index of the piece's first residue in that frame's translation.
    pub start: usize,
    pub peptide: Vec<u8>,
}

/// Translate one frame of `seq`, adding to `counts` (indexed by `Rule`).
pub fn translate_frame(table: &CodonTable, seq: &[u8], frame: usize, counts: &mut [u64; 5]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seq.len() / 3);
    let mut i = frame;
    while i + 3 <= seq.len() {
        let (aa, rule) = table.get(&seq[i..i + 3]);
        counts[rule as usize] += 1;
        out.push(aa);
        i += 3;
    }
    out
}

/// The six frames of `read`, cut at stops, keeping pieces of at least `MIN_AA` residues.
pub fn kept_translations(table: &CodonTable, read: &[u8], counts: &mut [u64; 5]) -> Vec<Piece> {
    let rc: Vec<u8> = read.iter().rev().map(|&b| complement(b)).collect();
    let mut kept = Vec::new();
    for (strand, s) in [(b'+', read), (b'-', rc.as_slice())] {
        for frame in 0..3 {
            let prot = translate_frame(table, s, frame, counts);
            let mut start = 0;
            for piece in prot.split(|&c| c == b'*') {
                if piece.len() >= MIN_AA {
                    kept.push(Piece { strand, frame: frame as u8, start, peptide: piece.to_vec() });
                }
                start += piece.len() + 1;
            }
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_checked_codons() {
        // The hand check of codon_table_with_N.tsv, repeated against this table.
        let t = CodonTable::new();
        for (c, aa) in [("GCN", b'A'), ("CGN", b'R'), ("GGN", b'G'), ("CTN", b'L'),
                        ("CCN", b'P'), ("TCN", b'S'), ("ACN", b'T'), ("GTN", b'V')] {
            assert_eq!(t.get(c.as_bytes()), (aa, Rule::OneNResolved), "{c}");
        }
        assert_eq!(t.get(b"TTN"), (b'X', Rule::OneNAmbiguous));
        assert_eq!(t.get(b"AGN"), (b'X', Rule::OneNAmbiguous));
        assert_eq!(t.get(b"TAN"), (b'X', Rule::OneNCouldBeStop));
        assert_eq!(t.get(b"NNN"), (b'X', Rule::TwoOrMoreN));
        assert_eq!(t.get(b"TAA"), (b'*', Rule::NoN));
        assert_eq!(t.get(b"ATG"), (b'M', Rule::NoN));
    }

    #[test]
    fn rule_counts_match_the_python_table() {
        // codon_table_with_N.tsv: 8 resolved, 7 could be a stop, 33 ambiguous, 13 with 2+ N.
        let t = CodonTable::new();
        let mut n = [0; 5];
        for r in t.rule.iter() {
            n[*r as usize] += 1;
        }
        assert_eq!(n, [64, 8, 7, 33, 13]);
    }

    #[test]
    fn x_is_not_a_stop() {
        // 31 alanines, then TAN, then 31 more: TAN is X, so the piece runs through it.
        let t = CodonTable::new();
        let read = format!("{}TAN{}", "GCT".repeat(31), "GCT".repeat(31));
        let mut c = [0; 5];
        let kept = kept_translations(&t, read.as_bytes(), &mut c);
        let plus0 = kept.iter().find(|p| p.strand == b'+' && p.frame == 0).unwrap();
        assert_eq!(plus0.peptide.len(), 63);
        assert_eq!(plus0.peptide[31], b'X');
    }
}
