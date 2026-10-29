//! Fixed-memory Bloom membership index with self-describing, checksummed storage.
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Read, Write},
    path::Path,
};
/// Amino-acid grouping; class membership follows upstream Orpheum's published mappings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
pub enum Alphabet {
    Protein,
    Dayhoff,
    Hp,
}
impl Alphabet {
    /// Number of bits required to encode one residue class exactly.
    pub fn width(self) -> u8 {
        match self {
            Self::Protein => 5,
            Self::Dayhoff => 3,
            Self::Hp => 1,
        }
    }
    /// Encode one canonical amino acid. Stops and unknown residues reset the rolling word.
    pub fn encode(self, b: u8) -> Option<u64> {
        match self {
            Self::Protein => b"ACDEFGHIKLMNPQRSTVWY"
                .iter()
                .position(|&x| x == b)
                .map(|x| x as u64),
            Self::Dayhoff => [b"C".as_slice(), b"AGPST", b"DENQ", b"HKR", b"ILMV", b"FWY"]
                .iter()
                .position(|x| x.contains(&b))
                .map(|x| x as u64),
            Self::Hp => {
                if b"AFGILMPVWY".contains(&b) {
                    Some(0)
                } else if b"CDEHKNQRST".contains(&b) {
                    Some(1)
                } else {
                    None
                }
            }
        }
    }
}
/// Compute exact packed words; no word crosses a stop or unknown residue.
pub fn words(peptide: &[u8], alphabet: Alphabet, k: u8) -> Vec<u64> {
    let mask = (1u64 << (alphabet.width() * k)) - 1;
    let mut value = 0;
    let mut length = 0usize;
    let mut out = Vec::with_capacity(peptide.len());
    for &b in peptide {
        if let Some(code) = alphabet.encode(b) {
            value = ((value << alphabet.width()) | code) & mask;
            length += 1;
            if length >= usize::from(k) {
                out.push(value)
            }
        } else {
            length = 0;
            value = 0
        }
    }
    out
}
/// Metadata recording the format, settings, reference fingerprint and Bloom occupancy.
#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    /// Format version, currently 1. Not compatible with khmer or sourmash signature files.
    pub version: u32,
    /// Encoded alphabet.
    pub alphabet: Alphabet,
    /// Word length in amino-acid residues.
    pub k: u8,
    /// Bytes in the fixed-memory bit array.
    pub bytes: usize,
    /// Fraction of bit positions set; used to estimate false-positive probability.
    pub occupancy: f64,
    /// Approximate false-positive probability, occupancy to the fourth power.
    pub estimated_fpr: f64,
    /// SHA256 of canonical reference identifiers and sequences, separated by zero bytes.
    pub reference_sha256: String,
    /// SHA256 of the serialized bit array, used to detect corruption.
    pub bits_sha256: String,
    /// Number of reference sequence records.
    pub reference_records: u64,
}
/// Four-probe Bloom filter shared immutably across scoring threads.
pub struct Index {
    pub meta: Metadata,
    bits: Vec<u8>,
}
fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
impl Index {
    /// Allocate an empty fixed-size filter; k must fit within 63 packed bits.
    pub fn new(alphabet: Alphabet, k: u8, mib: usize) -> io::Result<Self> {
        if k == 0 || k.saturating_mul(alphabet.width()) > 63 || !(1..=4096).contains(&mib) {
            return Err(bad("Invalid word length or memory allowance"));
        }
        let bytes = mib * 1024 * 1024;
        Ok(Self {
            meta: Metadata {
                version: 1,
                alphabet,
                k,
                bytes,
                occupancy: 0.,
                estimated_fpr: 0.,
                reference_sha256: String::new(),
                bits_sha256: String::new(),
                reference_records: 0,
            },
            bits: vec![0; bytes],
        })
    }
    fn positions(&self, word: u64) -> [usize; 4] {
        let a = mix(word);
        let b = mix(word ^ 0x517cc1b727220a95) | 1;
        std::array::from_fn(|i| {
            a.wrapping_add((i as u64).wrapping_mul(b)) as usize % (self.bits.len() * 8)
        })
    }
    /// Insert an exact packed peptide word.
    pub fn insert(&mut self, word: u64) {
        for p in self.positions(word) {
            self.bits[p / 8] |= 1 << (p % 8)
        }
    }
    /// Test approximate membership. False positives are possible; false negatives are not.
    pub fn contains(&self, word: u64) -> bool {
        self.positions(word)
            .iter()
            .all(|&p| self.bits[p / 8] & (1 << (p % 8)) != 0)
    }
    /// Serialize without overwriting an existing index.
    pub fn save(&mut self, path: &Path) -> io::Result<()> {
        self.meta.occupancy = self.bits.iter().map(|x| x.count_ones() as u64).sum::<u64>() as f64
            / (self.bits.len() * 8) as f64;
        self.meta.estimated_fpr = self.meta.occupancy.powi(4);
        self.meta.bits_sha256 = format!("{:x}", Sha256::digest(&self.bits));
        let header = serde_json::to_vec(&self.meta)?;
        let mut f = BufWriter::new(File::options().write(true).create_new(true).open(path)?);
        f.write_all(b"ORPHRS01")?;
        f.write_all(&(header.len() as u32).to_le_bytes())?;
        f.write_all(&header)?;
        f.write_all(&self.bits)?;
        let mut checksum = Sha256::new();
        checksum.update(&header);
        checksum.update(&self.bits);
        f.write_all(&checksum.finalize())?;
        f.flush()
    }
    /// Load a checksummed index, rejecting incompatible or truncated files.
    pub fn load(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let size = file.metadata()?.len();
        let mut f = BufReader::new(file);
        let mut magic = [0; 8];
        f.read_exact(&mut magic)?;
        if &magic != b"ORPHRS01" {
            return Err(bad("Not an orpheum-rs index"));
        }
        let mut n = [0; 4];
        f.read_exact(&mut n)?;
        let n = u32::from_le_bytes(n) as usize;
        if n > 65536 {
            return Err(bad("Invalid index header length"));
        }
        let mut header = vec![0; n];
        f.read_exact(&mut header)?;
        let meta: Metadata = serde_json::from_slice(&header)?;
        if meta.version != 1
            || meta.bytes == 0
            || meta.k == 0
            || meta.k.saturating_mul(meta.alphabet.width()) > 63
            || size.checked_sub(44 + n as u64) != Some(meta.bytes as u64)
        {
            return Err(bad("Index metadata/length mismatch"));
        }
        let mut bits = vec![0; meta.bytes];
        f.read_exact(&mut bits)?;
        if format!("{:x}", Sha256::digest(&bits)) != meta.bits_sha256 {
            return Err(bad("Index checksum mismatch"));
        }
        let mut expected = [0; 32];
        f.read_exact(&mut expected)?;
        let mut checksum = Sha256::new();
        checksum.update(&header);
        checksum.update(&bits);
        if checksum.finalize()[..] != expected[..] {
            return Err(bad("Index metadata/data checksum mismatch"));
        }
        let occupancy = bits.iter().map(|x| u64::from(x.count_ones())).sum::<u64>() as f64
            / (bits.len() * 8) as f64;
        if (occupancy - meta.occupancy).abs() > 1e-12
            || (occupancy.powi(4) - meta.estimated_fpr).abs() > 1e-12
        {
            return Err(bad("Index occupancy metadata mismatch"));
        }
        Ok(Self { meta, bits })
    }
}
fn bad(s: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stops_break_words() {
        assert_eq!(words(b"AC*DE", Alphabet::Protein, 2).len(), 2);
        assert!(words(b"A*D", Alphabet::Protein, 2).is_empty());
    }
    #[test]
    fn membership() {
        let mut index = Index::new(Alphabet::Protein, 7, 1).unwrap();
        for x in 0..1000 {
            index.insert(x)
        }
        for x in 0..1000 {
            assert!(index.contains(x))
        }
    }
    #[test]
    fn mappings() {
        assert_eq!(
            Alphabet::Dayhoff.encode(b'A'),
            Alphabet::Dayhoff.encode(b'T')
        );
        assert_eq!(Alphabet::Hp.encode(b'C'), Some(1));
        assert_eq!(Alphabet::Protein.encode(b'*'), None);
    }
}
