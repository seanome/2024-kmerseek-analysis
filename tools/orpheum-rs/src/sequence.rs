//! Bounded, streaming FASTA/four-line FASTQ input and standard-code translation.
use flate2::read::MultiGzDecoder;
use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read},
    path::Path,
};
/// A sequence record; the identifier is the first whitespace-delimited header field.
#[derive(Debug)]
pub struct Record {
    pub id: String,
    pub sequence: Vec<u8>,
}
/// Streaming parser supporting plain or gzip input, including concatenated gzip members.
pub struct Records {
    reader: Box<dyn BufRead>,
    pending: Option<String>,
}
impl Records {
    /// Open a path, or standard input for `-`. Compression is detected from magic bytes.
    pub fn open(path: &Path) -> io::Result<Self> {
        let input: Box<dyn Read> = if path.as_os_str() == "-" {
            Box::new(io::stdin())
        } else {
            Box::new(File::open(path)?)
        };
        let mut buffered = BufReader::new(input);
        let gzip = buffered.fill_buf()?.starts_with(&[0x1f, 0x8b]);
        let reader: Box<dyn BufRead> = if gzip {
            Box::new(BufReader::new(MultiGzDecoder::new(buffered)))
        } else {
            Box::new(buffered)
        };
        Ok(Self {
            reader,
            pending: None,
        })
    }
    /// Read one record; malformed/truncated records are errors rather than empty results.
    pub fn next_record(&mut self) -> io::Result<Option<Record>> {
        let mut header = if let Some(h) = self.pending.take() {
            h
        } else {
            let mut h = String::new();
            if self.reader.read_line(&mut h)? == 0 {
                return Ok(None);
            }
            h
        };
        while header.trim().is_empty() {
            header.clear();
            if self.reader.read_line(&mut header)? == 0 {
                return Ok(None);
            }
        }
        let marker = header.as_bytes()[0];
        if marker != b'>' && marker != b'@' {
            return Err(invalid("Expected FASTA or FASTQ header"));
        }
        let id = header[1..]
            .split_whitespace()
            .next()
            .ok_or_else(|| invalid("Empty sequence identifier"))?
            .to_string();
        let mut sequence = Vec::new();
        let mut line = String::new();
        if marker == b'>' {
            loop {
                line.clear();
                if self.reader.read_line(&mut line)? == 0 {
                    break;
                }
                if line.starts_with('>') {
                    self.pending = Some(line);
                    break;
                }
                sequence.extend(
                    line.split_whitespace()
                        .flat_map(str::bytes)
                        .map(|b| b.to_ascii_uppercase()),
                );
            }
        } else {
            if self.reader.read_line(&mut line)? == 0 {
                return Err(invalid("Truncated FASTQ sequence"));
            }
            sequence.extend(line.trim_end().bytes().map(|b| b.to_ascii_uppercase()));
            line.clear();
            if self.reader.read_line(&mut line)? == 0 || !line.starts_with('+') {
                return Err(invalid(
                    "Expected FASTQ separator; wrapped FASTQ is unsupported",
                ));
            }
            line.clear();
            if self.reader.read_line(&mut line)? == 0 || line.trim_end().len() != sequence.len() {
                return Err(invalid("FASTQ sequence/quality length mismatch"));
            }
        }
        if sequence.is_empty() {
            return Err(invalid("Empty sequence"));
        }
        Ok(Some(Record { id, sequence }))
    }
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
/// Standard nuclear code, indexed in T,C,A,G order at each codon position.
const CODE: &[u8; 64] = b"FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG";
fn base(b: u8) -> Option<usize> {
    match b {
        b'T' | b'U' => Some(0),
        b'C' => Some(1),
        b'A' => Some(2),
        b'G' => Some(3),
        _ => None,
    }
}
/// Translate one codon. Ambiguous codons are `X`; no genetic code is inferred.
pub fn codon(c: &[u8]) -> u8 {
    if let (Some(a), Some(b), Some(d)) = (base(c[0]), base(c[1]), base(c[2])) {
        CODE[a * 16 + b * 4 + d]
    } else {
        b'X'
    }
}
/// Reverse complement; unsupported/ambiguous bases become `N`.
pub fn reverse_complement(dna: &[u8]) -> Vec<u8> {
    dna.iter()
        .rev()
        .map(|b| match b {
            b'A' => b'T',
            b'T' | b'U' => b'A',
            b'G' => b'C',
            b'C' => b'G',
            _ => b'N',
        })
        .collect()
}
/// Return six independent translations, forward offsets 0–2 then reverse offsets 0–2.
pub fn six_frames(dna: &[u8]) -> [Vec<u8>; 6] {
    let reverse = reverse_complement(dna);
    std::array::from_fn(|i| {
        let strand = if i < 3 { dna } else { &reverse };
        strand
            .get(i % 3..)
            .unwrap_or_default()
            .chunks_exact(3)
            .map(codon)
            .collect()
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn codons() {
        assert_eq!(codon(b"ATG"), b'M');
        for c in [b"TAA", b"TAG", b"TGA"] {
            assert_eq!(codon(c), b'*')
        }
        assert_eq!(codon(b"NNN"), b'X');
        assert_eq!(codon(b"GCT"), b'A');
    }
    #[test]
    fn six_translation() {
        let f = six_frames(b"ATGTAAGCT");
        assert_eq!(f[0], b"M*A");
        assert_eq!(f[1], b"CK");
        assert_eq!(f[2], b"VS");
        assert_eq!(f[3], b"SLH");
    }
}
