//! best-frame: for each read, translate all six frames, count how many of each frame's
//! k-mers occur in a reference protein set (Swiss-Prot), and keep only the frame with the
//! most matches. Rule `stopfree` first drops frames that contain a stop codon, as orpheum
//! translate (Botvinnik et al.) does; rule `any` lets every frame compete. Written as a
//! compiled loop because orpheum's per-read Python is too slow for ~10^9 reads.
//!
//! A frame's k-mers are every window of `k` residues; a window that contains a stop (`*`) or
//! an unknown residue (`X`) cannot match and still counts in the denominator, so
//! `fraction = matches / windows`. Ties between frames are broken by the frame order
//! (+1, +2, +3, -1, -2, -3) and counted. Reads shorter than `k` codons have no windows.
//!
//! Output: for every cutoff `c`, the chosen frame of each read with `fraction >= c` is written
//! to `<prefix>.ge<c>.faa`, split at stops into pieces of at least `min_piece` residues.
//! `<prefix>.hist.tsv` counts reads by (windows, matches) of the chosen frame, and
//! `<prefix>.summary.tsv` counts reads at each step.
use std::collections::HashSet;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};

use crate::{revcomp, translate_frame, Res};

/// Multiplicative hasher for already well-spread u64 k-mer codes (SipHash is 5x slower here).
#[derive(Default)]
pub struct MulHasher(u64);
impl Hasher for MulHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    fn write_u64(&mut self, x: u64) {
        let h = (x ^ (x >> 31)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.0 = h ^ (h >> 29);
    }
}
pub type KmerSet = HashSet<u64, BuildHasherDefault<MulHasher>>;

/// Letter code of one amino acid in a reduced alphabet; `None` for stops and unknowns.
/// Groups follow sourmash's `hp` and `dayhoff` encodings.
pub fn aa_code(alphabet: &str, aa: u8) -> Option<u64> {
    let aa = aa.to_ascii_uppercase();
    match alphabet {
        "protein" => b"ACDEFGHIKLMNPQRSTVWY"
            .iter()
            .position(|&c| c == aa)
            .map(|i| i as u64),
        "dayhoff" => match aa {
            b'C' => Some(0),
            b'A' | b'G' | b'P' | b'S' | b'T' => Some(1),
            b'D' | b'E' | b'N' | b'Q' => Some(2),
            b'H' | b'K' | b'R' => Some(3),
            b'I' | b'L' | b'M' | b'V' => Some(4),
            b'F' | b'W' | b'Y' => Some(5),
            _ => None,
        },
        "hp" => match aa {
            b'A' | b'F' | b'G' | b'I' | b'L' | b'M' | b'P' | b'V' | b'W' | b'Y' => Some(0),
            b'C' | b'D' | b'E' | b'H' | b'K' | b'N' | b'Q' | b'R' | b'S' | b'T' => Some(1),
            _ => None,
        },
        _ => panic!("alphabet must be protein, dayhoff or hp"),
    }
}

pub fn alphabet_size(alphabet: &str) -> u64 {
    match alphabet {
        "protein" => 20,
        "dayhoff" => 6,
        "hp" => 2,
        _ => panic!("alphabet must be protein, dayhoff or hp"),
    }
}

/// Calls `f(code)` for every window of `k` valid letters; returns the number of windows
/// (valid or not). Codes are base-`size` numbers of the window's letters.
pub fn for_each_kmer(seq: &[u8], alphabet: &str, k: usize, mut f: impl FnMut(u64)) -> usize {
    if seq.len() < k {
        return 0;
    }
    let size = alphabet_size(alphabet);
    let top = size.pow(k as u32 - 1);
    let (mut code, mut run) = (0u64, 0usize);
    for &aa in seq {
        match aa_code(alphabet, aa) {
            Some(c) => {
                code = if run >= k {
                    (code % top) * size + c
                } else {
                    code * size + c
                };
                run += 1;
                if run >= k {
                    f(code);
                }
            }
            None => {
                code = 0;
                run = 0;
            }
        }
    }
    seq.len() - k + 1
}

pub fn load_reference(path: &str, alphabet: &str, k: usize) -> Res<KmerSet> {
    let mut set = KmerSet::default();
    let mut seq = Vec::new();
    let add = |s: &mut Vec<u8>, set: &mut KmerSet| {
        for_each_kmer(s, alphabet, k, |c| {
            set.insert(c);
        });
        s.clear();
    };
    for line in BufReader::with_capacity(1 << 20, File::open(path)?).lines() {
        let line = line?;
        if line.starts_with('>') {
            add(&mut seq, &mut set);
        } else {
            seq.extend(line.trim().bytes());
        }
    }
    add(&mut seq, &mut set);
    Ok(set)
}

/// Score of one frame: (matches, windows).
pub fn score(aa: &[u8], set: &KmerSet, alphabet: &str, k: usize) -> (usize, usize) {
    let mut m = 0;
    let w = for_each_kmer(aa, alphabet, k, |c| {
        if set.contains(&c) {
            m += 1;
        }
    });
    (m, w)
}

/// The six translated frames in order +1, +2, +3, -1, -2, -3.
pub fn six_frames(dna: &[u8]) -> Vec<Vec<u8>> {
    let rc = revcomp(dna);
    let mut out = Vec::with_capacity(6);
    for s in [dna, &rc[..]] {
        for off in 0..3 {
            out.push(if s.len() > off {
                translate_frame(&s[off..])
            } else {
                Vec::new()
            });
        }
    }
    out
}

/// Which frames may be chosen: `Any` = all six; `StopFree` = only frames with no stop codon,
/// as orpheum translate does (a read inside a coding sequence has no stop in its true frame).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Rule {
    Any,
    StopFree,
}

impl std::str::FromStr for Rule {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "any" => Ok(Rule::Any),
            "stopfree" => Ok(Rule::StopFree),
            _ => Err(format!("rule must be any or stopfree, got {s}")),
        }
    }
}

/// Among the frames the rule allows, the one with most matches (first in frame order on ties):
/// (index, matches, windows, frames tied). `None` when the rule allows no frame (StopFree and
/// every frame has a stop).
pub fn best_frame(
    frames: &[Vec<u8>],
    set: &KmerSet,
    alphabet: &str,
    k: usize,
    rule: Rule,
) -> Option<(usize, usize, usize, usize)> {
    let allowed: Vec<usize> = (0..frames.len())
        .filter(|&i| rule == Rule::Any || !frames[i].contains(&b'*'))
        .collect();
    let scores: Vec<(usize, (usize, usize))> = allowed
        .iter()
        .map(|&i| (i, score(&frames[i], set, alphabet, k)))
        .collect();
    let mut best = *scores.first()?;
    for s in &scores[1..] {
        if s.1 .0 > best.1 .0 {
            best = *s;
        }
    }
    let ties = scores.iter().filter(|s| s.1 .0 == best.1 .0).count();
    Some((best.0, best.1 .0, best.1 .1, ties))
}

/// frame-scores <ref.faa> <alphabet> <k> < reads.fastq > scores.tsv
/// One row per read with the frame each rule chooses (-1: none allowed), the windows of frame +1,
/// and per frame the matching k-mers and stop codons, for testing on reads of known frame.
pub fn cmd_frame_scores(args: &[String]) -> Res<()> {
    if args.len() < 3 {
        return Err("usage: pec-tools frame-scores <ref.faa> <protein|dayhoff|hp> <k> < reads.fastq > scores.tsv".into());
    }
    let (refp, alphabet, k) = (&args[0], args[1].as_str(), args[2].parse::<usize>()?);
    let t0 = std::time::Instant::now();
    let set = load_reference(refp, alphabet, k)?;
    eprintln!(
        "reference {alphabet} k={k}: {} distinct k-mers in {:.1} s",
        set.len(),
        t0.elapsed().as_secs_f64()
    );
    let stdout = std::io::stdout();
    let mut out = BufWriter::with_capacity(1 << 20, stdout.lock());
    writeln!(
        out,
        "read_id\tread_len\tbest_any\tbest_stopfree\twindows\tm0\tm1\tm2\tm3\tm4\tm5\tstops0\tstops1\tstops2\tstops3\tstops4\tstops5"
    )?;
    let mut lines = BufReader::with_capacity(1 << 22, std::io::stdin().lock()).lines();
    let mut n = 0u64;
    while let Some(h) = lines.next() {
        let h = h?;
        if h.is_empty() {
            continue;
        }
        let id = h
            .strip_prefix('@')
            .ok_or("expected FASTQ header")?
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_string();
        let seq = lines.next().ok_or("truncated FASTQ")??;
        lines.next().ok_or("truncated FASTQ")??;
        lines.next().ok_or("truncated FASTQ")??;
        let frames = six_frames(seq.as_bytes());
        let any = best_frame(&frames, &set, alphabet, k, Rule::Any).map_or(-1, |b| b.0 as i64);
        let sf = best_frame(&frames, &set, alphabet, k, Rule::StopFree).map_or(-1, |b| b.0 as i64);
        let w = frames.first().map_or(0, |f| f.len().saturating_sub(k - 1));
        let all: Vec<String> = frames
            .iter()
            .map(|f| score(f, &set, alphabet, k).0.to_string())
            .collect();
        let stops: Vec<String> = frames
            .iter()
            .map(|f| f.iter().filter(|&&a| a == b'*').count().to_string())
            .collect();
        writeln!(
            out,
            "{id}\t{}\t{any}\t{sf}\t{w}\t{}\t{}",
            seq.len(),
            all.join("\t"),
            stops.join("\t")
        )?;
        n += 1;
    }
    out.flush()?;
    eprintln!("reads {n} in {:.1} s", t0.elapsed().as_secs_f64());
    Ok(())
}

pub fn cmd_best_frame(args: &[String]) -> Res<()> {
    // best-frame <ref.faa> <alphabet> <k> <out_prefix> <cutoffs,comma> <any|stopfree> [min_piece=6]  (FASTQ on stdin)
    if args.len() < 6 {
        return Err("usage: pec-tools best-frame <ref.faa> <protein|dayhoff|hp> <k> <out_prefix> <cutoff[,cutoff..]> <any|stopfree> [min_piece] < reads.fastq".into());
    }
    let rule: Rule = args[5].parse()?;
    let (refp, alphabet, k, prefix) = (
        &args[0],
        args[1].as_str(),
        args[2].parse::<usize>()?,
        &args[3],
    );
    let cutoffs: Vec<f64> = args[4]
        .split(',')
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let min_piece: usize = args.get(6).map(|s| s.parse()).transpose()?.unwrap_or(6);
    let t0 = std::time::Instant::now();
    let set = load_reference(refp, alphabet, k)?;
    eprintln!(
        "reference {alphabet} k={k}: {} distinct k-mers in {:.1} s",
        set.len(),
        t0.elapsed().as_secs_f64()
    );
    let mut outs: Vec<BufWriter<File>> = cutoffs
        .iter()
        .map(|c| {
            File::create(format!("{prefix}.ge{c}.faa"))
                .map(|f| BufWriter::with_capacity(1 << 20, f))
        })
        .collect::<Result<_, _>>()?;
    let mut kept = vec![0u64; cutoffs.len()];
    let mut hist: std::collections::HashMap<(usize, usize), u64> = Default::default();
    let (mut n, mut no_window, mut zero, mut tied, mut bases, mut no_frame) =
        (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let stdin = std::io::stdin();
    let mut lines = BufReader::with_capacity(1 << 22, stdin.lock()).lines();
    while let Some(h) = lines.next() {
        let h = h?;
        if h.is_empty() {
            continue;
        }
        if !h.starts_with('@') {
            return Err(format!("expected FASTQ header, got {h:.40}").into());
        }
        let seq = lines.next().ok_or("truncated FASTQ")??;
        lines.next().ok_or("truncated FASTQ")??;
        lines.next().ok_or("truncated FASTQ")??;
        n += 1;
        bases += seq.len() as u64;
        let frames = six_frames(seq.as_bytes());
        let Some((bi, m, w, ties)) = best_frame(&frames, &set, alphabet, k, rule) else {
            no_frame += 1;
            continue;
        };
        *hist.entry((w, m)).or_default() += 1;
        if w == 0 {
            no_window += 1;
            continue;
        }
        if m == 0 {
            zero += 1;
            continue;
        }
        if ties > 1 {
            tied += 1;
        }
        let frac = m as f64 / w as f64;
        for (j, c) in cutoffs.iter().enumerate() {
            if frac >= *c {
                kept[j] += 1;
                for (p, piece) in frames[bi].split(|&a| a == b'*').enumerate() {
                    if piece.len() >= min_piece {
                        writeln!(outs[j], ">r{n}.{p}")?;
                        outs[j].write_all(piece)?;
                        outs[j].write_all(b"\n")?;
                    }
                }
            }
        }
    }
    for o in outs.iter_mut() {
        o.flush()?;
    }
    let mut hs = BufWriter::new(File::create(format!("{prefix}.hist.tsv"))?);
    writeln!(hs, "windows\tmatches\treads")?;
    let mut keys: Vec<_> = hist.keys().copied().collect();
    keys.sort();
    for key in keys {
        writeln!(hs, "{}\t{}\t{}", key.0, key.1, hist[&key])?;
    }
    let mut sm = BufWriter::new(File::create(format!("{prefix}.summary.tsv"))?);
    writeln!(sm, "step\treads")?;
    writeln!(sm, "all_reads\t{n}")?;
    writeln!(sm, "bases\t{bases}")?;
    writeln!(sm, "no_allowed_frame\t{no_frame}")?;
    writeln!(sm, "no_window\t{no_window}")?;
    writeln!(sm, "zero_match\t{zero}")?;
    writeln!(sm, "any_match\t{}", n - no_frame - no_window - zero)?;
    writeln!(sm, "any_match_frame_tied\t{tied}")?;
    for (j, c) in cutoffs.iter().enumerate() {
        writeln!(sm, "fraction_ge_{c}\t{}", kept[j])?;
    }
    eprintln!(
        "reads {n}  no allowed frame {no_frame}  any match {}  kept {:?}  in {:.1} s",
        n - no_frame - no_window - zero,
        kept,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set_of(seqs: &[&[u8]], a: &str, k: usize) -> KmerSet {
        let mut s = KmerSet::default();
        for q in seqs {
            for_each_kmer(q, a, k, |c| {
                s.insert(c);
            });
        }
        s
    }

    #[test]
    fn kmer_windows_break_at_stops_but_count_in_denominator() {
        let mut n = 0;
        let w = for_each_kmer(b"MKV*LLA", "protein", 3, |_| n += 1);
        assert_eq!(w, 5); // MKV KV* V*L *LL LLA
        assert_eq!(n, 2); // MKV, LLA
    }

    #[test]
    fn rolling_code_equals_direct_code() {
        let seq = b"ACDEFGHIK";
        let mut rolled = vec![];
        for_each_kmer(seq, "protein", 4, |c| rolled.push(c));
        let direct: Vec<u64> = seq
            .windows(4)
            .map(|w| {
                w.iter()
                    .fold(0, |acc, &a| acc * 20 + aa_code("protein", a).unwrap())
            })
            .collect();
        assert_eq!(rolled, direct);
    }

    #[test]
    fn picks_the_frame_that_matches_the_reference() {
        // MAEGKLLWRPSTQ encoded by this DNA in frame +1; reference holds that protein only.
        let dna = b"ATGGCTGAAGGTAAACTGCTGTGGCGTCCGAGCACCCAG";
        let set = set_of(&[b"MAEGKLLWRPSTQ"], "protein", 5);
        let frames = six_frames(dna);
        let (bi, m, w, ties) = best_frame(&frames, &set, "protein", 5, Rule::Any).unwrap();
        assert_eq!((bi, m, w, ties), (0, 9, 9, 1));
        // the same read reverse-complemented is found on frame -1 (index 3)
        let rc = revcomp(dna);
        let (bi, m, _, _) = best_frame(&six_frames(&rc), &set, "protein", 5, Rule::Any).unwrap();
        assert_eq!((bi, m), (3, 9));
        assert_eq!(
            best_frame(&frames, &set, "protein", 5, Rule::StopFree)
                .unwrap()
                .0,
            0
        );
    }

    #[test]
    fn stop_free_rule_skips_frames_with_stops() {
        // frame +1 carries a stop but matches more; the stop-free rule must pick another frame
        let frames = vec![b"MAEGK*LLW".to_vec(), b"PSTQRPST".to_vec()];
        let set = set_of(&[b"MAEGK", b"LLWAA", b"PSTQR"], "protein", 3);
        assert_eq!(
            best_frame(&frames, &set, "protein", 3, Rule::Any)
                .unwrap()
                .0,
            0
        );
        assert_eq!(
            best_frame(&frames, &set, "protein", 3, Rule::StopFree)
                .unwrap()
                .0,
            1
        );
        let all_stop = vec![b"MA*".to_vec(), b"*KL".to_vec()];
        assert!(best_frame(&all_stop, &set, "protein", 3, Rule::StopFree).is_none());
    }

    #[test]
    fn dayhoff_matches_conservative_substitution() {
        // L->I is the same Dayhoff letter, so the k-mer still matches.
        let set = set_of(&[b"MAEGKLLW"], "dayhoff", 6);
        assert_eq!(score(b"MAEGKILW", &set, "dayhoff", 6), (3, 3));
        assert_eq!(
            score(
                b"MAEGKILW",
                &set_of(&[b"MAEGKLLW"], "protein", 6),
                "protein",
                6
            )
            .0,
            0
        );
    }
}
