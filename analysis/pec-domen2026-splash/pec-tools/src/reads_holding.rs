//! reads-holding: the reads that hold the k-mers of a set of contigs, so that a route's short
//! contigs can be replaced with reads, which are up to 150 bases and cross forks in a k-mer table.
//!
//!     pec-tools reads-holding <contigs.tsv> <input.txt> <out_prefix> [k] [max_reads]
//!
//! contigs.tsv: tab-separated with a header holding `contig` and `direction`. input.txt: no
//! header; sample, interleaved FASTQ read through `pigz -dc`. Both mates are read as single reads
//! and both strands are matched. k defaults to 31.
//!
//! A contig takes at most `max_reads` (default 100) distinct reads from each sample, the first in
//! file order, so that a run of A's or a repeat does not pull in every read that holds it. A read
//! and its reverse complement are one distinct read.
//!
//! Writes <out_prefix>.reads.tsv (read, direction, contigs; one row per read and direction, so a
//! read holding k-mers of an up and a down contig has two rows), and <out_prefix>.contigs.tsv
//! (contig, direction, reads_holding, reads_kept). A k-mer in two contigs stays with the first.
use std::collections::HashMap;
use std::fs::File;
use std::hash::BuildHasherDefault;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};

use crate::best_frame::MulHasher;
use crate::{base, revcomp, Res};

type Build = BuildHasherDefault<MulHasher>;

/// Calls `f` with the smaller of each `k`-mer of `seq` and its reverse complement, as 2 bits per
/// base. A base other than ACGT restarts the window. `k` is at most 31.
fn for_each_canonical(seq: &[u8], k: usize, mut f: impl FnMut(u64)) {
    let mask = (1u64 << (2 * k)) - 1;
    let shift = 2 * (k - 1);
    let (mut fwd, mut rev, mut run) = (0u64, 0u64, 0usize);
    for &c in seq {
        match base(c) {
            Some(b) => {
                fwd = ((fwd << 2) | b as u64) & mask;
                rev = (rev >> 2) | ((3 - b as u64) << shift);
                run += 1;
                if run >= k {
                    f(fwd.min(rev));
                }
            }
            None => run = 0,
        }
    }
}

struct Contigs {
    seqs: Vec<String>,
    directions: Vec<String>,
    index: HashMap<u64, u32, Build>,
    shared_kmers: usize,
}

fn read_contigs(path: &str, k: usize) -> Res<Contigs> {
    let mut lines = BufReader::new(File::open(path)?).lines();
    let header = lines.next().ok_or("empty contig file")??;
    let cols: Vec<&str> = header.split('\t').collect();
    let col = |name: &str| cols.iter().position(|c| *c == name).ok_or(format!("no column {name}"));
    let (cs, cd) = (col("contig")?, col("direction")?);
    let mut c = Contigs { seqs: Vec::new(), directions: Vec::new(), index: HashMap::default(), shared_kmers: 0 };
    for line in lines {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        let i = c.seqs.len() as u32;
        let (index, shared) = (&mut c.index, &mut c.shared_kmers);
        for_each_canonical(f[cs].as_bytes(), k, |code| {
            let e = index.entry(code).or_insert(i);
            if *e != i {
                *shared += 1;
            }
        });
        c.seqs.push(f[cs].to_string());
        c.directions.push(f[cd].to_string());
    }
    Ok(c)
}

/// Reads kept so far, by sequence, each with the contigs it was kept for.
#[derive(Default)]
struct Kept {
    reads: HashMap<Vec<u8>, Vec<u32>>,
    /// reads holding each contig, kept or not, one count per mate
    holding: Vec<u64>,
    kept_per_contig: Vec<u64>,
}

impl Kept {
    /// One read: finds its contigs, and links it to those that have room in this sample.
    fn add(&mut self, c: &Contigs, read: &[u8], k: usize, max_reads: u64, in_sample: &mut [u64], hit: &mut Vec<u32>) {
        hit.clear();
        for_each_canonical(read, k, |code| {
            if let Some(&i) = c.index.get(&code) {
                if !hit.contains(&i) {
                    hit.push(i);
                }
            }
        });
        if hit.is_empty() {
            return;
        }
        let rc = revcomp(read);
        let key = if rc.as_slice() < read { rc } else { read.to_vec() };
        for &i in hit.iter() {
            self.holding[i as usize] += 1;
            let linked = self.reads.get(&key).is_some_and(|v| v.contains(&i));
            if !linked && in_sample[i as usize] < max_reads {
                in_sample[i as usize] += 1;
                self.kept_per_contig[i as usize] += 1;
                self.reads.entry(key.clone()).or_default().push(i);
            }
        }
    }
}

pub fn cmd_reads_holding(args: &[String]) -> Res<()> {
    if args.len() < 3 {
        return Err("usage: pec-tools reads-holding <contigs.tsv> <input.txt> <out_prefix> [k] [max_reads]".into());
    }
    let k: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(31);
    let max_reads: u64 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(100);
    if k == 0 || k > 31 {
        return Err("k must be 1 to 31".into());
    }
    let c = read_contigs(&args[0], k)?;
    eprintln!("contigs {}  k-mers {}  in two contigs {}", c.seqs.len(), c.index.len(), c.shared_kmers);
    let mut kept = Kept { holding: vec![0; c.seqs.len()], kept_per_contig: vec![0; c.seqs.len()], ..Default::default() };
    let mut hit = Vec::new();
    for line in BufReader::new(File::open(&args[1])?).lines() {
        let line = line?;
        let (sample, path) = line.split_once('\t').or_else(|| line.split_once(' ')).ok_or("input.txt: need sample and path")?;
        let mut child = Command::new("pigz").args(["-dc", path.trim()]).stdout(Stdio::piped()).spawn()?;
        let mut r = BufReader::with_capacity(1 << 20, child.stdout.take().ok_or("no pigz stdout")?);
        let mut in_sample = vec![0u64; c.seqs.len()];
        let (mut line, mut n) = (Vec::new(), 0u64);
        loop {
            line.clear();
            if r.read_until(b'\n', &mut line)? == 0 {
                break;
            }
            if n % 4 == 1 {
                while line.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
                    line.pop();
                }
                kept.add(&c, &line, k, max_reads, &mut in_sample, &mut hit);
            }
            n += 1;
        }
        if !child.wait()?.success() {
            return Err(format!("pigz failed on {path}").into());
        }
        if n % 4 != 0 {
            return Err(format!("{path}: line count {n} is not a multiple of 4").into());
        }
        eprintln!("{sample}: reads {}  distinct reads kept so far {}", n / 4, kept.reads.len());
    }

    let mut rows: Vec<(&Vec<u8>, &Vec<u32>)> = kept.reads.iter().collect();
    rows.sort_unstable_by(|a, b| a.0.cmp(b.0)); // HashMap order is not fixed; a rerun gives the same IDs
    let mut out = BufWriter::new(File::create(format!("{}.reads.tsv", args[2]))?);
    writeln!(out, "read\tdirection\tcontigs")?;
    let mut both = 0usize;
    for (seq, ids) in rows {
        let mut dirs: Vec<&str> = ids.iter().map(|&i| c.directions[i as usize].as_str()).collect();
        dirs.sort_unstable();
        dirs.dedup();
        both += usize::from(dirs.len() > 1);
        for d in dirs {
            let n = ids.iter().filter(|&&i| c.directions[i as usize] == d).count();
            writeln!(out, "{}\t{d}\t{n}", String::from_utf8_lossy(seq))?;
        }
    }
    out.flush()?;
    let mut out = BufWriter::new(File::create(format!("{}.contigs.tsv", args[2]))?);
    writeln!(out, "contig\tdirection\treads_holding\treads_kept")?;
    for i in 0..c.seqs.len() {
        writeln!(out, "{}\t{}\t{}\t{}", c.seqs[i], c.directions[i], kept.holding[i], kept.kept_per_contig[i])?;
    }
    out.flush()?;
    eprintln!("distinct reads {}  holding both directions {both}", kept.reads.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canon(s: &[u8], k: usize) -> Vec<u64> {
        let mut v = Vec::new();
        for_each_canonical(s, k, |c| v.push(c));
        v
    }

    fn contigs(rows: &[(&str, &str)], k: usize) -> Contigs {
        let p = std::env::temp_dir().join(format!("reads_holding_{}_{:?}.tsv", std::process::id(), std::thread::current().id()));
        let body: String = rows.iter().map(|(s, d)| format!("{s}\t{d}\n")).collect();
        std::fs::write(&p, format!("contig\tdirection\n{body}")).unwrap();
        read_contigs(p.to_str().unwrap(), k).unwrap()
    }

    fn new_kept(c: &Contigs) -> Kept {
        Kept { holding: vec![0; c.seqs.len()], kept_per_contig: vec![0; c.seqs.len()], ..Default::default() }
    }

    #[test]
    fn canonical_kmers_match_across_strands() {
        let s = b"ACGTTGCAAGGCTTAACCGGA";
        let mut a = canon(s, 7);
        let mut b = canon(&revcomp(s), 7);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
        assert_eq!(a.len(), s.len() - 6);
    }

    #[test]
    fn n_restarts_the_window() {
        assert_eq!(canon(b"ACGTNACGT", 5).len(), 0);
        assert_eq!(canon(b"ACGTANACGTA", 5).len(), 2);
    }

    #[test]
    fn full_length_31_mers_do_not_overflow() {
        let s = b"ACGTTGCAAGGCTTAACCGGATTACAGGCATCGA";
        let mut a = canon(s, 31);
        let mut b = canon(&revcomp(s), 31);
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b);
    }

    #[test]
    fn a_read_on_the_other_strand_is_found_and_collapsed() {
        let contig = "ACGTTGCAAGGCTTAACCGGATTACAGG";
        let c = contigs(&[(contig, "up")], 11);
        let mut kept = new_kept(&c);
        let (mut room, mut hit) = (vec![0; 1], Vec::new());
        let read = format!("TTTT{contig}CCCC");
        kept.add(&c, read.as_bytes(), 11, 100, &mut room, &mut hit);
        kept.add(&c, &revcomp(read.as_bytes()), 11, 100, &mut room, &mut hit);
        assert_eq!(kept.reads.len(), 1);
        assert_eq!(kept.holding[0], 2);
        assert_eq!(kept.kept_per_contig[0], 1);
    }

    #[test]
    fn a_contig_stops_taking_reads_at_the_limit() {
        let contig = "ACGTTGCAAGGCTTA";
        let c = contigs(&[(contig, "up")], 11);
        let mut kept = new_kept(&c);
        let (mut room, mut hit) = (vec![0; 1], Vec::new());
        for tail in ["AAAA", "CCCC", "GGGG", "TTTT"] {
            kept.add(&c, format!("{contig}{tail}").as_bytes(), 11, 2, &mut room, &mut hit);
        }
        assert_eq!(kept.reads.len(), 2);
        assert_eq!(kept.holding[0], 4);
    }

    #[test]
    fn a_read_holding_two_contigs_links_to_both() {
        let (a, b) = ("ACGTTGCAAGGCTTA", "GGATCCTTGACTCAT");
        let c = contigs(&[(a, "up"), (b, "down")], 11);
        let mut kept = new_kept(&c);
        let (mut room, mut hit) = (vec![0; 2], Vec::new());
        kept.add(&c, format!("{a}TTTT{b}").as_bytes(), 11, 100, &mut room, &mut hit);
        assert_eq!(kept.reads.len(), 1);
        assert_eq!(kept.reads.values().next().unwrap().len(), 2);
    }

    #[test]
    fn a_read_with_no_contig_k_mer_is_not_kept() {
        let c = contigs(&[("ACGTTGCAAGGCTTA", "up")], 11);
        let mut kept = new_kept(&c);
        let (mut room, mut hit) = (vec![0; 1], Vec::new());
        kept.add(&c, b"TTTTTTTTTTTTTTTTTTTT", 11, 100, &mut room, &mut hit);
        assert!(kept.reads.is_empty());
    }
}
