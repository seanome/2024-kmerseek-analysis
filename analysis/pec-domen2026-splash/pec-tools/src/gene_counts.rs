//! gene-counts: reads per gene per sample, from protein k-mers, with no alignment and no genome.
//!
//! Each read pair is translated in six frames, and every window of `k` residues is looked up in
//! an index of k-mers from a protein set. A k-mer found in two or more genes is left out. A pair
//! goes to the gene with the most matching windows over both mates, if that is at least
//! `min_hits`; a tie between genes counts the pair for neither. With `scaled` above 1, only k-mers
//! whose hash falls in 1/`scaled` of the hash range are indexed and looked up, which trades
//! sensitivity for memory and time.
//!
//! Random matches: the old gene-model proteins hold about 17 M residues. At k = 7 about 1.3 % of
//! random windows would match, about 3 per read pair, so k = 10 (about 2 in a million) is the
//! default, and 3 hits are asked for so that one chance match cannot place a pair.
//!
//!     pec-tools gene-counts <proteins.faa> <gene_map.tsv> <manifest.tsv> <out_prefix> <threads> [k] [min_hits] [scaled]
//!
//! gene_map.tsv: no header; protein ID (the FASTA header's first word), gene. Proteins not listed
//! are skipped. manifest.tsv: no header; sample, R1 path, R2 path, largest number of pairs to read
//! (0 = all), in file order, so a limit reads only the start of each file. For single-end reads
//! the R2 path is `-`, and each read is counted as a pair with one mate. Files are read through
//! `pigz -dc`. Writes <out_prefix>.counts.tsv (gene, one column per sample; genes with any count)
//! and <out_prefix>.summary.tsv (sample, pairs, no_hit, tied, assigned).
use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::best_frame::{aa_code, for_each_kmer, MulHasher};
use crate::{base, Res, CODE};

const SHARED: u32 = u32::MAX; // k-mer found in two or more genes
const STOP: u8 = 255;
type Build = BuildHasherDefault<MulHasher>;

pub struct Index {
    kmers: HashMap<u64, u32, Build>,
    pub genes: Vec<String>,
    k: usize,
    scaled: u64,
    /// residue code (0..20, or STOP) of each codon, index as in `crate::CODE`
    codon: [u8; 64],
}

fn keep(code: u64, scaled: u64) -> bool {
    if scaled <= 1 {
        return true;
    }
    let mut h = Build::default().build_hasher();
    code.hash(&mut h);
    h.finish() % scaled == 0
}

impl Index {
    pub fn new(k: usize, scaled: u64) -> Self {
        let mut codon = [STOP; 64];
        for (i, &aa) in CODE.iter().enumerate() {
            if let Some(c) = aa_code("protein", aa) {
                codon[i] = c as u8;
            }
        }
        Index { kmers: HashMap::default(), genes: Vec::new(), k, scaled, codon }
    }

    fn add(&mut self, seq: &[u8], gene: u32) {
        let (kmers, scaled) = (&mut self.kmers, self.scaled);
        for_each_kmer(seq, "protein", self.k, |c| {
            if keep(c, scaled) {
                let e = kmers.entry(c).or_insert(gene);
                if *e != gene {
                    *e = SHARED;
                }
            }
        });
    }

    /// Adds to `hits` (gene, matching windows) every window of the six frames of `read`.
    fn lookup(&self, read: &[u8], hits: &mut Vec<(u32, u32)>) {
        const SIZE: u64 = 20;
        let top = SIZE.pow(self.k as u32 - 1);
        for strand in 0..2 {
            for off in 0..3usize {
                let (mut code, mut run) = (0u64, 0usize);
                let n = read.len();
                let mut i = off;
                while i + 3 <= n {
                    let b = |j: usize| {
                        if strand == 0 {
                            base(read[j])
                        } else {
                            base(read[n - 1 - j]).map(|x| 3 - x)
                        }
                    };
                    let aa = match (b(i), b(i + 1), b(i + 2)) {
                        (Some(x), Some(y), Some(z)) => self.codon[(16 * x + 4 * y + z) as usize],
                        _ => STOP,
                    };
                    i += 3;
                    if aa == STOP {
                        run = 0;
                        code = 0;
                        continue;
                    }
                    code = if run >= self.k { (code % top) * SIZE + aa as u64 } else { code * SIZE + aa as u64 };
                    run += 1;
                    if run >= self.k && keep(code, self.scaled) {
                        if let Some(&g) = self.kmers.get(&code) {
                            if g != SHARED {
                                match hits.iter_mut().find(|h| h.0 == g) {
                                    Some(h) => h.1 += 1,
                                    None => hits.push((g, 1)),
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, PartialEq)]
enum Call {
    NoHit,
    Tied,
    Gene(u32),
}

/// The gene with the most hits, if it has at least `min_hits` and no other gene has as many.
fn call(hits: &[(u32, u32)], min_hits: u32) -> Call {
    let best = hits.iter().map(|h| h.1).max().unwrap_or(0);
    if best < min_hits {
        return Call::NoHit;
    }
    let mut top = hits.iter().filter(|h| h.1 == best);
    match (top.next(), top.next()) {
        (Some(h), None) => Call::Gene(h.0),
        _ => Call::Tied,
    }
}

pub fn load_index(faa: &str, gene_map: &str, k: usize, scaled: u64) -> Res<Index> {
    let mut idx = Index::new(k, scaled);
    let mut gene_ids: HashMap<String, u32> = HashMap::new();
    let mut protein_gene: HashMap<String, u32> = HashMap::new();
    for line in BufReader::new(File::open(gene_map)?).lines() {
        let line = line?;
        let (p, g) = line.split_once('\t').ok_or("gene_map needs two columns")?;
        let next = gene_ids.len() as u32;
        let id = *gene_ids.entry(g.to_string()).or_insert_with(|| {
            idx.genes.push(g.to_string());
            next
        });
        protein_gene.insert(p.to_string(), id);
    }
    let mut seq = Vec::new();
    let mut cur: Option<u32> = None;
    for line in BufReader::with_capacity(1 << 20, File::open(faa)?).lines() {
        let line = line?;
        if let Some(h) = line.strip_prefix('>') {
            if let Some(g) = cur {
                idx.add(&seq, g);
            }
            seq.clear();
            cur = h.split_whitespace().next().and_then(|p| protein_gene.get(p)).copied();
        } else {
            seq.extend(line.trim().bytes());
        }
    }
    if let Some(g) = cur {
        idx.add(&seq, g);
    }
    Ok(idx)
}

struct Fastq {
    child: Child,
    r: BufReader<ChildStdout>,
    head: String,
    plus: String,
    qual: String,
}

impl Fastq {
    fn open(path: &str) -> Res<Self> {
        let mut child = Command::new("pigz").args(["-dc", path]).stdout(Stdio::piped()).spawn()?;
        let out = child.stdout.take().ok_or("no stdout from pigz")?;
        Ok(Fastq {
            child,
            r: BufReader::with_capacity(1 << 20, out),
            head: String::new(),
            plus: String::new(),
            qual: String::new(),
        })
    }

    /// Reads one record's sequence into `seq`; false at the end of the file.
    fn next(&mut self, seq: &mut String) -> Res<bool> {
        self.head.clear();
        if self.r.read_line(&mut self.head)? == 0 {
            return Ok(false);
        }
        seq.clear();
        self.r.read_line(seq)?;
        self.plus.clear();
        self.r.read_line(&mut self.plus)?;
        self.qual.clear();
        self.r.read_line(&mut self.qual)?;
        if self.qual.is_empty() {
            return Err("FASTQ ends inside a record".into());
        }
        Ok(true)
    }
}

impl Drop for Fastq {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[derive(Default)]
struct Tally {
    counts: Vec<u32>,
    pairs: u64,
    no_hit: u64,
    tied: u64,
    assigned: u64,
}

fn count_sample(idx: &Index, r1: &str, r2: &str, max_pairs: u64, min_hits: u32) -> Res<Tally> {
    // single-end reads: the manifest gives `-` as the R2 path, and there is no second mate
    let mut a = Fastq::open(r1)?;
    let mut b = if r2 == "-" { None } else { Some(Fastq::open(r2)?) };
    let (mut s1, mut s2) = (String::new(), String::new());
    let mut hits = Vec::new();
    let mut t = Tally { counts: vec![0; idx.genes.len()], ..Default::default() };
    while max_pairs == 0 || t.pairs < max_pairs {
        let more1 = a.next(&mut s1)?;
        if let Some(b) = b.as_mut() {
            if more1 != b.next(&mut s2)? {
                return Err(format!("{r1} and {r2} have different numbers of reads").into());
            }
        }
        if !more1 {
            if max_pairs == 0 {
                for f in std::iter::once(&mut a).chain(b.as_mut()) {
                    if !f.child.wait()?.success() {
                        return Err(format!("pigz failed on {r1} or {r2}").into());
                    }
                }
            }
            break;
        }
        t.pairs += 1;
        hits.clear();
        idx.lookup(s1.trim_end().as_bytes(), &mut hits);
        if b.is_some() {
            idx.lookup(s2.trim_end().as_bytes(), &mut hits);
        }
        match call(&hits, min_hits) {
            Call::NoHit => t.no_hit += 1,
            Call::Tied => t.tied += 1,
            Call::Gene(g) => {
                t.counts[g as usize] += 1;
                t.assigned += 1;
            }
        }
    }
    Ok(t)
}

pub fn cmd_gene_counts(args: &[String]) -> Res<()> {
    if args.len() < 5 {
        return Err("usage: pec-tools gene-counts <proteins.faa> <gene_map.tsv> <manifest.tsv> <out_prefix> <threads> [k] [min_hits] [scaled]".into());
    }
    let threads: usize = args[4].parse()?;
    let k: usize = args.get(5).map_or(Ok(10), |s| s.parse())?;
    let min_hits: u32 = args.get(6).map_or(Ok(3), |s| s.parse())?;
    let scaled: u64 = args.get(7).map_or(Ok(1), |s| s.parse())?;
    let idx = load_index(&args[0], &args[1], k, scaled)?;
    eprintln!("genes {}, k-mers {}", idx.genes.len(), idx.kmers.len());

    let mut samples: Vec<(String, String, String, u64)> = Vec::new();
    for line in BufReader::new(File::open(&args[2])?).lines() {
        let line = line?;
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err("manifest needs sample, R1, R2, largest pairs".into());
        }
        samples.push((f[0].into(), f[1].into(), f[2].into(), f[3].parse()?));
    }
    let next = AtomicUsize::new(0);
    let results: Vec<Vec<(usize, Result<Tally, String>)>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|_| {
                sc.spawn(|| {
                    let mut got = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some((name, r1, r2, n)) = samples.get(i) else { break };
                        let t = count_sample(&idx, r1, r2, *n, min_hits).map_err(|e| e.to_string());
                        eprintln!("{name} done");
                        got.push((i, t));
                    }
                    got
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().expect("worker panicked")).collect()
    });
    let mut tallies: Vec<Option<Tally>> = samples.iter().map(|_| None).collect();
    for (i, t) in results.into_iter().flatten() {
        tallies[i] = Some(t.map_err(|e| format!("{}: {e}", samples[i].0))?);
    }
    let tallies: Vec<Tally> = tallies.into_iter().map(|t| t.expect("every sample counted")).collect();

    let mut out = BufWriter::new(File::create(format!("{}.counts.tsv", args[3]))?);
    write!(out, "gene")?;
    for s in &samples {
        write!(out, "\t{}", s.0)?;
    }
    writeln!(out)?;
    for (g, name) in idx.genes.iter().enumerate() {
        if tallies.iter().any(|t| t.counts[g] > 0) {
            write!(out, "{name}")?;
            for t in &tallies {
                write!(out, "\t{}", t.counts[g])?;
            }
            writeln!(out)?;
        }
    }
    let mut sum = BufWriter::new(File::create(format!("{}.summary.tsv", args[3]))?);
    writeln!(sum, "sample\tpairs\tno_hit\ttied\tassigned")?;
    for (s, t) in samples.iter().zip(&tallies) {
        writeln!(sum, "{}\t{}\t{}\t{}\t{}", s.0, t.pairs, t.no_hit, t.tied, t.assigned)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::revcomp;

    // MKWDEFHIKLMNPQRSTVWYACDE back-translated by hand with one codon per residue
    const DNA: &[u8] = b"ATGAAATGGGATGAATTTCATATTAAACTGATGAATCCGCAGCGTAGCACCGTGTGGTATGCGTGCGATGAA";
    const AA: &[u8] = b"MKWDEFHIKLMNPQRSTVWYACDE";

    fn index(k: usize, scaled: u64) -> Index {
        let mut idx = Index::new(k, scaled);
        idx.genes = vec!["a".into(), "b".into()];
        idx.add(AA, 0);
        idx
    }

    fn hits(idx: &Index, read: &[u8]) -> Vec<(u32, u32)> {
        let mut h = Vec::new();
        idx.lookup(read, &mut h);
        h
    }

    #[test]
    fn every_window_matches_on_either_strand_and_frame_offset() {
        let idx = index(10, 1);
        assert_eq!(hits(&idx, DNA), vec![(0, 15)]); // 24 residues, 15 windows of 10
        assert_eq!(hits(&idx, &revcomp(DNA)), vec![(0, 15)]);
        let mut shifted = b"AC".to_vec();
        shifted.extend_from_slice(DNA);
        assert_eq!(hits(&idx, &shifted), vec![(0, 15)]);
    }

    #[test]
    fn a_kmer_in_two_genes_is_left_out() {
        let mut idx = index(10, 1);
        idx.add(&AA[..12], 1); // first 3 windows shared
        assert_eq!(hits(&idx, DNA), vec![(0, 12)]);
    }

    #[test]
    fn a_stop_or_unknown_base_breaks_the_window() {
        let idx = index(10, 1);
        let mut d = DNA.to_vec();
        d[36] = b'N'; // codon 13 of 24: only windows entirely before or after it match
        assert_eq!(hits(&idx, &d), vec![(0, 5)]); // windows 0..=2 before, 14 residues minus 13 after
    }

    #[test]
    fn calls_need_enough_hits_and_a_clear_winner() {
        assert_eq!(call(&[], 3), Call::NoHit);
        assert_eq!(call(&[(0, 2)], 3), Call::NoHit);
        assert_eq!(call(&[(0, 3)], 3), Call::Gene(0));
        assert_eq!(call(&[(0, 4), (1, 4)], 3), Call::Tied);
        assert_eq!(call(&[(0, 5), (1, 4)], 3), Call::Gene(0));
    }

    /// Two copies of one read, gzipped, in a file of its own under the system temp folder.
    fn fastq_gz(name: &str) -> String {
        let path = std::env::temp_dir().join(format!("pec_tools_{}_{name}.fastq", std::process::id()));
        let read = String::from_utf8(DNA.to_vec()).unwrap();
        let qual = "I".repeat(read.len());
        std::fs::write(&path, format!("@r1\n{read}\n+\n{qual}\n@r2\n{read}\n+\n{qual}\n")).unwrap();
        assert!(Command::new("pigz").arg("-f").arg(&path).status().unwrap().success());
        format!("{}.gz", path.display())
    }

    #[test]
    fn single_end_reads_are_counted_without_a_second_mate() {
        if Command::new("pigz").arg("--version").output().is_err() {
            eprintln!("skipped: pigz is not installed here (it is on Sherlock, in the pipeline's environment)");
            return;
        }
        let idx = index(10, 1);
        let f = fastq_gz("single");
        let single = count_sample(&idx, &f, "-", 0, 3).unwrap();
        assert_eq!((single.pairs, single.assigned, single.counts[0]), (2, 2, 2));
        // the same file as both mates still gives two pairs, so `-` is not read as a file name
        let paired = count_sample(&idx, &f, &f, 0, 3).unwrap();
        assert_eq!((paired.pairs, paired.assigned), (2, 2));
        // a limit of one read stops after the first
        assert_eq!(count_sample(&idx, &f, "-", 1, 3).unwrap().pairs, 1);
        std::fs::remove_file(&f).unwrap();
    }

    #[test]
    fn scaled_keeps_a_share_of_kmers_consistently() {
        let idx = index(10, 3);
        let n = idx.kmers.len() as u32;
        assert!(n < 15);
        let got = hits(&idx, DNA);
        assert_eq!(got.iter().map(|h| h.1).sum::<u32>(), n);
    }
}
