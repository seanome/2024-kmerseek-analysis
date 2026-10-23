//! pec-tools: small Rust utilities for the Botryllus PEC reproduction.
//!
//! translate  — six-frame translation of compactors (TSV or FASTA) into stop-free
//!              peptides of at least `min_aa` residues, with DNA coordinates.
//! simulate   — paired-end read simulator from a transcript FASTA with a planted
//!              design (used only for the known-answer control).
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

// Standard genetic code (NCBI table 1), index = 16*b1 + 4*b2 + b3 with A=0,C=1,G=2,T=3.
const CODE: &[u8; 64] = b"KNKNTTTTRSRSIIMIQHQHPPPPRRRRLLLLEDEDAAAAGGGGVVVV*Y*YSSSS*CWCLFLF";

fn base(c: u8) -> Option<u8> {
    match c {
        b'A' | b'a' => Some(0),
        b'C' | b'c' => Some(1),
        b'G' | b'g' => Some(2),
        b'T' | b't' => Some(3),
        _ => None,
    }
}

fn revcomp(s: &[u8]) -> Vec<u8> {
    s.iter()
        .rev()
        .map(|&c| match c {
            b'A' | b'a' => b'T',
            b'C' | b'c' => b'G',
            b'G' | b'g' => b'C',
            b'T' | b't' => b'A',
            _ => b'N',
        })
        .collect()
}

/// Translate one frame; non-ACGT codons become 'X'.
fn translate_frame(s: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len() / 3);
    let mut i = 0;
    while i + 3 <= s.len() {
        let aa = match (base(s[i]), base(s[i + 1]), base(s[i + 2])) {
            (Some(a), Some(b), Some(c)) => CODE[(16 * a + 4 * b + c) as usize],
            _ => b'X',
        };
        out.push(aa);
        i += 3;
    }
    out
}

#[derive(Debug, PartialEq)]
struct Peptide {
    strand: char,
    frame: u8,       // 1..=3
    nt_start: usize, // 0-based, half-open, on the forward strand of the input
    nt_end: usize,
    aa: String,
}

/// All stop-free, X-free peptides of length >= min_aa in six frames.
fn six_frame(dna: &[u8], min_aa: usize) -> Vec<Peptide> {
    let mut out = Vec::new();
    let rc = revcomp(dna);
    let n = dna.len();
    for (strand, seq) in [('+', dna), ('-', &rc[..])] {
        for off in 0..3usize {
            if seq.len() < off + 3 {
                continue;
            }
            let aa = translate_frame(&seq[off..]);
            let mut start = 0usize;
            let mut i = 0usize;
            while i <= aa.len() {
                let brk = i == aa.len() || aa[i] == b'*' || aa[i] == b'X';
                if brk {
                    if i - start >= min_aa {
                        let (s, e) = (off + start * 3, off + i * 3);
                        let (ns, ne) = if strand == '+' {
                            (s, e)
                        } else {
                            (n - e, n - s)
                        };
                        out.push(Peptide {
                            strand,
                            frame: off as u8 + 1,
                            nt_start: ns,
                            nt_end: ne,
                            aa: String::from_utf8(aa[start..i].to_vec()).unwrap(),
                        });
                    }
                    start = i + 1;
                }
                i += 1;
            }
        }
    }
    out
}

struct Record {
    id: String,
    seq: Vec<u8>,
    extra: Vec<(String, String)>,
}

fn read_fasta(path: &str) -> Res<Vec<Record>> {
    let mut txt = String::new();
    File::open(path)?.read_to_string(&mut txt)?;
    let mut recs = Vec::new();
    let mut cur: Option<Record> = None;
    for line in txt.lines() {
        if let Some(h) = line.strip_prefix('>') {
            if let Some(r) = cur.take() {
                recs.push(r);
            }
            cur = Some(Record {
                id: h.split_whitespace().next().unwrap_or("").to_string(),
                seq: Vec::new(),
                extra: vec![],
            });
        } else if let Some(r) = cur.as_mut() {
            r.seq.extend(line.trim().bytes());
        }
    }
    if let Some(r) = cur.take() {
        recs.push(r);
    }
    Ok(recs)
}

/// Compactors TSV from SPLASH: must have columns `anchor` and `compactor`; other columns are carried along.
fn read_compactor_tsv(path: &str, seq_col: &str) -> Res<Vec<Record>> {
    let f = BufReader::new(File::open(path)?);
    let mut lines = f.lines();
    let header: Vec<String> = lines
        .next()
        .ok_or("empty TSV")??
        .split('\t')
        .map(|s| s.to_string())
        .collect();
    let si = header
        .iter()
        .position(|h| h == seq_col)
        .ok_or(format!("no column {seq_col}"))?;
    let mut recs = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != header.len() {
            return Err(format!(
                "line {} has {} fields, header has {}",
                i + 2,
                f.len(),
                header.len()
            )
            .into());
        }
        let extra = header
            .iter()
            .cloned()
            .zip(f.iter().map(|s| s.to_string()))
            .collect();
        recs.push(Record {
            id: format!("c{}", i + 1),
            seq: f[si].as_bytes().to_vec(),
            extra,
        });
    }
    Ok(recs)
}

fn cmd_translate(args: &[String]) -> Res<()> {
    // translate <in.tsv|in.fa> <out_prefix> [min_aa=20] [seq_col=compactor]
    if args.len() < 2 {
        return Err("usage: pec-tools translate <compactors.tsv|seqs.fa> <out_prefix> [min_aa] [seq_column]".into());
    }
    let min_aa: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(20);
    if min_aa == 0 {
        return Err("min_aa must be positive".into());
    }
    let seq_col = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "compactor".to_string());
    let recs = if args[0].ends_with(".tsv") {
        read_compactor_tsv(&args[0], &seq_col)?
    } else {
        read_fasta(&args[0])?
    };
    let mut fa = BufWriter::new(File::create(format!("{}.peptides.faa", args[1]))?);
    let mut tsv = BufWriter::new(File::create(format!("{}.peptides.tsv", args[1]))?);
    let mut dna_len = BufWriter::new(File::create(format!("{}.dna_lengths.tsv", args[1]))?);
    let extra_cols: Vec<String> = recs
        .first()
        .map(|r| r.extra.iter().map(|(k, _)| k.clone()).collect())
        .unwrap_or_default();
    writeln!(
        tsv,
        "peptide_id\tseq_id\tdna_length\tstrand\tframe\tnt_start\tnt_end\taa_length\tpeptide{}",
        extra_cols
            .iter()
            .map(|c| format!("\t{c}"))
            .collect::<String>()
    )?;
    writeln!(
        dna_len,
        "seq_id\tdna_length\tn_peptides\tlongest_peptide_aa"
    )?;
    let (mut n_pep, mut n_seq_with) = (0usize, 0usize);
    for r in &recs {
        let peps = six_frame(&r.seq, min_aa);
        let longest = peps.iter().map(|p| p.aa.len()).max().unwrap_or(0);
        writeln!(
            dna_len,
            "{}\t{}\t{}\t{}",
            r.id,
            r.seq.len(),
            peps.len(),
            longest
        )?;
        if !peps.is_empty() {
            n_seq_with += 1;
        }
        for (j, p) in peps.iter().enumerate() {
            let pid = format!(
                "{}|{}{}|{}-{}",
                r.id, p.strand, p.frame, p.nt_start, p.nt_end
            );
            writeln!(fa, ">{pid}\n{}", p.aa)?;
            writeln!(
                tsv,
                "{pid}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}{}",
                r.id,
                r.seq.len(),
                p.strand,
                p.frame,
                p.nt_start,
                p.nt_end,
                p.aa.len(),
                p.aa,
                r.extra
                    .iter()
                    .map(|(_, v)| format!("\t{v}"))
                    .collect::<String>()
            )?;
            n_pep += 1;
            let _ = j;
        }
    }
    fa.flush()?;
    tsv.flush()?;
    dna_len.flush()?;
    eprintln!(
        "sequences: {}  with >= {} aa peptide: {}  peptides: {}",
        recs.len(),
        min_aa,
        n_seq_with,
        n_pep
    );
    Ok(())
}

// ---------------- simulator (known-answer control only) ----------------
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.uniform() * n as f64) as usize
    }
    fn poisson(&mut self, lambda: f64) -> usize {
        // Knuth for small lambda, normal approx for large
        if lambda > 50.0 {
            let z = self.normal();
            return (lambda + z * lambda.sqrt()).round().max(0.0) as usize;
        }
        let l = (-lambda).exp();
        let (mut k, mut p) = (0usize, 1.0);
        loop {
            p *= self.uniform();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }
    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-12), self.uniform());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
    fn gamma(&mut self, shape: f64) -> f64 {
        // Marsaglia–Tsang, shape >= 1 assumed (we use shape = 1/dispersion >= 1)
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal();
            let v = (1.0 + c * x).powi(3);
            if v <= 0.0 {
                continue;
            }
            let u = self.uniform();
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

fn cmd_simulate(args: &[String]) -> Res<()> {
    // simulate <transcripts.fa> <design.tsv> <out_dir> <pairs_per_sample> <seed>
    // design.tsv: transcript<TAB>base_tpm<TAB>fold_change_in_treated  (one row per transcript to express)
    //             plus header line: sample<TAB>group columns are NOT here; samples are fixed as
    //             6 control + 6 treated (ctrl1..6, pec1..6).
    if args.len() != 5 {
        return Err("usage: pec-tools simulate <transcripts.fa> <design.tsv> <out_dir> <pairs_per_sample> <seed>".into());
    }
    let tx = read_fasta(&args[0])?;
    let by_id: HashMap<&str, &Record> = tx.iter().map(|r| (r.id.as_str(), r)).collect();
    if by_id.len() != tx.len() {
        return Err("duplicate transcript FASTA IDs".into());
    }
    let mut design: Vec<(String, f64, f64)> = Vec::new();
    for (i, line) in BufReader::new(File::open(&args[1])?).lines().enumerate() {
        let line = line?;
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 3 {
            return Err(format!("design line {} needs three fields", i + 1).into());
        }
        if !by_id.contains_key(f[0]) {
            return Err(format!("design transcript {} not in FASTA", f[0]).into());
        }
        let tpm: f64 = f[1].parse()?;
        let fc: f64 = f[2].parse()?;
        if !tpm.is_finite() || tpm < 0.0 || !fc.is_finite() || fc <= 0.0 {
            return Err(format!("invalid abundance or fold change on line {}", i + 1).into());
        }
        if design.iter().any(|(id, _, _)| id == f[0]) {
            return Err("duplicate design transcript".into());
        }
        design.push((f[0].to_string(), tpm, fc));
    }
    std::fs::create_dir_all(&args[2])?;
    let pairs: usize = args[3].parse()?;
    let seed: u64 = args[4].parse()?;
    let read_len = 150usize;
    let samples: Vec<(String, bool)> = (1..=6)
        .map(|i| (format!("ctrl{i}"), false))
        .chain((1..=6).map(|i| (format!("pec{i}"), true)))
        .collect();
    let mut truth = BufWriter::new(File::create(format!("{}/truth_counts.tsv", args[2]))?);
    write!(truth, "transcript")?;
    for (s, _) in &samples {
        write!(truth, "\t{s}")?;
    }
    writeln!(truth)?;
    let mut counts: Vec<Vec<usize>> = vec![vec![0; samples.len()]; design.len()];
    for (si, (name, treated)) in samples.iter().enumerate() {
        let mut rng = Rng(seed ^ (0x9E3779B97F4A7C15u64.wrapping_mul(si as u64 + 1)));
        // expected fragments per transcript: base_tpm * fc(if treated) * length weight, NB noise (dispersion 0.1)
        let mut expect: Vec<f64> = design
            .iter()
            .map(|(id, tpm, fc)| {
                let l = by_id[id.as_str()].seq.len() as f64;
                tpm * if *treated { *fc } else { 1.0 } * (l / 1000.0).max(0.2)
            })
            .collect();
        let tot: f64 = expect.iter().sum();
        if !tot.is_finite() || tot <= 0.0 {
            return Err("total expected abundance must be positive and finite".into());
        }
        for e in expect.iter_mut() {
            *e = *e / tot * pairs as f64;
            let g = rng.gamma(10.0) / 10.0; // mean 1, dispersion 0.1
            *e *= g;
        }
        let mut w1 = BufWriter::new(File::create(format!("{}/{name}_1.fastq", args[2]))?);
        let mut w2 = BufWriter::new(File::create(format!("{}/{name}_2.fastq", args[2]))?);
        let qual = "I".repeat(read_len);
        let mut rid = 0usize;
        for (ti, (id, _, _)) in design.iter().enumerate() {
            let seq = &by_id[id.as_str()].seq;
            if seq.len() < read_len + 20 {
                continue;
            }
            let n = rng.poisson(expect[ti]);
            counts[ti][si] = n;
            for _ in 0..n {
                let insert = (250.0 + 40.0 * rng.normal()).round().max(read_len as f64) as usize;
                let insert = insert.min(seq.len());
                let start = rng.below(seq.len() - insert + 1);
                let frag = &seq[start..start + insert];
                let fwd = rng.uniform() < 0.5; // unstranded library
                let (a, b) = if fwd {
                    (frag.to_vec(), revcomp(frag))
                } else {
                    (revcomp(frag), frag.to_vec())
                };
                let mut r1 = a[..read_len.min(a.len())].to_vec();
                let mut r2 = b[..read_len.min(b.len())].to_vec();
                // 0.3 % substitution errors
                for r in [&mut r1, &mut r2] {
                    for c in r.iter_mut() {
                        if rng.uniform() < 0.003 {
                            *c = b"ACGT"[rng.below(4)];
                        }
                    }
                }
                rid += 1;
                writeln!(
                    w1,
                    "@{name}.{rid}/1\n{}\n+\n{}",
                    String::from_utf8_lossy(&r1),
                    &qual[..r1.len()]
                )?;
                writeln!(
                    w2,
                    "@{name}.{rid}/2\n{}\n+\n{}",
                    String::from_utf8_lossy(&r2),
                    &qual[..r2.len()]
                )?;
            }
        }
        w1.flush()?;
        w2.flush()?;
        eprintln!("{name}: {rid} pairs");
    }
    for (ti, (id, _, _)) in design.iter().enumerate() {
        write!(truth, "{id}")?;
        for c in &counts[ti] {
            write!(truth, "\t{c}")?;
        }
        writeln!(truth)?;
    }
    truth.flush()?;
    Ok(())
}

// ---------------- anchor counts from SATC text ----------------

/// Sum target counts per (anchor, sample) from `satc_dump` text output.
///
/// Input lines are `sample<TAB>anchor<TAB>target<TAB>count` (SATC text format, SPLASH 2.11.9,
/// no barcode column). Samples are output in the order of `samples`; a sample name not in
/// that list is an error. Returns rows sorted by anchor: (anchor, per-sample counts, number of
/// distinct targets).
fn sum_anchor_counts<R: BufRead>(
    input: R,
    samples: &[String],
    keep: Option<&std::collections::HashSet<String>>,
) -> Res<Vec<(String, Vec<u64>, u32)>> {
    let col: HashMap<&str, usize> = samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();
    if col.len() != samples.len() || samples.is_empty() {
        return Err("sample list must be nonempty with unique names".into());
    }
    let mut acc: HashMap<String, (Vec<u64>, std::collections::HashSet<String>)> = HashMap::new();
    for (ln, line) in input.lines().enumerate() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 4 {
            return Err(format!(
                "line {}: expected 4 tab-separated fields, got {}",
                ln + 1,
                f.len()
            )
            .into());
        }
        if let Some(k) = keep {
            if !k.contains(f[1]) {
                continue;
            }
        }
        let si = *col.get(f[0]).ok_or(format!(
            "line {}: sample {} not in sample list",
            ln + 1,
            f[0]
        ))?;
        let c: u64 = f[3].parse()?;
        let e = acc
            .entry(f[1].to_string())
            .or_insert_with(|| (vec![0; samples.len()], Default::default()));
        e.0[si] = e.0[si].checked_add(c).ok_or("anchor count exceeds u64")?;
        e.1.insert(f[2].to_string());
    }
    let mut rows: Vec<_> = acc
        .into_iter()
        .map(|(a, (v, t))| (a, v, t.len() as u32))
        .collect();
    rows.sort_by(|x, y| x.0.cmp(&y.0));
    Ok(rows)
}

fn cmd_anchor_counts(args: &[String]) -> Res<()> {
    // anchor-counts <samples.txt> [anchor_list.tsv|-]   (SATC text on stdin, matrix on stdout)
    if args.is_empty() {
        return Err("usage: pec-tools anchor-counts <samples.txt> [anchors.tsv with an 'anchor' column] < satc.txt".into());
    }
    let samples: Vec<String> = std::fs::read_to_string(&args[0])?
        .split_whitespace()
        .map(String::from)
        .collect();
    let keep = match args.get(1) {
        Some(p) if p != "-" => {
            let mut lines = BufReader::new(File::open(p)?).lines();
            let header: Vec<String> = lines
                .next()
                .ok_or("empty anchor list")??
                .split('\t')
                .map(String::from)
                .collect();
            let ai = header
                .iter()
                .position(|h| h == "anchor")
                .ok_or("anchor list has no 'anchor' column")?;
            let mut set = std::collections::HashSet::new();
            for l in lines {
                let l = l?;
                if let Some(a) = l.split('\t').nth(ai) {
                    set.insert(a.to_string());
                }
            }
            Some(set)
        }
        _ => None,
    };
    let rows = sum_anchor_counts(std::io::stdin().lock(), &samples, keep.as_ref())?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    for (a, v, t) in &rows {
        write!(out, "{a}\t{t}")?;
        for c in v {
            write!(out, "\t{c}")?;
        }
        writeln!(out)?;
    }
    eprintln!("anchors written: {}", rows.len());
    Ok(())
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("translate") => cmd_translate(&args[2..]),
        Some("simulate") => cmd_simulate(&args[2..]),
        Some("anchor-counts") => cmd_anchor_counts(&args[2..]),
        _ => Err("usage: pec-tools <translate|simulate|anchor-counts> ...".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn codon_table_matches_known_codons() {
        assert_eq!(translate_frame(b"ATGAAATGA"), b"MK*");
        assert_eq!(translate_frame(b"TGGTAGTAATGA"), b"W***");
        assert_eq!(translate_frame(b"GCNACG"), b"XT");
    }
    #[test]
    fn six_frame_coordinates_round_trip() {
        // forward ORF: ATG GCC GAA TAA  -> "MAE" at 0..9
        let dna = b"ATGGCCGAATAA";
        let peps = six_frame(dna, 3);
        let f: Vec<_> = peps.iter().filter(|p| p.strand == '+').collect();
        assert!(f
            .iter()
            .any(|p| p.aa == "MAE" && p.nt_start == 0 && p.nt_end == 9 && p.frame == 1));
        // every reported window re-translates to the reported peptide
        for p in &peps {
            let sub = &dna[p.nt_start..p.nt_end];
            let s = if p.strand == '+' {
                sub.to_vec()
            } else {
                revcomp(sub)
            };
            assert_eq!(
                String::from_utf8(translate_frame(&s)).unwrap(),
                p.aa,
                "{:?}",
                p
            );
        }
    }
    #[test]
    fn reverse_strand_peptide_found() {
        let fwd = b"ATGTTTGGGAAACCCTAA"; // MFGKP*
        let rc = revcomp(fwd);
        let peps = six_frame(&rc, 5);
        let hit = peps
            .iter()
            .find(|p| p.strand == '-' && p.aa == "MFGKP")
            .expect("reverse ORF");
        let sub = revcomp(&rc[hit.nt_start..hit.nt_end]);
        assert_eq!(translate_frame(&sub), b"MFGKP");
    }
    #[test]
    fn anchor_counts_sum_targets_per_sample() {
        let satc = "pec1\tAAAC\tGG\t3\nctrl1\tAAAC\tGT\t2\npec1\tAAAC\tGT\t4\npec1\tCCCA\tTT\t1\n";
        let samples = vec!["ctrl1".to_string(), "pec1".to_string()];
        let rows = sum_anchor_counts(satc.as_bytes(), &samples, None).unwrap();
        assert_eq!(
            rows,
            vec![
                ("AAAC".to_string(), vec![2, 7], 2),
                ("CCCA".to_string(), vec![0, 1], 1)
            ]
        );
        let keep: std::collections::HashSet<String> = ["CCCA".to_string()].into_iter().collect();
        let rows = sum_anchor_counts(satc.as_bytes(), &samples, Some(&keep)).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(sum_anchor_counts("x\tA\tC\t1\n".as_bytes(), &samples, None).is_err());
    }
    #[test]
    fn ambiguous_bases_break_peptides() {
        let peps = six_frame(b"ATGGCCNAAGCCGCCGCC", 3);
        assert!(peps
            .iter()
            .all(|p| !p.aa.contains('X') && !p.aa.contains('*')));
    }
    #[test]
    fn anchor_counts_reject_duplicate_samples_and_overflow() {
        let duplicate = vec!["s".to_string(), "s".to_string()];
        assert!(sum_anchor_counts(&b"s\tAAA\tCCC\t1\n"[..], &duplicate, None).is_err());
        let samples = vec!["s".to_string()];
        let data = b"s\tAAA\tCCC\t18446744073709551615\ns\tAAA\tGGG\t1\n";
        assert!(sum_anchor_counts(&data[..], &samples, None).is_err());
    }
}
