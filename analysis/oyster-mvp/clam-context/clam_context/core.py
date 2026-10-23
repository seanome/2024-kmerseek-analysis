"""Scaled sourmash sketches with a disk-backed, per-protein inverted index."""
import csv
import gzip
import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.Data import CodonTable
import sourmash

MODES = ('protein', 'dayhoff', 'hp')
HP = {aa: ('h' if aa in 'AFGILMPVWY' else 'p') for aa in 'ACDEFGHIKLMNPQRSTVWY'}
AA = set('ACDEFGHIKLMNPQRSTVWY')
DAYHOFF = {aa: str(i) for i, group in enumerate(['C', 'AGPST', 'DENQ', 'HKR', 'ILMV', 'FWY']) for aa in group}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sketch(peptide, mode, k, scaled=10):
    if mode not in MODES:
        raise ValueError('Encoding must be protein, dayhoff or hp')
    mh = sourmash.MinHash(n=0, ksize=k, scaled=scaled, seed=42,
                         **({'is_protein': True} if mode == 'protein' else {mode: True}))
    # Never bridge ambiguous residues or stops.
    for segment in re.findall('[ACDEFGHIKLMNPQRSTVWY]+', peptide.upper()):
        if len(segment) >= k:
            mh.add_protein(segment)
    return mh


def translate_segments(dna, min_aa=20, table=1):
    """Six-frame stop/ambiguity-free fragments, 0-based half-open DNA coordinates."""
    dna = dna.upper()
    if not dna or set(dna) - set('ACGTRYSWKMBDHVN'):
        raise ValueError('Expected IUPAC DNA sequence')
    CodonTable.unambiguous_dna_by_id[table]  # validate even if sequence too short
    for strand, oriented in [('+', dna), ('-', str(Seq(dna).reverse_complement()))]:
        for offset in range(3):
            usable = (len(oriented) - offset) // 3 * 3
            if usable <= 0:
                continue
            aa = str(Seq(oriented[offset:offset + usable]).translate(table=table))
            for match in re.finditer('[ACDEFGHIKLMNPQRSTVWY]+', aa):
                if len(match[0]) < min_aa:
                    continue
                start, end = offset + match.start() * 3, offset + match.end() * 3
                if strand == '-':
                    start, end = len(dna) - end, len(dna) - start
                yield dict(peptide=match[0], strand=strand, frame=offset + 1,
                           nt_start=start, nt_end=end)


def build_index(fasta, output, protein_k=7, dayhoff_k=12, hp_k=30, scaled=10, label='User protein FASTA'):
    if scaled < 1:
        raise ValueError('scaled must be a positive integer')
    if min(protein_k, dayhoff_k, hp_k) < 2:
        raise ValueError('k must be at least 2 amino acids')
    output = Path(output)
    if output.exists():
        raise ValueError(f'Refusing to overwrite index: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + '.building')
    if tmp.exists():
        raise ValueError(f'Remove stale build file before retrying: {tmp}')
    con = sqlite3.connect(tmp)
    try:
        con.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE proteins(id TEXT PRIMARY KEY, description TEXT, sequence TEXT);
            CREATE TABLE postings(mode TEXT, hash TEXT, protein_id TEXT);
            CREATE UNIQUE INDEX postings_lookup ON postings(mode, hash, protein_id);''')
        meta = dict(schema=1, label=label, protein_k=protein_k, dayhoff_k=dayhoff_k, hp_k=hp_k,
                    scaled=scaled, seed=42, sourmash=version('sourmash'), source_sha256=sha256(fasta),
                    source_name=Path(fasta).name, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        count = 0
        opener = gzip.open if str(fasta).endswith('.gz') else open
        with opener(fasta, 'rt') as handle:
            for rec in SeqIO.parse(handle, 'fasta'):
                peptide = str(rec.seq).upper()
                if not peptide or set(peptide) - (AA | set('XBZJUO*')):
                    raise ValueError(f'Invalid protein characters in {rec.id}')
                con.execute('INSERT INTO proteins VALUES (?,?,?)', (rec.id, rec.description, peptide))
                for mode, k in [('protein', protein_k), ('dayhoff', dayhoff_k), ('hp', hp_k)]:
                    mh = sketch(peptide, mode, k, scaled)
                    con.executemany('INSERT INTO postings VALUES (?,?,?)',
                                    ((mode, f'{h:016x}', rec.id) for h in mh.hashes))
                count += 1
                if count % 1000 == 0:
                    con.commit()
        if not count:
            raise ValueError('Protein FASTA is empty')
        meta['protein_count'] = count
        con.executemany('INSERT INTO meta VALUES (?,?)', ((k, json.dumps(v)) for k, v in meta.items()))
        con.commit()
        con.close()
        tmp.replace(output)
        return meta
    except BaseException:
        con.close()
        tmp.unlink(missing_ok=True)
        raise


def encoded_kmers(peptide, mode, k):
    encoded = peptide if mode == 'protein' else ''.join((DAYHOFF if mode == 'dayhoff' else HP).get(x, '?') for x in peptide)
    return [(i, encoded[i:i+k]) for i in range(len(encoded)-k+1)
            if '?' not in encoded[i:i+k] and all(x in AA for x in peptide[i:i+k])]


class ProteinIndex:
    def __init__(self, path):
        self.con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
        self.meta = {k: json.loads(v) for k, v in self.con.execute('SELECT key,value FROM meta')}
        if self.meta.get('schema') != 1:
            raise ValueError('Unsupported index format or sketch parameters')

    def close(self):
        self.con.close()

    def search(self, peptide, mode, min_shared=2, min_containment=0.1, top=5, scaled=10):
        if scaled < self.meta['scaled']:
            raise ValueError('Cannot search finer than index scaled; rebuild the index')
        k = self.meta[mode + '_k']
        query = sketch(peptide, mode, k, scaled)
        hashes = list(query.hashes)
        if not hashes:
            return []
        votes = Counter()
        for start in range(0, len(hashes), 800):
            chunk = [f'{h:016x}' for h in hashes[start:start+800]]
            sql = 'SELECT protein_id,count(*) FROM postings WHERE mode=? AND hash IN (' + ','.join('?' for _ in chunk) + ') GROUP BY protein_id'
            votes.update(dict(self.con.execute(sql, [mode, *chunk])))
        result = []
        # All qualifying candidates have known shared-hash counts; rank before reconstruction.
        candidates = sorted(votes.items(), key=lambda x: (-x[1], x[0]))
        for protein_id, shared in candidates:
            if shared < min_shared or shared / len(query) < min_containment:
                continue
            description, seq = self.con.execute('SELECT description,sequence FROM proteins WHERE id=?', (protein_id,)).fetchone()
            subject = sketch(seq, mode, k, scaled)
            containment = query.contained_by(subject)
            subject_words = {word for _, word in encoded_kmers(seq, mode, k)}
            matched = [(i, word) for i, word in encoded_kmers(peptide, mode, k) if word in subject_words]
            # Evidence is encoded-string verified, not just hash equality.
            unique = len({word for _, word in matched})
            if unique < min_shared:
                continue
            positions = sorted({j for i, _ in matched for j in range(i, i+k)})
            result.append(dict(protein_id=protein_id, description=description,
                containment=containment, shared_hashes=shared, query_hashes=len(query),
                target_hashes=len(subject), shared_encoded_kmers=unique,
                shared_query_words=sorted({peptide[i:i+k] for i, _ in matched}),
                matched_positions=positions, k=k, mode=mode, scaled=scaled))
            if len(result) >= top:
                break
        return result


def read_tsv(path):
    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        if not reader.fieldnames:
            raise ValueError(f'No TSV header: {path}')
        return list(reader)


def number(value, label):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f'Non-finite {label}')
    return x


def load_events(scores, compactors, q_column, sequence_column, effect_column, top, fdr):
    selected = {}
    rows = read_tsv(scores)
    if rows and any(c not in rows[0] for c in ['anchor', q_column]):
        raise ValueError(f'Scores require anchor and {q_column}; specify --q-column for your SPLASH version')
    for row in rows:
        anchor = row['anchor'].upper()
        if not anchor or set(anchor) - set('ACGT'):
            raise ValueError('Score anchor must be an ACGT nucleotide sequence')
        q = number(row[q_column], q_column)
        if not 0 <= q <= 1:
            raise ValueError('Adjusted p-values must be between 0 and 1')
        if anchor in selected:
            raise ValueError(f'Duplicate anchor in score table: {anchor}')
        if q <= fdr:
            effect = number(row[effect_column], effect_column) if row.get(effect_column) else None
            selected[anchor] = dict(anchor=anchor, qvalue=q, effect_size=effect, sequences=[], hits=[], counts=[])
    events = sorted(selected.values(), key=lambda e: (e['qvalue'], e['anchor']))[:top]
    lookup = {e['anchor']: e for e in events}
    for row in read_tsv(compactors):
        if 'anchor' not in row or sequence_column not in row:
            raise ValueError(f'Compactors require anchor and {sequence_column}; use --sequence-column')
        anchor, seq = row['anchor'].upper(), row[sequence_column].upper()
        if anchor not in lookup:
            continue
        if not seq or set(seq) - set('ACGTRYSWKMBDHVN'):
            raise ValueError(f'Invalid compactor sequence for {anchor}')
        if anchor not in seq and str(Seq(anchor).reverse_complement()) not in seq:
            raise ValueError(f'Compactor does not contain its anchor: {anchor}')
        if seq not in lookup[anchor]['sequences']:
            lookup[anchor]['sequences'].append(seq)
    return events


def analyze(scores, compactors, database, output, q_column='pval_opt_corrected',
            sequence_column='compactor', effect_column='effect_size_bin', top=100,
            fdr=0.05, min_aa=20, translation_table=1, min_shared=2,
            min_containment=0.1, top_hits=5, counts=None, demo=False, scales=(10, 100)):
    if not 0 <= fdr <= 1 or not 0 <= min_containment <= 1:
        raise ValueError('FDR and containment must be between 0 and 1')
    if min(top, min_aa, min_shared, top_hits) < 1:
        raise ValueError('Count and length parameters must be positive')
    if not scales or any(x < 1 for x in scales):
        raise ValueError('scales must be positive integers')
    scales = tuple(dict.fromkeys(scales))
    started = time.monotonic()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output directory must be new or empty')
    events = load_events(scores, compactors, q_column, sequence_column, effect_column, top, fdr)
    if counts:
        lookup = {e['anchor']: e for e in events}
        for row in read_tsv(counts):
            if not {'anchor', 'sample', 'target', 'count'} <= row.keys():
                raise ValueError('Counts TSV requires anchor, sample, target, count; optional condition')
            n = number(row['count'], 'count')
            if n < 0 or n != int(n):
                raise ValueError('Counts must be nonnegative integers')
            if row['anchor'] in lookup:
                lookup[row['anchor']]['counts'].append(dict(sample=row['sample'], target=row['target'], count=int(n), condition=row.get('condition', '')))
    index = ProteinIndex(database)
    peptides = []
    comparison = {(mode, scale): dict(mode=mode, scaled=scale, queries=0, zero_hash_queries=0, eligible_queries=0, reported_hits=0, seconds=0.0) for mode in MODES for scale in scales}
    try:
        for event in events:
            event['peptide_count'] = 0
            event['sketch_signal'] = []
            for si, dna in enumerate(event['sequences']):
                for pi, fragment in enumerate(translate_segments(dna, min_aa, translation_table)):
                    fragment_id = f"event{events.index(event)+1}_extension{si+1}_{fragment['strand']}{fragment['frame']}_{pi+1}"
                    peptides.append((fragment_id, fragment['peptide']))
                    event['peptide_count'] += 1
                    for mode in MODES:
                        for scaled in scales:
                            tick = time.monotonic()
                            stat = comparison[(mode, scaled)]
                            mh = sketch(fragment['peptide'], mode, index.meta[mode+'_k'], scaled)
                            event['sketch_signal'].append(dict(peptide_id=fragment_id, mode=mode, scaled=scaled,
                                query_hashes=len(mh), aa_length=len(fragment['peptide']),
                                eligible=len(mh)>=min_shared))
                            found = index.search(fragment['peptide'], mode, min_shared, min_containment, top_hits, scaled)
                            stat['queries'] += 1
                            stat['zero_hash_queries'] += int(len(mh)==0)
                            stat['eligible_queries'] += int(len(mh)>=min_shared)
                            stat['reported_hits'] += len(found)
                            stat['seconds'] += time.monotonic()-tick
                            for hit in found:
                                event['hits'].append(dict(**hit, **fragment, sequence_index=si, peptide_id=fragment_id))
            event['hits'].sort(key=lambda h: (h['mode'], -h['containment'], -h['shared_hashes'], h['protein_id']))
            # Top hits per encoding, across all translations and extensions. Preserve ties deterministically.
            event['hits'] = [h for mode in MODES for scaled in scales for h in [h for h in event['hits'] if h['mode']==mode and h['scaled']==scaled][:top_hits]]
            event['status'] = ('no extension' if not event['sequences'] else 'no peptide above length cutoff' if not event['peptide_count'] else 'candidate matches' if event['hits'] else 'no match at current settings')
        report = dict(schema_version=1, demo=demo, events=events, database=index.meta,
            search_comparison=[{**row, "seconds": round(row["seconds"],6)} for row in comparison.values()],
            parameters=dict(q_column=q_column, sequence_column=sequence_column, effect_column=effect_column,
                top=top, fdr=fdr, min_aa=min_aa, translation_table=translation_table,
                scales=scales, min_shared=min_shared, min_containment=min_containment, top_hits=top_hits),
            provenance=dict(scores_sha256=sha256(scores), compactors_sha256=sha256(compactors),
                counts_sha256=sha256(counts) if counts else None,
                database_sha256=sha256(database), sourmash=version('sourmash'), biopython=version('biopython'),
                created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())),
            search_seconds=round(time.monotonic()-started, 3),
            interpretation='Exploratory candidate matches. Containment is not identity, an alignment, a p-value, or functional confidence. SPLASH q-values describe sequence variation, not protein matches or treatment differential expression. Thresholds are uncalibrated.')
    finally:
        index.close()
    output.mkdir(parents=True, exist_ok=True)
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    (output/'peptides.fasta').write_text(''.join(f'>{i}\n{p}\n' for i,p in peptides))
    with (output/'hits.tsv').open('w', newline='') as f:
        fields = ['anchor', 'qvalue', 'mode', 'scaled', 'protein_id', 'containment', 'shared_hashes', 'query_hashes', 'k', 'strand', 'frame', 'nt_start', 'nt_end', 'peptide', 'description']
        writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        for event in events:
            for hit in event['hits']:
                writer.writerow(dict(anchor=event['anchor'], qvalue=event['qvalue'], **hit))
    template = Path(__file__).with_name('viewer.html').read_text()
    payload = json.dumps(report, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (output/'report.html').write_text(template.replace('__REPORT_DATA__', payload))
    return report
