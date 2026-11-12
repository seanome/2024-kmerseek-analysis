"""Attach result-specific annotation evidence without promoting candidate function."""
import csv
import json

FIELDS = ['result_id', 'analysis_scope', 'status', 'evidence_type', 'mapped_gene_id',
          'gene_name', 'matched_protein_id', 'protein_name', 'pfam_accession',
          'pfam_domain_name', 'go_id', 'go_term_name', 'pfam2go_mapping',
          'source', 'source_version', 'evidence_record', 'query_start', 'query_end']
EVIDENCE = {'direct_sequence', 'direct_domain', 'inherited_gene',
            'inherited_protein', 'pfam2go', 'not_measured'}


def attach(findings, source, out):
    records = json.loads(source.read_text()) if source.exists() else []
    known = {r['id'] for r in findings}
    by_id = {}
    for row in records:
        if row['result_id'] not in known:
            raise ValueError('Annotation has no matching displayed result ID')
        if row['evidence_type'] not in EVIDENCE:
            raise ValueError('Unknown annotation evidence type')
        if row['status'] not in {'completed', 'pending', 'missing'}:
            raise ValueError('Unknown annotation status')
        if row['status'] == 'completed' and not all(row.get(k) for k in
                ('source', 'source_version', 'evidence_record', 'analysis_scope')):
            raise ValueError('Completed annotation lacks provenance')
        by_id.setdefault(row['result_id'], []).append(row)
    exported = []
    for result in findings:
        rows = by_id.get(result['id'], [dict(result_id=result['id'],
            analysis_scope=result.get('analysis_scope', 'historical_prefix_pilot'),
            status='pending', evidence_type='not_measured')])
        result['biological_annotations'] = rows
        exported.extend(rows)
    with (out / 'biological-annotations.tsv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows([{k: r.get(k, '') for k in FIELDS} for r in exported])
    (out / 'biological-annotations.json').write_text(json.dumps(exported, indent=2)+'\n')
