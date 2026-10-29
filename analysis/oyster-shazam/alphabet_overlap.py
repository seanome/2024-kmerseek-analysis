"""Compare named alphabets without conflating query and protein membership."""

def compare(alphabet_a, alphabet_b, matches):
    if alphabet_a == alphabet_b:
        raise ValueError('Choose two different alphabets')
    groups = {}
    for alphabet in (alphabet_a, alphabet_b):
        rows = [r for r in matches if r['alphabet'] == alphabet]
        groups[alphabet] = dict(rows=rows,
            queries={r['query_id'] for r in rows},
            pairs={(r['query_id'], r['protein_id']) for r in rows},
            proteins={r['protein_id'] for r in rows})
    a, b = groups[alphabet_a], groups[alphabet_b]
    counts = {}
    for unit in ('queries', 'pairs', 'proteins'):
        counts[unit] = {f'{alphabet_a} total': len(a[unit]),
            f'{alphabet_b} total': len(b[unit]), 'Both': len(a[unit] & b[unit]),
            f'{alphabet_a} only': len(a[unit]-b[unit]),
            f'{alphabet_b} only': len(b[unit]-a[unit])}
    shared = sorted(a['queries'] & b['queries'])
    return dict(alphabets=[alphabet_a, alphabet_b], counts=counts,
        shared_queries=[dict(query_id=q, matches={
            name: [r for r in group['rows'] if r['query_id']==q]
            for name, group in groups.items()}) for q in shared])
