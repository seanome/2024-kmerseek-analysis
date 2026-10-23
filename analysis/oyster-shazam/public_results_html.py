"""Render matched-scale public search coverage without claiming whole-archive coverage."""
from html import escape


def render(data):
    n=data['extended_records_searched'];completed=len(data['completed_batches']);rows=[];bars=[];empty=[]
    maximum=max(s.get('matches_10',s.get('match_rows_10',0)) for s in data['encodings'].values())
    limit=max(100000,((maximum+99999)//100000)*100000)
    for i,(mol,name) in enumerate([('dna','DNA k31'),('protein','Protein k7'),('dayhoff','Dayhoff k12'),('hp','HP k30')]):
        s=data['encodings'][mol]
        ten=s.get('matches_10',s.get('match_rows_10'));hundred=s.get('matches_100',s.get('match_rows_100'))
        lost=s.get('lost_pairs',ten-hundred)
        e10=s.get('empty_10',s.get('empty_sketches_10'));e100=s.get('empty_100',s.get('empty_sketches_100'))
        rows.append(f'<tr><td>{name}</td><td>{ten:,}</td><td>{hundred:,}</td><td>{lost:,}</td></tr>')
        empty.append(f'<tr><td>{name}</td><td>{n-e10:,}</td><td>{n-e100:,}</td><td>{e100:,}</td></tr>')
        y=35+i*53
        bars.append(f'<text x="125" y="{y+10}" text-anchor="end">{name}</text>')
        for delta,value,colour in [(0,ten,'#0072B2'),(17,hundred,'#009E73')]:
            width=value/limit*570
            bars.append(f'<rect x="140" y="{y+delta}" width="{width:.2f}" height="11" fill="{colour}"/><text x="{150+width:.2f}" y="{y+delta+10}">{value:,}</text>')
    examples=[]
    for r in data.get('lost_examples',[]):
        examples.append('<tr>'+''.join(f'<td>{escape(str(r.get(k,"")))}</td>' for k in ['query_id','run','encoding','containment_scaled10'])+'</tr>')
    plot=f'<div class="key"><span><i class="sw" style="background:#0072B2"></i>scaled 10</span><span><i class="sw" style="background:#009E73"></i>scaled 100</span></div><svg class="plot" viewBox="0 0 900 300" role="img" aria-label="Candidate query-run match counts at scaled10 and scaled100">'+''.join(bars)+f'<line x1="140" x2="710" y1="260" y2="260"/><text x="140" y="280">0</text><text x="425" y="280" text-anchor="middle">{limit//2:,}</text><text x="710" y="280" text-anchor="middle">{limit:,} pairs</text></svg>'
    return f'''<div class="card" id="public-results"><h3>Public matches: scaled 10 versus scaled 100</h3>
<p>{n:,} extended DNA records have completed searches against eight public runs from eight studies ({completed} of 22 query batches). These reference runs are the earlier <strong>untrimmed pilot</strong>; the new trimmed full-archive comparison is separate. This is not coverage of all 17,623 runs.</p>
<p>A pair is one query record and one public run with at least one shared hash. These are candidate occurrences, not validated functions or distinct genes. Reduced-alphabet matches can occur by chance.</p>
{plot}<div class="scroll"><table><thead><tr><th>Encoding</th><th>Pairs at 10</th><th>Pairs at 100</th><th>Net fewer pairs at 100</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<details><summary>Which queries can be searched?</summary><p>An empty sketch is unsearchable at that scale. It is not evidence that a sequence is absent.</p><div class="scroll"><table><thead><tr><th>Encoding</th><th>Nonempty at 10</th><th>Nonempty at 100</th><th>Empty at 100</th></tr></thead><tbody>{''.join(empty)}</tbody></table></div></details>
<details><summary>Examples of exact query–run pairs recovered at 10 but missing at 100</summary><p>DNA examples from the first completed batch, checked against its complete scaled100 match table. Query IDs include the original anchor. The independent full-pair audit is compiling the complete downloadable loss table.</p><div class="scroll"><table><thead><tr><th>Query ID</th><th>Public run</th><th>Encoding</th><th>Containment at 10</th></tr></thead><tbody>{''.join(examples)}</tbody></table></div></details>
<p class="small muted">{escape(data.get('audit_status','Independent result hashes, pair counts and nested-match consistency checked.'))}</p><a href="public-match-summary.json" download>Completed public-match summary · JSON</a></div>'''
