"""Render the oyster map with nf-metro2.1.0 and place its legend before the routes."""
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as E
root=Path(__file__).resolve().parent
source=root/'metro/workflow-overview.mmd'
S='http://www.w3.org/2000/svg';E.register_namespace('',S)
with tempfile.TemporaryDirectory() as tmp:
    raw=Path(tmp)/'map.svg'
    subprocess.run([sys.argv[1],'render',str(source),'-o',str(raw),'--mode','light','--strict','--row-align','top','--section-x-gap','90'],check=True)
    old=E.parse(raw).getroot();width=float(old.attrib['width']);height=float(old.attrib['height'])+120
    svg=E.Element(f'{{{S}}}svg',dict(width=str(width),height=str(height),viewBox=f'0 0 {width} {height}',style='color-scheme:light'))
    E.SubElement(svg,f'{{{S}}}rect',dict(width='100%',height='100%',fill='white'))
    title=next(e for e in old if e.attrib.get('class')=='nf-metro-title')
    E.SubElement(svg,f'{{{S}}}text',dict(x='25',y='30',**{'font-family':'Arial','font-size':'23','font-weight':'bold'})).text=title.text
    x=25
    for line in source.read_text().splitlines():
        m=re.match(r'%%metro line: (.*?) \| (.*?) \| (#[a-fA-F0-9]+)',line)
        if m:
            attrs=dict(d=f'M{x},60 h30',stroke=m[3],**{'stroke-width':'5'})
            if line.endswith('| dashed'):attrs['stroke-dasharray']='12,8'
            E.SubElement(svg,f'{{{S}}}path',attrs)
            E.SubElement(svg,f'{{{S}}}text',dict(x=str(x+40),y='65',**{'font-family':'Arial','font-size':'15'})).text=m[2]
            x+=len(m[2])*8+65
    E.SubElement(svg,f'{{{S}}}text',dict(x='25',y='95',**{'font-family':'Arial','font-size':'15'})).text='Circle: program. Document: file. Dashed purple: planned Swiss-Prot branch. Read left to right.'
    g=E.SubElement(svg,f'{{{S}}}g',dict(transform='translate(0,120)'))
    for e in old:
        if e.attrib.get('class')=='nf-metro-legend-bg':break
        if e is title or e.attrib.get('class')=='nf-metro-bg':continue
        g.append(e)
    # Keep DE-kupl on its own lower lane across section boundaries. The
    # automatic router otherwise climbs to the SPLASH trunk at every port.
    for path in g.iter(f'{{{S}}}path'):
        if path.attrib.get('data-line-id') != 'dekupl':
            continue
        coords = re.findall(r'[-+]?[0-9]*\.?[0-9]+', path.attrib.get('d', ''))
        if coords and max(map(float, coords[::2])) <= 1154.0:
            path.set('d', re.sub(r'([-+]?[0-9]*\.?[0-9]+),([-+]?[0-9]*\.?[0-9]+)',
                                lambda m: m[1] + ',178.4', path.attrib['d']))
    E.ElementTree(svg).write(root/'metro/workflow-overview.svg',encoding='unicode')
