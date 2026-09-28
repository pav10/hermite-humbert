"""Where fields come from: LMFDB, rvisser7's field lists, simple text files.

All readers yield (label, coeffs) with coeffs ascending and monic (LMFDB
convention, e.g. [1, -2, -1, 1] for x^3 - x^2 - 2x + 1).
"""
import ast
import json
import os
import re
import time
import urllib.parse
import urllib.request

LMFDB_API = 'https://www.lmfdb.org/api/nf_fields/'
VISSER_RAW = 'https://raw.githubusercontent.com/rvisser7/indecomposables/main/'


def _disc(coeffs):
    from .field import pari
    return int(pari.nfdisc(pari.Pol(list(reversed(coeffs)))))


# ----------------------------------------------------------------------------
# rvisser7/indecomposables: totally_real_fields/degN.txt  (lmfdb_index|coeffs|monogenic)

def read_visser_fields(path, degree=None, disc_min=None, disc_max=None, labels=True):
    """Fields of a totally_real_fields/degN.txt file (coefficients without the
    leading 1).  Labels n.n.disc.index are rebuilt (needs one nfdisc per field)."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('lmfdb') or '|' not in line:
                continue
            idx, co, *_ = line.split('|')
            coeffs = [int(t) for t in co.split(',')] + [1]
            n = len(coeffs) - 1
            if degree and n != degree:
                continue
            label = None
            if labels or disc_min or disc_max:
                D = _disc(coeffs)
                if disc_min and D < disc_min:
                    continue
                if disc_max and D > disc_max:
                    break                               # files are sorted by discriminant
                label = '%d.%d.%d.%s' % (n, n, D, idx)
            out.append((label, coeffs))
    return out


def fetch_visser_fields(degree, dest):
    url = VISSER_RAW + 'totally_real_fields/deg%d.txt' % degree
    urllib.request.urlretrieve(url, dest)
    return dest


def read_visser_reference(path):
    """{label: (num_indecomposables, max_norm)} from rvisser7's data/degN.txt."""
    out = {}
    with open(path) as fh:
        for line in fh:
            p = line.strip().split('|')
            if len(p) < 4 or not re.match(r'^\d+\.\d+\.\d+\.\d+$', p[0]):
                continue
            try:
                out[p[0]] = (int(p[-3]), int(p[-2]))
            except ValueError:
                continue
    return out


# ----------------------------------------------------------------------------
# LMFDB

def fetch_lmfdb(degree, disc_min=None, disc_max=None, max_fields=None, page=1000,
                pause=1.0, verbose=True):
    """Totally real fields of the given degree from the LMFDB API, sorted by |disc|.

    Uses the documented API conventions (typed values 'i3', python-literal
    queries 'py{...}', paging with _offset/_max_count, 'next' links); the
    discriminant range is also re-checked locally.
    """
    query = {'degree': 'i%d' % degree, 'r2': 'i0', '_format': 'json',
             '_fields': 'label,coeffs,disc_abs', '_sort': 'disc_abs,label',
             '_max_count': str(page)}
    cond = {}
    if disc_min:
        cond['$gte'] = int(disc_min)
    if disc_max:
        cond['$lte'] = int(disc_max)
    if cond:
        query['disc_abs'] = 'py' + json.dumps(cond)
    out, offset = [], 0
    url = LMFDB_API + '?' + urllib.parse.urlencode(query)
    while url:
        req = urllib.request.Request(url, headers={'User-Agent': 'sails (research; indecomposables)'})
        with urllib.request.urlopen(req, timeout=120) as fh:
            js = json.load(fh)
        data = js.get('data', [])
        for rec in data:
            D = int(rec.get('disc_abs', 0) or 0)
            if (disc_min and D < disc_min) or (disc_max and D > disc_max):
                continue
            out.append((rec['label'], [int(t) for t in rec['coeffs']]))
        if verbose:
            print('lmfdb: %d fields' % len(out), flush=True)
        if max_fields and len(out) >= max_fields:
            return out[:max_fields]
        nxt = js.get('next')
        if nxt:
            url = nxt if nxt.startswith('http') else urllib.parse.urljoin(LMFDB_API, nxt)
        elif len(data) >= page:
            offset += page
            query['_offset'] = str(offset)
            url = LMFDB_API + '?' + urllib.parse.urlencode(query)
        else:
            url = None
        time.sleep(pause)
    return out


def read_lmfdb_download(path):
    """An LMFDB search download (Sage / GP / Python / JSON export) with at least the
    columns label and coeffs (ascending polynomial coefficients)."""
    text = open(path).read()
    if path.endswith('.json'):
        js = json.loads(text)
        rows = js['data'] if isinstance(js, dict) else js
        return [(r['label'], [int(t) for t in r['coeffs']]) for r in rows]
    m = re.search(r'columns\s*=\s*(\[[^\]]*\])', text)
    d = re.search(r'data\s*=\s*(\[.*\])\s*$', text, re.S)
    if m and d:
        cols = ast.literal_eval(m.group(1))
        rows = ast.literal_eval(d.group(1))
        il, ic = cols.index('label'), cols.index('coeffs')
        return [(r[il], [int(t) for t in r[ic]]) for r in rows]
    return read_labelled_polynomials(path)


def read_labelled_polynomials(path):
    """Generic fallback, also for the notebook's make_data() .sage files: every
    LMFDB label followed (on the same line or record) by a coefficient list or a
    polynomial in x."""
    from .field import pari
    text = open(path).read()
    out = []
    pat = re.compile(r"""['"](\d+\.\d+\.\d+\.\d+)['"]\s*[,:]?\s*(?:['"]?\w+['"]?\s*:\s*)?(\[[-\d,\s]+\]|[^,}\]\n]+)""")
    for label, poly in pat.findall(text):
        poly = poly.strip().strip('\'"')
        if poly.startswith('['):
            coeffs = [int(t) for t in ast.literal_eval(poly)]
        else:
            p = pari(poly.replace('^', '^'))
            coeffs = [int(t) for t in pari.Vecrev(p)]
        out.append((label, coeffs))
    return out


def read_simple(path):
    """Lines 'label|c0,c1,...,cn' or 'c0,c1,...,cn' (full ascending coefficients)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '|' in line:
            label, co = line.split('|')[:2]
        else:
            label, co = None, line
        out.append((label or None, [int(t) for t in co.strip('[]').split(',')]))
    return out


def load_fields(path, degree=None, disc_min=None, disc_max=None):
    """Dispatch on the file format."""
    head = open(path).readline()
    if head.startswith('lmfdb_index'):
        return read_visser_fields(path, degree, disc_min, disc_max)
    if path.endswith(('.sage', '.gp', '.py', '.json')):
        fields = read_lmfdb_download(path)
    else:
        fields = read_simple(path)
    res = []
    for label, co in fields:
        n = len(co) - 1
        if degree and n != degree:
            continue
        if disc_min or disc_max:
            D = int(label.split('.')[2]) if label else _disc(co)
            if (disc_min and D < disc_min) or (disc_max and D > disc_max):
                continue
        res.append((label, co))
    return res
