"""Command line interface.

  python -m sails field 1,-2,-1,1 [--label 3.3.49.1] [--json]
  python -m sails run --source data/deg3.txt --out results/deg3.txt --jobs 8
  python -m sails fetch-lmfdb --degree 3 --disc-max 100000 --out data/lmfdb_deg3.txt
  python -m sails fetch-visser --degree 3 --out data/deg3.txt
  python -m sails compare results/deg3.txt --reference ../indecomposables/data/deg3.txt
"""
import argparse
import json
import os
import sys
import time
import traceback


def _parse_coeffs(s):
    return [int(t) for t in s.strip('[]').split(',')]


# ----------------------------------------------------------------------------
# worker

_OPTS = {}


def _init_worker(opts):
    _OPTS.update(opts)


class FieldTimeout(Exception):
    pass


def _work(task):
    label, coeffs = task
    from .compute import compute_field
    timeout = _OPTS.get('timeout') or 0
    alarm = cancel = None
    if timeout:
        try:
            from cysignals.alarm import alarm, cancel_alarm as cancel
        except ImportError:                            # pragma: no cover
            import signal

            def _h(*_):
                raise FieldTimeout()
            signal.signal(signal.SIGALRM, _h)
            alarm, cancel = signal.alarm, lambda: signal.alarm(0)
        alarm(timeout)
    t0 = time.time()
    try:
        res = compute_field(coeffs, label=label, tp_only=_OPTS.get('tp_only', False),
                            seed_bound=_OPTS.get('seed_bound'), certify=_OPTS.get('certify', True),
                            lll_units=_OPTS.get('lll_units', True))
        line = res.gordon_line()
        rec = json.dumps(res.record(), separators=(',', ':')) if _OPTS.get('json') else None
        if timeout:
            cancel()
        return label, coeffs, line, rec, None, time.time() - t0
    except BaseException as exc:                        # noqa: BLE001 (AlarmInterrupt is a KeyboardInterrupt)
        if timeout:
            try:
                cancel()
            except Exception:
                pass
        name = type(exc).__name__
        if name in ('AlarmInterrupt', 'FieldTimeout'):
            err = 'timeout after %ss' % timeout
        elif isinstance(exc, KeyboardInterrupt):
            raise
        else:
            err = '%s: %s' % (name, exc)
            if _OPTS.get('traceback'):
                err += '\n' + traceback.format_exc()
        return label, coeffs, None, None, err, time.time() - t0


# ----------------------------------------------------------------------------

def cmd_field(a):
    from .compute import compute_field, COLUMNS
    res = compute_field(_parse_coeffs(a.coeffs), label=a.label, tp_only=a.tp_only,
                        seed_bound=a.seed_bound, certify=not a.no_certify,
                        lll_units=not a.gordon_units)
    if a.json:
        print(json.dumps(res.record(), indent=1))
    else:
        if a.header:
            print('|'.join(COLUMNS))
        print(res.gordon_line())
    print('# %s' % json.dumps(res.timings), file=sys.stderr)


def _done_labels(path):
    done = set()
    if path and os.path.exists(path):
        for line in open(path):
            if line.strip() and not line.startswith('#'):
                done.add(line.split('|', 1)[0].split('\t', 1)[0])
    return done


def cmd_run(a):
    import multiprocessing as mp
    from .sources import load_fields
    fields = load_fields(a.source, degree=a.degree, disc_min=a.disc_min, disc_max=a.disc_max)
    if a.shard:
        i, m = [int(t) for t in a.shard.split('/')]
        fields = [f for k, f in enumerate(fields) if k % m == i]
    if a.limit:
        fields = fields[:a.limit]
    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    err_path = out + '.errors'
    json_path = (out + '.jsonl') if a.json else None
    done = _done_labels(out) | (set() if a.retry_errors else _done_labels(err_path))
    todo = [f for f in fields if (f[0] or ','.join(map(str, f[1]))) not in done]
    print('fields: %d, already done: %d, to do: %d' % (len(fields), len(fields) - len(todo), len(todo)),
          flush=True)
    opts = {'timeout': a.timeout, 'tp_only': a.tp_only, 'seed_bound': a.seed_bound,
            'certify': not a.no_certify, 'lll_units': not a.gordon_units, 'json': a.json,
            'traceback': a.traceback}
    fo = open(out, 'a')
    fe = open(err_path, 'a')
    fj = open(json_path, 'a') if json_path else None
    t0 = time.time()
    n_ok = n_err = 0
    slowest = (0.0, None)
    if a.jobs > 1:
        ctx = mp.get_context('fork')
        pool = ctx.Pool(a.jobs, initializer=_init_worker, initargs=(opts,), maxtasksperchild=500)
        it = pool.imap_unordered(_work, todo, chunksize=1)
    else:
        _init_worker(opts)
        pool = None
        it = map(_work, todo)
    try:
        for k, (label, coeffs, line, rec, err, dt) in enumerate(it, 1):
            if err is None:
                fo.write(line + '\n')
                fo.flush()
                if fj:
                    fj.write(rec + '\n')
                    fj.flush()
                n_ok += 1
            else:
                fe.write('%s|%s|%s\n' % (label, ','.join(map(str, coeffs)), err.replace('\n', ' / ')))
                fe.flush()
                n_err += 1
            if dt > slowest[0]:
                slowest = (dt, label)
            if k % a.progress == 0 or k == len(todo):
                el = time.time() - t0
                print('%d/%d done (%d errors) in %.1fs, %.3fs/field, slowest %s (%.1fs)'
                      % (k, len(todo), n_err, el, el / k, slowest[1], slowest[0]), flush=True)
    finally:
        if pool is not None:
            pool.terminate()
        fo.close()
        fe.close()
        if fj:
            fj.close()


def cmd_fetch_lmfdb(a):
    from .sources import fetch_lmfdb
    fields = fetch_lmfdb(a.degree, a.disc_min, a.disc_max, a.max_fields)
    with open(a.out, 'w') as fh:
        for label, co in fields:
            fh.write('%s|%s\n' % (label, ','.join(map(str, co))))
    print('wrote %d fields to %s' % (len(fields), a.out))


def cmd_fetch_visser(a):
    from .sources import fetch_visser_fields
    fetch_visser_fields(a.degree, a.out)
    print('wrote %s' % a.out)


def cmd_compare(a):
    """Compare the totally positive column of a results file with reference tables."""
    from .sources import read_visser_reference
    ref = {}
    for r in a.reference:
        ref.update(read_visser_reference(r))
    n = ok = 0
    bad = []
    for line in open(a.results):
        p = line.rstrip('\n').split('|')
        if len(p) < 11:
            continue
        label = p[0]
        if label not in ref:
            continue
        cnt = int(p[4].strip('[]').split(',')[0])
        mx = int(p[6].strip('[]').split(',')[0])
        n += 1
        if (cnt, mx) == ref[label]:
            ok += 1
        else:
            bad.append((label, (cnt, mx), ref[label]))
    print('compared %d fields: %d agree, %d differ' % (n, ok, len(bad)))
    for b in bad[:50]:
        print('  %s ours %s reference %s' % b)
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sails', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p):
        p.add_argument('--tp-only', action='store_true', help='only the totally positive sail')
        p.add_argument('--seed-bound', type=int, default=None,
                       help='norm bound for seed elements (default floor(disc^(1/n)))')
        p.add_argument('--no-certify', action='store_true', help='skip bnfcertify (units under GRH)')
        p.add_argument('--gordon-units', action='store_true',
                       help="use the notebook's TPFU basis without LLL reduction")

    p = sub.add_parser('field', help='one field, output line on stdout')
    p.add_argument('coeffs', help='ascending coefficients, e.g. 1,-2,-1,1')
    p.add_argument('--label')
    p.add_argument('--json', action='store_true')
    p.add_argument('--header', action='store_true')
    common(p)
    p.set_defaults(func=cmd_field)

    p = sub.add_parser('run', help='batch computation with resume')
    p.add_argument('--source', required=True,
                   help='field list: rvisser7 degN.txt, LMFDB download, or label|coeffs lines')
    p.add_argument('--out', required=True)
    p.add_argument('--degree', type=int)
    p.add_argument('--disc-min', type=int)
    p.add_argument('--disc-max', type=int)
    p.add_argument('--limit', type=int)
    p.add_argument('--shard', help='i/m: only fields with index = i mod m')
    p.add_argument('--jobs', type=int, default=os.cpu_count() or 1)
    p.add_argument('--timeout', type=int, default=0, help='seconds per field (0 = none)')
    p.add_argument('--json', action='store_true', help='also write OUT.jsonl with full data')
    p.add_argument('--retry-errors', action='store_true')
    p.add_argument('--traceback', action='store_true')
    p.add_argument('--progress', type=int, default=100)
    common(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('fetch-lmfdb', help='download totally real fields from the LMFDB API')
    p.add_argument('--degree', type=int, required=True)
    p.add_argument('--disc-min', type=int)
    p.add_argument('--disc-max', type=int)
    p.add_argument('--max-fields', type=int)
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_fetch_lmfdb)

    p = sub.add_parser('fetch-visser', help="download rvisser7's totally_real_fields/degN.txt")
    p.add_argument('--degree', type=int, required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=cmd_fetch_visser)

    p = sub.add_parser('compare', help='compare TP counts / max norms with reference tables')
    p.add_argument('results')
    p.add_argument('--reference', nargs='+', required=True)
    p.set_defaults(func=cmd_compare)

    argv = list(sys.argv[1:] if argv is None else argv)
    # coefficient lists such as -1,-4,0,1 would be taken for options
    argv = [('[%s]' % t) if (t[:1] == '-' and len(t) > 1 and all(c in '-0123456789,' for c in t)
                            and ',' in t) else t for t in argv]
    a = ap.parse_args(argv)
    return a.func(a)
