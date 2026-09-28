"""Sweep gamma_{K,2} over real quadratic fields with h_K = 1; one line per field
appended to the output file (tab separated)."""
import sys, time, signal, mpmath
from hermite_humbert import compute
from field import RealQuadraticField

def handler(s, f):
    raise TimeoutError
signal.signal(signal.SIGALRM, handler)

def main():
    out, lim = sys.argv[1], int(sys.argv[2])
    Ds = [int(x) for x in sys.argv[3:]]
    with open(out, "a") as fh:
        for D in Ds:
            F = RealQuadraticField(D)
            t0 = time.time()
            signal.alarm(lim)
            try:
                r = compute(D, verbose=False, exact=True, verify=True)
                ex = r["exact"]
                st = "certified" if ex and ex["proved"] else "upper<=%.15f" % r["certificate"]["upper"]
                c0 = r["classes"][0]
                line = "%d\t%d\t%d\t%s\t%s\t%s\t%d\t%d\t%s\t%.0f\t%s" % (
                    D, r["d"], r["h"], F.fmt(F.fundamental_unit()), mpmath.nstr(r["gamma"], 25),
                    c0["closed_form"][1], c0["n_min"], len(r["classes"]), st, time.time() - t0,
                    ";".join(r["notes"]))
            except TimeoutError:
                line = "%d\t%d\t-\t%s\t-\t-\t-\t-\ttimeout(%ds)\t%.0f\t" % (D, F.disc, F.fmt(F.fundamental_unit()), lim, time.time() - t0)
            except Exception as e:
                line = "%d\t%d\t-\t%s\t-\t-\t-\t-\terror:%r\t%.0f\t" % (D, F.disc, F.fmt(F.fundamental_unit()), e, time.time() - t0)
            finally:
                signal.alarm(0)
            fh.write(line + "\n")
            fh.flush()

if __name__ == "__main__":
    main()
