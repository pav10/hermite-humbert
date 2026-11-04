# usage: python analysis/robustness.py D1 D2 ... > results/perfect/robustness.txt (body; run from the repository root)
# rerun with half the leaf width (different boxes and Newton starts); compare the class lists
import json, sys
from multiprocessing import Pool
from hermite_humbert.perfect import perfect_forms, record
def one(D):
    r = record(perfect_forms(D, w_min=0.025), None)
    old = json.load(open("results/perfect/D%d.json" % D))
    k = lambda c: (c["gamma"][:25], c["n_min"], c["eutaxy"], c["aut"])
    a, b = sorted(map(k, old["classes"])), sorted(map(k, r["classes"]))
    return D, old["n_perfect"], r["n_perfect"], a == b
if __name__ == "__main__":
    Ds = [int(x) for x in sys.argv[1:]]
    with Pool(4) as p:
        for res in p.imap_unordered(one, Ds):
            print(*res, flush=True)
