"""paper/table_counts.tex: per-field counts from results/perfect/summary.tsv and h_K, gamma from results/gamma_K2.tsv."""
import csv
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = list(csv.DictReader(open(os.path.join(ROOT, "results/perfect/summary.tsv")), delimiter="\t"))
T = {r["D"]: r for r in csv.DictReader(open(os.path.join(ROOT, "results/gamma_K2.tsv")), delimiter="\t")}
k = 3
out = [r"\begin{small}", r"\begin{longtable}{" + "|".join(["rrrrr"] * k) + "}",
       r"\caption{Per field: $D$, $h_K$, $\gK$ (rounded), number of perfect and of extreme classes \CERT.}\label{tab:all}\\",
       r"\toprule", " & ".join([r"$D$ & $h$ & $\gK$ & perf & ext"] * k) + r"\\", r"\midrule", r"\endhead"]
cells = ["%s & %s & %.4g & %s & %s" % (r["D"], r["h_K"], float(T[r["D"]]["gamma"]), r["n_perfect"], r["n_extreme"]) for r in S]
rows = (len(cells) + k - 1) // k
for i in range(rows):
    c = [cells[i + j * rows] if i + j * rows < len(cells) else " & & & & " for j in range(k)]
    out.append(" & ".join(c) + r"\\")
out += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]
open(os.path.join(ROOT, "paper/table_counts.tex"), "w").write("\n".join(out) + "\n")
