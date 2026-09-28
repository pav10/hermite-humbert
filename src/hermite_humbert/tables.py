"""Render a sweep table (TSV) as Markdown, for results/README.md.

    python -m hermite_humbert.tables results/gamma_K2.tsv > results/README.md
"""
import csv
import sys


def _fmt_status(st):
    if st.startswith("certified"):
        return "certified"
    if st.startswith("timeout"):
        return "open (" + st + ")"
    return st


def render(path):
    with open(path) as fh:
        rows = sorted(csv.DictReader(fh, delimiter="\t"), key=lambda r: int(r["D"]))
    done = [r for r in rows if r["status"].startswith("certified")]
    open_ = [r for r in rows if not r["status"].startswith("certified")]
    Ds = [int(r["D"]) for r in rows]
    out = ["# Hermite–Humbert constants γ_{K,2}, K = Q(√D)", ""]
    out.append("Generated from `%s` by `python -m hermite_humbert.tables`. "
               "All squarefree D in [%d, %d] were attempted: %d certified (exact mode, "
               "certificate re-checked with outward rounding), %d open."
               % (path.split("/")[-1], min(Ds), max(Ds), len(done), len(open_)))
    out.append("")
    out.append("γ is given to 25 significant digits; the per-field JSON files in "
               "`fields/` hold 50 digits, every extreme class found, the normal forms "
               "and minimal vectors.  `#M` = number of minimal vectors (up to units) of "
               "the critical form, `#cl` = number of GL₂(O_K)-classes of extreme forms "
               "found (the list of non-critical ones is not certified complete).")
    out.append("")
    out.append("| D | d_K | h_K | ε₀ | γ_{K,2} | closed form | #M | #cl | time (s) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for r in done:
        cf = r["closed_form"] if r["closed_form"] not in ("?", "-", "") else "not found by PSLQ"
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["D"], r["d_K"], r["h_K"], r["eps0"], r["gamma"], cf.replace("*", "·"),
            r["#M"], r["#classes"], r["seconds"]))
    if open_:
        out += ["", "## Not (yet) certified", "",
                "| D | d_K | ε₀ | status | notes |", "|---|---|---|---|---|"]
        for r in open_:
            out.append("| %s | %s | %s | %s | %s |" % (r["D"], r["d_K"], r["eps0"],
                                                       _fmt_status(r["status"]), r["notes"]))
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    sys.stdout.write(render(sys.argv[1] if len(sys.argv) > 1 else "results/gamma_K2.tsv"))
