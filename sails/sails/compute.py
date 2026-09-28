"""Per-field driver and output in the format of the "Simple hybrid" notebook.

Output line (one per field, '|' separated), as data_exporter in
Simple_hybrid_signed_Version_260803.ipynb:

  LMFDB label | Polynomial coefficients | Sail signatures | TPFUs |
  # Indecomposables | # Facets | Max norm (for indecomposables) |
  Max integer distance | Max integer volume | List of indecomposables |
  List of sail faces (vertices, int distance, int volume)

Every per-signature column is a list indexed like the signature column.
Elements are printed as polynomials in the generator a of the defining
polynomial, facet vertices as vectors of power basis coordinates.
"""
import time
from fractions import Fraction

from .field import NumberFieldData, sage_vector_str
from .indecomposables import compute_indecomposables, small_norm_seeds
from .sail import Orthant

COLUMNS = ['LMFDB label', 'Polynomial coefficients', 'Sail signatures', 'TPFUs',
           '# Indecomposables', '# Facets', 'Max norm (for indecomposables)',
           'Max integer distance', 'Max integer volume', 'List of indecomposables',
           'List of sail faces (vertices, int distance, int volume)']


def list_to_string(items):
    """The notebook's list_to_string: '[' + ','.join(str(item)) + ']'."""
    return '[' + ','.join(str(t) for t in items) + ']'


class FieldResult:
    def __init__(self, K, orthants, seed_bound, timings):
        self.K = K
        self.orthants = orthants
        self.seed_bound = seed_bound
        self.timings = timings

    @property
    def label(self):
        return self.K.label or ''

    # --- per signature data ------------------------------------------------

    def facet_rows(self, O):
        """[(vertices in power basis, integer distance, integer volume)], sorted."""
        K = self.K
        rows = []
        for F in O.facets.values():
            verts = sorted(tuple(K.to_power_basis(v)) for v in F.vertices)
            rows.append((F.d, F.volume, verts))
        rows.sort()
        return [(verts, d, vol) for d, vol, verts in rows]

    # --- the notebook's output line ------------------------------------------

    def gordon_line(self):
        K = self.K
        sigs = [K.sail_signatures[O.index] for O in self.orthants]
        counts = [len(O.indecomposables) for O in self.orthants]
        nfacets = [len(O.facets) for O in self.orthants]
        maxnorm = [max(N for N, _ in O.indecomposables) for O in self.orthants]
        face_data = [self.facet_rows(O) for O in self.orthants]
        maxdist = [max(d for _, d, _ in rows) for rows in face_data]
        maxvol = [max(v for _, _, v in rows) for rows in face_data]
        ind_str = ['[' + ', '.join(K.elt_str(x) for _, x in O.indecomposables) + ']'
                   for O in self.orthants]
        face_str = ['[' + ', '.join('[[' + ', '.join(sage_vector_str(v) for v in verts) + '], %d, %d]'
                                    % (d, vol) for verts, d, vol in rows) + ']'
                    for rows in face_data]
        fields = [
            self.label,
            list_to_string(K.poly_coeffs()),
            list_to_string('(' + ', '.join(str(t) for t in s) + ')' for s in sigs),
            list_to_string(K.elt_str(u) for u in K.tpfu),
            list_to_string(counts),
            list_to_string(nfacets),
            list_to_string(maxnorm),
            list_to_string(maxdist),
            list_to_string(maxvol),
            list_to_string(ind_str),
            list_to_string(face_str),
        ]
        return '|'.join(fields)

    # --- machine readable record ---------------------------------------------

    def record(self):
        K = self.K
        orth = []
        for O in self.orthants:
            facets = []
            for F in O.facets.values():
                facets.append({'c': list(F.c), 'd': F.d, 'volume': F.volume,
                               'vertices': [list(v) for v in F.vertices],
                               'lattice_points': len(F.lattice_points)})
            facets.sort(key=lambda f: (f['d'], f['volume'], f['vertices']))
            orth.append({
                'signature': list(K.sail_signatures[O.index]),
                'num_indecomposables': len(O.indecomposables),
                'max_norm': max(N for N, _ in O.indecomposables),
                'indecomposables': [[N, list(x)] for N, x in O.indecomposables],
                'num_facets': len(O.facets),
                'S': sum(F.d * F.volume for F in O.facets.values()),
                'facets': facets,
                'stats': O.stats,
            })
        return {
            'label': self.label,
            'coeffs': K.poly_coeffs(),
            'degree': K.n,
            'disc': K.disc,
            'class_number': K.class_number,
            'regulator': K.regulator,
            'certified': K.certified,
            'integral_basis': [K.elt_str(tuple(int(i == j) for i in range(K.n))) for j in range(K.n)],
            'tpfu': [list(u) for u in K.tpfu],
            'unit_signature_rank': K.n - len(K.sail_signatures).bit_length() + 1,
            'seed_norm_bound': self.seed_bound,
            'orthants': orth,
            'timings': self.timings,
        }


def compute_field(coeffs, label=None, tp_only=False, seed_bound=None, certify=True,
                  lll_units=True):
    """Sails and indecomposables of every sail signature class of one field."""
    t0 = time.time()
    K = NumberFieldData(coeffs, label=label, certify=certify, lll_units=lll_units)
    t1 = time.time()
    classes = [0] if tp_only else list(range(len(K.sail_signatures)))
    seeds, B = small_norm_seeds(K, classes, seed_bound)
    t2 = time.time()
    orthants, t_sail, t_ind = [], 0.0, 0.0
    for j in classes:
        O = Orthant(K, j)
        a = time.time()
        O.compute_sail(seeds[j])
        b = time.time()
        compute_indecomposables(O)
        c = time.time()
        t_sail += b - a
        t_ind += c - b
        orthants.append(O)
    timings = {'field': round(t1 - t0, 4), 'seeds': round(t2 - t1, 4),
               'sail': round(t_sail, 4), 'indecomposables': round(t_ind, 4),
               'total': round(time.time() - t0, 4)}
    return FieldResult(K, orthants, B, timings)
