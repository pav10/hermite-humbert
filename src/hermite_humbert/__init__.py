"""Hermite-Humbert constants gamma_{K,2} of real quadratic fields K = Q(sqrt D).

    >>> from hermite_humbert import compute
    >>> r = compute(13, verbose=False)
    >>> r["gamma"], r["classes"][0]["closed_form"][1]

See README.md for the method and what is (and is not) rigorous.
"""
from .core import compute, gamma_of_form, chart_point
from .field import RealQuadraticField

__version__ = "0.1.0"
__all__ = ["compute", "gamma_of_form", "chart_point", "RealQuadraticField", "__version__"]
