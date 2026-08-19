"""Natural-hail drag model based on Theis et al. (2026), JAS.

Source-derived equations:
  Psi(Dmax_cm) = 1 - 0.025 Dmax_cm^1.24
  CD0(Psi) = 0.52 / Psi  (Newton-regime power law)
  CD,gr = 0.58 * (1 + 6.27/sqrt(Re))^2
  h = exp(-(Re/6000)^m(Psi))
  m(Psi) = 2.15 - 1.19 Psi
  CD = CD,gr*h + CD0*(1-h)

The transition parameterization is reported as valid for approximately
10 <= Re <= 1e5 and 0.67 < Psi <= 1 for the fitted hailstone range.
"""
from __future__ import annotations
import math


def sphericity_from_dmax_cm(dmax_cm: float) -> float:
    return 1.0 - 0.025 * dmax_cm ** 1.24


def drag_coefficient(Re: float, psi: float) -> float:
    Re = max(float(Re), 1e-12)
    psi = float(psi)
    cd0 = 0.52 / psi
    mpsi = 2.15 - 1.19 * psi
    cdgr = 0.58 * (1.0 + 6.27 / math.sqrt(Re)) ** 2
    h = math.exp(-((Re / 6000.0) ** mpsi))
    return cdgr * h + cd0 * (1.0 - h)


def range_status(psi: float, Re: float) -> str:
    if psi < 0.67:
        return "extrapolative_shape"
    if Re < 10 or Re > 1e5:
        return "outside_reported_Re_range"
    return "within_theis_parameterization"
