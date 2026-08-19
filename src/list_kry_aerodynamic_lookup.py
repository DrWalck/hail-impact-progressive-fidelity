from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LookupResult:
    value: float
    status: str
    details: str
    source: str = "List et al. (1973), working plot digitization"


class List73Aerodynamics:
    """Conservative lookup/interpolation wrapper for List et al. (1973).

    The working datasets are plot-digitized from Figs. 2-8 of List et al.
    They are not exact tabulated measurements. This class intentionally
    refuses unsupported 2-D/3-D interpolation or silent extrapolation.
    """

    def __init__(self, data_dir: str | Path = "/mnt/data") -> None:
        data_dir = Path(data_dir)
        self.cd = pd.read_csv(data_dir / "list73_aero_digitized" / "list73_CD_digitized_working.csv")
        self.cl = pd.read_csv(data_dir / "list73_aero_digitized" / "list73_CL_digitized_working.csv")
        self.cm7 = pd.read_csv(data_dir / "list73_fig7_CM_digitized_working.csv")
        self.cm8 = pd.read_csv(data_dir / "list73_fig8_CM_digitized_working.csv")

    @staticmethod
    def _interp_1d(x: float, xs: np.ndarray, ys: np.ndarray, allow_extrapolation: bool = False) -> LookupResult:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        xmin, xmax = float(xs.min()), float(xs.max())
        if x < xmin or x > xmax:
            if not allow_extrapolation:
                return LookupResult(np.nan, "extrapolation_required", f"Requested x={x:g} lies outside [{xmin:g}, {xmax:g}].")
            value = float(np.interp(np.clip(x, xmin, xmax), xs, ys))
            return LookupResult(value, "extrapolated", f"Requested x={x:g} outside [{xmin:g}, {xmax:g}]; endpoint value used as conservative placeholder.")
        value = float(np.interp(x, xs, ys))
        exact = np.any(np.isclose(xs, x))
        status = "direct" if exact else "interpolated_1d"
        return LookupResult(value, status, f"1-D interpolation in x over [{xmin:g}, {xmax:g}].")

    @staticmethod
    def _supported_bracket(value: float, values: np.ndarray) -> bool:
        vals = np.sort(np.unique(np.asarray(values, dtype=float)))
        return vals.min() <= value <= vals.max()

    def _lookup_cd_at_slice(self, Re: float, theta: float, chi: float) -> Optional[LookupResult]:
        # Exact chi slice with theta variation: only chi=0.50 is fully represented.
        exact_chi = np.isclose(self.cd["axis_ratio"].to_numpy(dtype=float), chi)
        if np.any(exact_chi) and np.isclose(chi, 0.50):
            sub = self.cd[np.isclose(self.cd["axis_ratio"], chi)]
            thetas = np.sort(sub["angle_deg"].unique().astype(float))
            if self._supported_bracket(theta, thetas):
                theta_low = thetas[tetas_idx := np.searchsorted(thetas, theta, side="right") - 1]
                theta_high = thetas[min(tetas_idx + 1, len(thetas) - 1)]
                rows = []
                for th in [theta_low, theta_high] if theta_high != theta_low else [theta_low]:
                    r = sub[np.isclose(sub["angle_deg"], th)]
                    rr = self._interp_1d(Re, r["Re_x1e4"].to_numpy() * 1e4, r["CD"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((th, rr.value, rr.status))
                if theta_high == theta_low:
                    return LookupResult(rows[0][1], "direct_slice" if rows[0][2] == "direct" else "interpolated_1d", "chi=0.50 experimental slice; interpolated in Re.")
                value = float(np.interp(theta, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re within bracketing chi=0.50 angle slices.")

        # Exact angle slices with chi variation: available at 0, 45, 90 degrees.
        available_angles = np.sort(self.cd["angle_deg"].unique().astype(float))
        exact_theta = np.any(np.isclose(available_angles, theta))
        if exact_theta and any(np.isclose(theta, a) for a in [0.0, 45.0, 90.0]):
            sub = self.cd[np.isclose(self.cd["angle_deg"], theta)]
            chis = np.sort(sub["axis_ratio"].unique().astype(float))
            if self._supported_bracket(chi, chis):
                chi_low_idx = np.searchsorted(chis, chi, side="right") - 1
                chi_low = chis[chi_low_idx]
                chi_high = chis[min(chi_low_idx + 1, len(chis) - 1)]
                rows = []
                for ch in [chi_low, chi_high] if chi_high != chi_low else [chi_low]:
                    r = sub[np.isclose(sub["axis_ratio"], ch)]
                    rr = self._interp_1d(Re, r["Re_x1e4"].to_numpy() * 1e4, r["CD"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((ch, rr.value, rr.status))
                if chi_high == chi_low:
                    return LookupResult(rows[0][1], "direct_slice" if rows[0][2] == "direct" else "interpolated_1d", f"theta={theta:g} degree experimental slice; interpolated in Re.")
                value = float(np.interp(chi, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re within bracketing theta slice data.")

        return None

    def get_cd(self, Re: float, theta: float, chi: float) -> LookupResult:
        result = self._lookup_cd_at_slice(Re, theta, chi)
        if result is not None:
            return result
        return LookupResult(np.nan, "unsupported_parameter_combination", "No published/digitized slice brackets this (Re, theta, chi) combination without unsupported 2-D/3-D extrapolation.")

    def _lookup_cl(self, Re: float, theta: float, chi: float) -> Optional[LookupResult]:
        # Fig. 5: chi=0.50, theta slices. Fig. 6: theta=45, chi slices.
        if np.isclose(chi, 0.50):
            sub = self.cl[np.isclose(self.cl["axis_ratio"], 0.50)]
            thetas = np.sort(sub["angle_deg"].unique().astype(float))
            if self._supported_bracket(theta, thetas):
                rows = []
                theta_low_idx = np.searchsorted(thetas, theta, side="right") - 1
                theta_low = thetas[theta_low_idx]
                theta_high = thetas[min(theta_low_idx + 1, len(thetas) - 1)]
                for th in [theta_low, theta_high] if theta_high != theta_low else [theta_low]:
                    r = sub[np.isclose(sub["angle_deg"], th)]
                    rr = self._interp_1d(Re, r["Re_x1e4"].to_numpy() * 1e4, r["CL"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((th, rr.value))
                if theta_high == theta_low:
                    return LookupResult(rows[0][1], "direct_slice", "chi=0.50 lift slice; interpolated in Re.")
                value = float(np.interp(theta, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re and theta within chi=0.50 measurements.")
        if np.isclose(theta, 45.0):
            sub = self.cl[np.isclose(self.cl["angle_deg"], 45.0)]
            chis = np.sort(sub["axis_ratio"].unique().astype(float))
            if self._supported_bracket(chi, chis):
                chi_low_idx = np.searchsorted(chis, chi, side="right") - 1
                chi_low = chis[chi_low_idx]
                chi_high = chis[min(chi_low_idx + 1, len(chis) - 1)]
                rows = []
                for ch in [chi_low, chi_high] if chi_high != chi_low else [chi_low]:
                    r = sub[np.isclose(sub["axis_ratio"], ch)]
                    rr = self._interp_1d(Re, r["Re_x1e4"].to_numpy() * 1e4, r["CL"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((ch, rr.value))
                if chi_high == chi_low:
                    return LookupResult(rows[0][1], "direct_slice", "theta=45 degree lift slice; interpolated in Re.")
                value = float(np.interp(chi, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re and aspect ratio within theta=45 measurements.")
        return None

    def get_cl(self, Re: float, theta: float, chi: float) -> LookupResult:
        result = self._lookup_cl(Re, theta, chi)
        if result is not None:
            return result
        return LookupResult(np.nan, "unsupported_parameter_combination", "No published/digitized slice brackets this (Re, theta, chi) combination without unsupported 2-D/3-D extrapolation.")

    def _lookup_cm(self, Re: float, theta: float, chi: float) -> Optional[LookupResult]:
        if np.isclose(chi, 0.50):
            sub = self.cm7
            thetas = np.sort(sub["theta_deg"].unique().astype(float))
            if self._supported_bracket(theta, thetas):
                theta_low_idx = np.searchsorted(thetas, theta, side="right") - 1
                theta_low = thetas[theta_low_idx]
                theta_high = thetas[min(theta_low_idx + 1, len(thetas) - 1)]
                rows = []
                for th in [theta_low, theta_high] if theta_high != theta_low else [theta_low]:
                    r = sub[np.isclose(sub["theta_deg"], th)]
                    rr = self._interp_1d(Re, r["Re"].to_numpy(), r["CM"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((th, rr.value))
                if theta_high == theta_low:
                    return LookupResult(rows[0][1], "direct_slice", "chi=0.50 moment slice; interpolated in Re.")
                value = float(np.interp(theta, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re and theta within chi=0.50 measurements.")
        if np.isclose(theta, 45.0):
            sub = self.cm8
            chis = np.sort(sub["aspect_ratio"].unique().astype(float))
            if self._supported_bracket(chi, chis):
                chi_low_idx = np.searchsorted(chis, chi, side="right") - 1
                chi_low = chis[chi_low_idx]
                chi_high = chis[min(chi_low_idx + 1, len(chis) - 1)]
                rows = []
                for ch in [chi_low, chi_high] if chi_high != chi_low else [chi_low]:
                    r = sub[np.isclose(sub["aspect_ratio"], ch)]
                    rr = self._interp_1d(Re, r["Re"].to_numpy(), r["CM"].to_numpy())
                    if np.isnan(rr.value):
                        return rr
                    rows.append((ch, rr.value))
                if chi_high == chi_low:
                    return LookupResult(rows[0][1], "direct_slice", "theta=45 degree moment slice; interpolated in Re.")
                value = float(np.interp(chi, [rows[0][0], rows[1][0]], [rows[0][1], rows[1][1]]))
                return LookupResult(value, "interpolated_2d_within_measured_slices", "Interpolated in Re and aspect ratio within theta=45 measurements.")
        return None

    def get_cm(self, Re: float, theta: float, chi: float) -> LookupResult:
        result = self._lookup_cm(Re, theta, chi)
        if result is not None:
            return result
        return LookupResult(np.nan, "unsupported_parameter_combination", "No published/digitized slice brackets this (Re, theta, chi) combination without unsupported 2-D/3-D extrapolation.")


if __name__ == "__main__":
    aero = List73Aerodynamics()
    tests = [
        (1.0e5, 45.0, 0.50),
        (1.0e5, 37.5, 0.50),
        (1.0e5, 45.0, 0.65),
        (5.0e5, 45.0, 0.50),
        (1.0e5, 37.5, 0.65),
    ]
    for Re, theta, chi in tests:
        print({
            "Re": Re,
            "theta": theta,
            "chi": chi,
            "CD": aero.get_cd(Re, theta, chi),
            "CL": aero.get_cl(Re, theta, chi),
            "CM": aero.get_cm(Re, theta, chi),
        })
