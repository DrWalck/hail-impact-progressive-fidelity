"""
Paper 1 - Production Model 4 scaffold
Progressive hail-impact model with literature-grounded projectile dynamics.

Purpose
-------
This module is intentionally conservative. It does not invent aerodynamic data
where List et al. (1973) do not provide a supported interpolation path. It also
keeps the Sun et al. (2015) contact law separate from roofing response.

Model 4 architecture
--------------------
1) Finite-height trajectory through altitude-varying atmosphere
2) Literature-based drag, lift, and aerodynamic moment (List et al. 1973)
3) Coupled translation and Euler rotation
4) Contact-point velocity at impact
5) Sun et al. (2015) nonlinear viscoelastic hail-contact law
6) Friction enters only after the normal contact law is solved

Important scope boundaries
--------------------------
- Unmelted hailstone is assumed.
- No hail growth/ablation/melting is modeled.
- No roof structural deformation is modeled here.
- List aerodynamic coefficients are working plot digitizations, not exact
  tabulated measurements; unsupported combinations are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, pi, sin, sqrt
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.integrate import solve_ivp

from list73_aerodynamic_lookup import List73Aerodynamics, LookupResult


G = 9.80665
R_AIR = 287.05287
P0 = 101325.0
T0 = 288.15
LAPSE = -0.0065
MU_AIR = 1.7894e-5  # Pa s; production sensitivity can replace this later


@dataclass(frozen=True)
class HailState:
    z_m: float
    x_m: float
    vx_mps: float
    vz_mps: float
    gamma_rad: float
    omega_rads: float


@dataclass(frozen=True)
class AtmosphereState:
    T_K: float
    p_Pa: float
    rho_kg_m3: float


@dataclass(frozen=True)
class WindProfile:
    kind: str
    speed_mps: float = 0.0
    top_speed_mps: float = 0.0
    top_height_m: float = 10000.0

    def velocity(self, z_m: float) -> np.ndarray:
        z = max(0.0, float(z_m))
        if self.kind == "calm":
            return np.array([0.0, 0.0])
        if self.kind == "uniform":
            return np.array([self.speed_mps, 0.0])
        if self.kind == "linear_shear":
            frac = min(z / self.top_height_m, 1.0)
            u = self.speed_mps + frac * (self.top_speed_mps - self.speed_mps)
            return np.array([u, 0.0])
        raise ValueError(f"Unknown wind profile: {self.kind}")


@dataclass(frozen=True)
class HailGeometry:
    diameter_m: float
    aspect_ratio: float
    density_kg_m3: float

    @property
    def volume_m3(self) -> float:
        # Oblate spheroid: major diameter D, minor diameter = chi*D.
        return (pi / 6.0) * self.aspect_ratio * self.diameter_m ** 3

    @property
    def mass_kg(self) -> float:
        return self.density_kg_m3 * self.volume_m3

    @property
    def inertia_major_kgm2(self) -> float:
        # List et al. (1973): I = m D^2 (1 + e^2) / 20
        e = self.aspect_ratio
        return self.mass_kg * self.diameter_m ** 2 * (1.0 + e ** 2) / 20.0


@dataclass(frozen=True)
class ContactParameters:
    # Sun et al. (2015) correlations are based on impact velocity in m/s.
    # The valid calibration range is 7.9-29.1 m/s for 50-mm spherical SHI.
    mu: float = 0.20

    def from_impact_speed(self, v_imp_mps: float) -> tuple[float, float, float, str]:
        v = float(v_imp_mps)
        kn = 2.200 * v + 170.269  # kN/m^p, Sun et al. Fig. 18
        p = 0.010 * v + 1.263
        cor = -0.001 * v + 0.049
        status = "within_sun2015_velocity_range" if 7.9 <= v <= 29.1 else "extrapolated_beyond_sun2015_velocity_range"
        return kn * 1e3, p, max(cor, 1e-4), status

    def damping_from_sun2015(self, kn_N_m_p: float, p: float, cor: float, delta_dot0_mps: float) -> float:
        if delta_dot0_mps <= 0.0:
            raise ValueError("Initial indentation speed must be positive for Sun et al. damping relation.")
        return (0.2 * p + 1.3) * ((1.0 - cor) / cor) * kn_N_m_p / delta_dot0_mps


class Model4:
    """Production scaffold for the literature-grounded Model 4."""

    def __init__(self, aero: Optional[List73Aerodynamics] = None) -> None:
        self.aero = aero or List73Aerodynamics()

    @staticmethod
    def atmosphere(z_m: float) -> AtmosphereState:
        """U.S. Standard Atmosphere, troposphere 0-11 km."""
        z = min(max(float(z_m), 0.0), 11000.0)
        T = T0 + LAPSE * z
        p = P0 * (T / T0) ** (-G / (LAPSE * R_AIR))
        rho = p / (R_AIR * T)
        return AtmosphereState(T, p, rho)

    @staticmethod
    def relative_angle_deg(vx_rel: float, vz_rel: float, gamma_rad: float) -> float:
        # For List et al.'s convention, beta is atan(-u'/w').
        beta = np.degrees(np.arctan2(-vx_rel, -vz_rel))
        theta = np.degrees(gamma_rad) + beta
        return float(theta)

    @staticmethod
    def signed_restoring_cm(cm_magnitude: float, theta_deg: float) -> float:
        if abs(theta_deg) < 1e-12:
            return 0.0
        return -np.sign(theta_deg) * abs(cm_magnitude)

    def aerodynamic_coefficients(
        self,
        Re: float,
        theta_deg: float,
        aspect_ratio: float,
        *,
        allow_extrapolation: bool = False,
    ) -> tuple[LookupResult, LookupResult, LookupResult]:
        cd = self.aero.get_cd(Re, theta_deg, aspect_ratio)
        cl = self.aero.get_cl(Re, theta_deg, aspect_ratio)
        cm = self.aero.get_cm(Re, abs(theta_deg), aspect_ratio)
        return cd, cl, cm

    def trajectory_rhs(
        self,
        t: float,
        y: np.ndarray,
        hail: HailGeometry,
        wind: WindProfile,
        *,
        small_angle_scale: float = 1.0,
        strict_aero: bool = True,
    ) -> np.ndarray:
        x, z, vx, vz, gamma, omega = y
        atm = self.atmosphere(z)
        wind_v = wind.velocity(z)

        vrel_x = vx - wind_v[0]
        vrel_z = vz - wind_v[1]
        vrel = np.hypot(vrel_x, vrel_z)
        if vrel < 1e-12:
            return np.zeros_like(y)

        # Reynolds number on major-axis diameter.
        Re = atm.rho_kg_m3 * vrel * hail.diameter_m / MU_AIR
        theta_deg = self.relative_angle_deg(vrel_x, vrel_z, gamma)

        # The production solver must remain in the experimentally represented
        # angle range. Small-angle reconstruction is explicit and separate.
        abs_theta = abs(theta_deg)
        if hail.aspect_ratio == 0.50 and abs_theta < 15.0:
            base_angle = 15.0
            cd = self.aero.get_cd(Re, base_angle, hail.aspect_ratio)
            cl = self.aero.get_cl(Re, base_angle, hail.aspect_ratio)
            cm = self.aero.get_cm(Re, base_angle, hail.aspect_ratio)
            if any(np.isnan(r.value) for r in (cd, cl, cm)):
                raise ValueError("List coefficient lookup unavailable for small-angle reconstruction.")
            scale = min(abs_theta / base_angle, 1.0)
            cm_mag = abs(cm.value) * scale * small_angle_scale
            cm_val = self.signed_restoring_cm(cm_mag, theta_deg)
            cd_val = cd.value
            cl_val = np.sign(theta_deg) * cl.value * scale
            status = (cd.status, cl.status, cm.status, "small_angle_reconstruction")
        else:
            cd_r, cl_r, cm_r = self.aerodynamic_coefficients(Re, theta_deg, hail.aspect_ratio)
            if strict_aero and any(np.isnan(r.value) for r in (cd_r, cl_r, cm_r)):
                raise ValueError(
                    f"Unsupported List aerodynamic state: Re={Re:.3g}, theta={theta_deg:.3f}, chi={hail.aspect_ratio:.3f}; "
                    f"statuses={cd_r.status}/{cl_r.status}/{cm_r.status}"
                )
            cd_val = cd_r.value
            cl_val = cl_r.value
            cm_val = self.signed_restoring_cm(cm_r.value, theta_deg)

        q = 0.5 * atm.rho_kg_m3 * vrel ** 2
        A_ref = pi * (hail.diameter_m ** 2) / 4.0
        Fd = q * cd_val * A_ref
        Fl = q * cl_val * A_ref
        # List et al. moment coefficient definition:
        # CM = GR / [(1/8) rho v^2 pi D^3]
        M_aero = cm_val * (0.125 * atm.rho_kg_m3 * vrel ** 2 * pi * hail.diameter_m ** 3)

        # Resolve lift perpendicular to relative flow; drag opposes relative flow.
        er_x = vrel_x / vrel
        er_z = vrel_z / vrel
        # clockwise 2-D normal to relative flow
        en_x = -er_z
        en_z = er_x

        Fx = -Fd * er_x + Fl * en_x
        Fz = -Fd * er_z + Fl * en_z - hail.mass_kg * G

        ax = Fx / hail.mass_kg
        az = Fz / hail.mass_kg

        # Rotational damping is intentionally parameterized, not guessed here.
        # Production calibration will provide this from List's K or a directly
        # defensible dimensional equivalent.
        angular_damping = 0.0
        alpha = (M_aero - angular_damping * omega) / hail.inertia_major_kgm2

        return np.array([vx, vz, ax, az, omega, alpha], dtype=float)

    def integrate_descent(
        self,
        hail: HailGeometry,
        release_height_m: float,
        wind: WindProfile,
        *,
        gamma0_deg: float = 5.0,
        omega0_rads: float = 0.0,
        small_angle_scale: float = 1.0,
        max_time_s: float = 300.0,
        strict_aero: bool = True,
    ):
        y0 = np.array([0.0, release_height_m, 0.0, 0.0, np.deg2rad(gamma0_deg), omega0_rads], dtype=float)

        def rhs(t, y):
            return self.trajectory_rhs(t, y, hail, wind, small_angle_scale=small_angle_scale, strict_aero=strict_aero)

        def impact_event(t, y):
            return y[1]

        impact_event.terminal = True
        impact_event.direction = -1

        return solve_ivp(
            rhs,
            (0.0, max_time_s),
            y0,
            events=impact_event,
            rtol=1e-7,
            atol=1e-9,
            max_step=0.25,
        )


if __name__ == "__main__":
    print("Model 4 production scaffold loaded.")
    print("No final production run is performed until List aerodynamic coverage, density, wind, and damping choices are finalized.")
