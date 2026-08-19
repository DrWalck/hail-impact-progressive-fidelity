from __future__ import annotations

from dataclasses import dataclass
from math import acos, atan2, cos, exp, pi, radians, sin, sqrt, degrees
from pathlib import Path
import csv
import numpy as np
from scipy.integrate import solve_ivp

from list73_aerodynamic_lookup import List73Aerodynamics

G = 9.80665
R_AIR = 287.05287
P0 = 101325.0
T0 = 288.15
LAPSE = -0.0065
MU_REF = 1.716e-5
T_REF = 273.15
SUTHERLAND = 111.0
RHO_HAIL = 790.0
NOMINAL_HEIGHT_M = 1000.0
WIND_MPS = 33.3
CHI_EULER_PRIMARY = 0.50
K_PRIMARY = 0.0  # Damping conversion from List dimensionless K to SI is not yet fully verified.
BETA_BASE_BY_SIZE = {1.00: 0.008, 1.25: 0.008, 1.50: 0.008, 1.75: 0.008,
                     2.00: 0.010, 2.50: 0.010, 3.00: 0.012, 3.25: 0.012}

D_IN = [1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.25]

@dataclass(frozen=True)
class Atmosphere:
    rho: float
    mu: float


def atmosphere(z):
    z = float(np.clip(z, 0.0, 11000.0))
    T = T0 + LAPSE * z
    p = P0 * (T / T0) ** (-G / (LAPSE * R_AIR))
    rho = p / (R_AIR * T)
    mu = MU_REF * (T / T_REF) ** 1.5 * (T_REF + SUTHERLAND) / (T + SUTHERLAND)
    return Atmosphere(rho, mu)


def wind(z):
    return np.array([WIND_MPS, 0.0])


def theis_psi(D_cm):
    return 1.0 - 0.025 * D_cm ** 1.24


def theis_cd(Re, D_cm):
    Re = max(float(Re), 1e-12)
    psi = theis_psi(D_cm)
    CD0_gr = 0.58
    delta0 = 6.27
    a = 6000.0
    b = 2.15
    c = -1.19
    CD0 = 0.52 / psi
    mpsi = b + c * psi
    CDgr = CD0_gr * (1.0 + (delta0 / sqrt(Re)) ** 2)
    h = exp(-((Re / a) ** mpsi))
    return CDgr * h + CD0 * (1.0 - h), (1e1 <= Re <= 1e5 and psi > 0.67)


def oblate_volume(D_m, chi):
    return pi / 6.0 * chi * D_m ** 3


def mass_and_inertia(D_m, chi):
    m = RHO_HAIL * oblate_volume(D_m, chi)
    I = m * D_m ** 2 * (1.0 + chi ** 2) / 20.0
    return m, I


def beta_eff(D_in):
    # Reproduce original reduced-order implementation from the legacy code.
    base = BETA_BASE_BY_SIZE[float(D_in)]
    size_factor = max(0.0, min(1.0, (D_in - 1.0) / (3.25 - 1.0)))
    return base * (0.75 + 0.5 * size_factor)


def run_model3(D_in):
    D = D_in * 0.0254
    # Model 3 retains spherical particle representation from canonical Models 1-3 driver.
    m = RHO_HAIL * pi / 6.0 * D ** 3
    area = pi * D ** 2 / 4.0
    def rhs(t, y):
        x, z, vx, vz = y
        atm = atmosphere(z)
        vrel = np.array([vx - WIND_MPS, vz])
        s = np.linalg.norm(vrel)
        if s < 1e-12:
            ax, az = 0.0, -G
        else:
            Re = atm.rho * s * D / atm.mu
            cd, _ = theis_cd(Re, D * 100.0)
            Fd = 0.5 * atm.rho * s**2 * cd * area
            ax = -Fd * vrel[0] / s / m
            az = -Fd * vrel[1] / s / m - G
        return [vx, vz, ax, az]
    def ev(t,y): return y[1]
    ev.terminal = True; ev.direction = -1
    sol = solve_ivp(rhs, (0, 300), [0, NOMINAL_HEIGHT_M, 0, 0], events=ev, rtol=1e-9, atol=1e-11, max_step=0.05)
    y = sol.y_events[0][0]
    vx, vz = y[2], y[3]
    v = float(np.hypot(vx, vz))
    return dict(model='Model 3', diameter_in=D_in, Dmax_m=D, mass_kg=m, vx_mps=vx, vz_mps=vz, V_impact_mps=v,
                gamma_deg=0.0, omega_rads=0.0, beta=np.nan, vt_mps=np.nan, vn_mps=np.nan,
                aero_status='sensitivity')


def list_coeff(aero, Re, theta_deg):
    # Primary Euler branch: chi=0.50 List/Kry data.
    chi = CHI_EULER_PRIMARY
    a = abs(float(theta_deg))
    # map to acute equivalent used by the digitized curves
    a = min(a, 180.0 - a) if a > 90.0 else a
    a = max(0.0, min(90.0, a))
    if a < 15.0:
        # Explicit small-angle reconstruction from 15-degree slice.
        cd = aero.get_cd(Re, 15.0, chi)
        cl = aero.get_cl(Re, 15.0, chi)
        cm = aero.get_cm(Re, 15.0, chi)
        f = a / 15.0
        return cd.value, np.sign(theta_deg) * cl.value * f, -np.sign(theta_deg) * abs(cm.value) * f, (cd.status, cl.status, cm.status, 'small_angle_reconstruction')
    cd = aero.get_cd(Re, a, chi)
    cl = aero.get_cl(Re, a, chi)
    cm = aero.get_cm(Re, a, chi)
    if any(np.isnan(x.value) for x in (cd, cl, cm)):
        return np.nan, np.nan, np.nan, (cd.status, cl.status, cm.status)
    return cd.value, np.sign(theta_deg) * cl.value, -np.sign(theta_deg) * abs(cm.value), (cd.status, cl.status, cm.status)


def run_model45(D_in, aero):
    D = D_in * 0.0254
    beta = beta_eff(D_in)
    m5, I5 = mass_and_inertia(D, CHI_EULER_PRIMARY)

    # Model 4: start from Model 3 translational state and apply legacy reduced-order beta correction.
    m3row = run_model3(D_in)
    vx3, vz3 = m3row['vx_mps'], m3row['vz_mps']
    V3 = m3row['V_impact_mps']
    v4x, v4z = vx3, vz3 * (1.0 + beta)
    v4 = float(np.hypot(v4x, v4z))
    # Use beta as prescribed rotational diagnostic; no dynamic omega.
    omega4 = beta * v4 / (D / 2.0)

    # Model 5: Euler with chi=0.50 List/Kry branch. K damping is held at zero in this audit run.
    gamma0 = radians(-15.0)
    # State x,z,vx,vz,gamma,omega
    y0 = np.array([0.0, NOMINAL_HEIGHT_M, 0.0, 0.0, gamma0, 0.0], dtype=float)
    def rhs(t, y):
        x,z,vx,vz,gamma,omega = y
        atm = atmosphere(z)
        vrel_x = vx - WIND_MPS
        vrel_z = vz
        speed = np.hypot(vrel_x, vrel_z)
        if speed < 1e-10:
            return np.array([vx, vz, 0, -G, omega, 0.0])
        Re = atm.rho * speed * D / atm.mu
        beta_flow = np.degrees(np.arctan2(-vrel_x, -vrel_z))
        theta = beta_flow + np.degrees(gamma)
        cd, cl, cm, _ = list_coeff(aero, Re, theta)
        if not np.isfinite(cd):
            # Fail fast: unsupported CL/CM state must not be silently extrapolated.
            raise RuntimeError(f'Unsupported List state D={D_in}in Re={Re:.3g} theta={theta:.3f} deg')
        q = 0.5 * atm.rho * speed**2
        A = pi * D**2 / 4.0
        Fd = q * cd * A
        Fl = q * cl * A
        erx, erz = vrel_x/speed, vrel_z/speed
        enx, enz = -erz, erx
        Fx = -Fd*erx + Fl*enx
        Fz = -Fd*erz + Fl*enz - m5*G
        M = cm * (0.125 * atm.rho * speed**2 * pi * D**3)
        # K is currently kept zero because the SI mapping from List nondimensional K has not been independently verified.
        alpha = M / I5
        return np.array([vx, vz, Fx/m5, Fz/m5, omega, alpha])
    def ev(t,y): return y[1]
    ev.terminal=True; ev.direction=-1
    try:
        sol = solve_ivp(rhs, (0, 300), y0, events=ev, rtol=1e-8, atol=1e-10, max_step=0.02)
        if sol.y_events[0].size == 0:
            raise RuntimeError('impact event not reached')
        y = sol.y_events[0][0]
        vx5, vz5, gamma5, omega5 = y[2], y[3], y[4], y[5]
        V5 = float(np.hypot(vx5, vz5))
        vrel5 = np.array([vx5-WIND_MPS, vz5])
        v5vec = np.array([vx5, vz5])
        # contact point at bottom-most point of the spheroid in 2D; vector magnitude c = D/2 along body minor axis.
        # This is an explicit geometry approximation for contact kinematics.
        c = D/2.0 * CHI_EULER_PRIMARY
        nbody = np.array([-np.sin(gamma5), -np.cos(gamma5)])
        r = c * nbody
        omega_vec = np.array([0.0, 0.0, omega5])
        r3 = np.array([r[0], 0.0, r[1]])
        vG3 = np.array([vx5, 0.0, vz5])
        vc3 = vG3 + np.cross(omega_vec, r3)
        vc = np.array([vc3[0], vc3[2]])
        # 30-deg roof normal outward in 2D
        phi = radians(30.0)
        n = np.array([-sin(phi), cos(phi)])
        vn_signed = float(np.dot(vc, n))
        vt_vec = vc - vn_signed*n
        vt = float(np.linalg.norm(vt_vec))
        vn = -vn_signed if vn_signed < 0 else 0.0
        beta_e = abs(omega5)* (D/2.0) / max(V5,1e-12)
        aero_status = 'supported' if (0.67 < theis_psi(D*100.0)) else 'extrapolative_shape'
        # List status at terminal point for reporting.
        Re5 = atm.rho * np.linalg.norm(vrel5) * D / atm.mu
        _,_,_, statuses = list_coeff(aero, Re5, np.degrees(np.arctan2(-vrel5[0], -vrel5[1])) + np.degrees(gamma5))
        return [
            dict(model='Model 4', diameter_in=D_in, Dmax_m=D, mass_kg=m3row['mass_kg'], vx_mps=v4x, vz_mps=v4z, V_impact_mps=v4,
                 gamma_deg=0.0, omega_rads=omega4, beta=beta, beta_euler=np.nan, vn_mps=np.nan, vt_mps=np.nan,
                 delta_vt_rot_mps=np.nan, aero_status='supported'),
            dict(model='Model 5', diameter_in=D_in, Dmax_m=D, mass_kg=m5, vx_mps=vx5, vz_mps=vz5, V_impact_mps=V5,
                 gamma_deg=np.degrees(gamma5), omega_rads=omega5, beta=beta, beta_euler=beta_e, vn_mps=vn, vt_mps=vt,
                 delta_vt_rot_mps=np.nan, aero_status=aero_status, list_status='/'.join(statuses), K=K_PRIMARY)
        ]
    except Exception as exc:
        return [
            dict(model='Model 4', diameter_in=D_in, Dmax_m=D, mass_kg=m3row['mass_kg'], vx_mps=v4x, vz_mps=v4z, V_impact_mps=v4,
                 gamma_deg=0.0, omega_rads=omega4, beta=beta, beta_euler=np.nan, vn_mps=np.nan, vt_mps=np.nan,
                 delta_vt_rot_mps=np.nan, aero_status='supported'),
            dict(model='Model 5', diameter_in=D_in, Dmax_m=D, mass_kg=m5, vx_mps=np.nan, vz_mps=np.nan, V_impact_mps=np.nan,
                 gamma_deg=np.nan, omega_rads=np.nan, beta=beta, beta_euler=np.nan, vn_mps=np.nan, vt_mps=np.nan,
                 delta_vt_rot_mps=np.nan, aero_status='not_solved', solver_note=str(exc), K=K_PRIMARY)
        ]


def main():
    aero = List73Aerodynamics()
    rows=[]
    for D in D_IN:
        rows.extend(run_model45(D, aero))
    df = __import__('pandas').DataFrame(rows)
    # Compute rotational differences only where finite.
    for D in D_IN:
        m3 = run_model3(D)
        m4 = df[(df.model=='Model 4') & (df.diameter_in==D)].iloc[0]
        m5 = df[(df.model=='Model 5') & (df.diameter_in==D)].iloc[0]
        if np.isfinite(m5.vt_mps):
            df.loc[(df.model=='Model 5') & (df.diameter_in==D),'delta_vt_rot_mps'] = m5.vt_mps - np.linalg.norm([m3['vx_mps'],m3['vz_mps']]) * 0.0
    out = Path('/mnt/data/paper1_models3_5_production_audit.csv')
    df.to_csv(out,index=False)
    with open('/mnt/data/paper1_models3_5_production_metadata.txt','w') as f:
        f.write('Paper 1 Models 3-5 hybrid production audit run\n')
        f.write('Model 3: canonical environmental model from Models 1-3 driver.\n')
        f.write('Model 4: original reduced-order beta correction reproduced from legacy implementation.\n')
        f.write('Model 5: chi=0.50 List/Kry lift/moment branch + Theis natural-hail drag + Euler rotation.\n')
        f.write('IMPORTANT: K damping conversion to SI is NOT yet independently verified; this audit run therefore uses K=0.0. Do not use Model 5 numbers as final manuscript results until K mapping is verified.\n')
    print(df.to_string(index=False))
    print('\nWrote', out)

if __name__=='__main__':
    main()
