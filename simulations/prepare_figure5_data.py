"""
Data prep for Figure 5: extends the existing roof-pitch sweep with a
physically-derived dimensionless rotation parameter

    beta = omega * R / v_G

computed from the rigid-body model's actual tangential impulse at each
roof pitch (via angular-momentum change of a solid sphere), rather than
reusing the fixed beta_eff INPUT assumption (which by construction does
not vary with roof pitch). This makes beta a genuine simulation output
that can legitimately differ across the three roof pitches.
"""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
import hail_model_rb_particle_selection_v7 as hm

OUTDIR = "outputs_hail_model_selection"

# Base geometry/kinematics from Model 1 (particle) rows - same source used
# by make_roof_pitch_sweep_outputs in the main script.
full = hm.run_particle_model_suite()
base = full.loc[full["model_name"] == "model1_particle"].copy()

rows = []
for pitch in hm.ROOF_PITCH_SWEEP_DEG:
    for _, row in base.iterrows():
        m = row["mass_kg"]
        r_m = row["r_m"] if "r_m" in row else (row["diam_in"] * 2.54 / 100.0) / 2.0
        beta_eff = float(row["beta_eff"])
        v_rb = np.array([hm.WIND_SPEED_MPS, 0.0, -row["v_terminal_mps"] * (1.0 + beta_eff)])
        v_G_mag = float(np.linalg.norm(v_rb))

        resp4 = hm.simulate_contact_response(
            mass=m, v_in_vec=v_rb, roof_pitch_deg=pitch,
            k=hm.CONTACT_K, c=hm.CONTACT_C, mu_friction=hm.MU_FRICTION,
        )

        I_t = resp4["I_tangential_Ns"]
        # Solid sphere: I_sphere = (2/5) m R^2; angular impulse = I_t * R (torque arm = R)
        # => omega = I_t * R / I_sphere = (5/2) I_t / (m R)
        omega = (5.0 / 2.0) * I_t / (m * r_m) if (m > 0 and r_m > 0) else 0.0
        beta_derived = omega * r_m / v_G_mag if v_G_mag > 0 else 0.0

        rows.append({
            "diam_in": row["diam_in"],
            "roof_pitch_deg": pitch,
            "model4_peak_normal_force_N": resp4["peak_normal_force_N"],
            "I_tangential_Ns": I_t,
            "omega_rad_s": omega,
            "v_G_mps": v_G_mag,
            "beta": beta_derived,
        })

df = pd.DataFrame(rows).sort_values(["roof_pitch_deg", "diam_in"])

# Merge in the already-computed force_error_pct from the existing sweep CSV
sweep = pd.read_csv(f"{OUTDIR}/roof_pitch_sweep_model1_vs_model4.csv")
df = df.merge(
    sweep[["diam_in", "roof_pitch_deg", "force_error_pct"]],
    on=["diam_in", "roof_pitch_deg"], how="left",
)

out_csv = f"{OUTDIR}/fig5_pitch_beta_data.csv"
df.to_csv(out_csv, index=False)
print(f"Saved: {out_csv}")
print(df.groupby("roof_pitch_deg")[["force_error_pct", "beta"]].agg(["min", "max", "mean"]))
