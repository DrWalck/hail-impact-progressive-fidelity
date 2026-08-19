# Model definitions

## Model 1 — Baseline Translational Representation
Gravity-only translational baseline.

## Model 2 — Aerodynamic Fidelity
Adds experimentally informed natural-hail drag in a stationary atmosphere with constant reference properties.

## Model 3 — Environmental Fidelity
Adds altitude-dependent atmospheric properties, finite release height, and ambient wind while retaining a translating-particle representation.

## Model 4 — Reduced-Order Rigid-Body Fidelity
Adds a prescribed dimensionless rotational correction (`beta`) without explicitly solving the rotational equations of motion.

## Model 5 — Euler Rigid-Body Fidelity
Introduces nonspherical rigid-body mechanics, aerodynamic lift/moment, orientation, angular velocity, and contact-point kinematics. Full nominal propagation is evidence-limited by the available orientation-dependent aerodynamic coefficient domain.
