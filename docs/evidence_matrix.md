# Evidence-aware implementation

The manuscript treats evidence status as local to each model component.

## Aerodynamic drag
Natural-hail drag uses experimentally informed parameterizations with explicit Reynolds-number / geometry boundaries. Extrapolative states remain labeled rather than silently promoted to supported states.

## Euler lift and moment
The usable Euler domain is the intersection of the available `C_L` and `C_M` evidence over Reynolds number, relative-flow angle, and aspect ratio. See `data/evidence_domains/`.

## Contact model
Peak normal contact force uses the Sun et al. analytical maximum-force formulation. Direct experimental calibration, broader published analytical application, and beyond-application conditions are treated separately.

## Friction
No measured dynamic hailstone–asphalt-shingle friction coefficient is asserted. The archived friction results are conditional Coulomb sliding limits over a prescribed sensitivity range.
