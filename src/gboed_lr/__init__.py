"""Design-time learning-rate calibration and online gating for GBOED."""

from .design_time import (
    DesignTimeCalibration,
    QuadratureConstants,
    certify_homoscedasticity,
    certify_scale_invariance,
    design_time_lhw_imq,
    imq_wsm_quadrature_constants,
    sup_psi_B_exact,
)
from .rule import (
    RuleDecision,
    annealed_conjugate_predc_posterior,
    annealed_predc_scales,
    compute_delta0_sq,
    compute_s_pred,
    conjugate_predc_imq_wsm,
    phi_conflict,
    tau_min_floor,
    unified_omega,
)

__all__ = [
    "DesignTimeCalibration",
    "QuadratureConstants",
    "RuleDecision",
    "annealed_conjugate_predc_posterior",
    "annealed_predc_scales",
    "certify_homoscedasticity",
    "certify_scale_invariance",
    "compute_delta0_sq",
    "compute_s_pred",
    "conjugate_predc_imq_wsm",
    "design_time_lhw_imq",
    "imq_wsm_quadrature_constants",
    "phi_conflict",
    "sup_psi_B_exact",
    "tau_min_floor",
    "unified_omega",
]
