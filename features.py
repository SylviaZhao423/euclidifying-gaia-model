"""
The 14 Gaia DR3 parameters used by the XGBoost star/galaxy classifier.

These are the features the model expects as input, in this exact order.
Derived features (corrected_bp_rp_excess, pm_significance, position_error)
must be computed from the raw Gaia catalogue columns.
"""

FEATURES = [
    "parallax_over_error",
    "astrometric_gof_al",
    "astrometric_chi2_al",
    "astrometric_excess_noise_sig",
    "ipd_gof_harmonic_amplitude",
    "ruwe",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_flux_over_error",
    "phot_rp_mean_flux_over_error",
    "bp_g",                        # phot_bp_mean_mag - phot_g_mean_mag
    "g_rp",                        # phot_g_mean_mag - phot_rp_mean_mag
    "corrected_bp_rp_excess",      # see compute_derived_features()
    "pm_significance",             # see compute_derived_features()
    "position_error",              # see compute_derived_features()
]


def compute_derived_features_sql():
    """SQL expressions for the 3 derived features.

    Use these in your SELECT when querying gaia_dr3.gaia_source directly.
    """
    return {
        "bp_g": "phot_bp_mean_mag - phot_g_mean_mag",
        "g_rp": "phot_g_mean_mag - phot_rp_mean_mag",
        "corrected_bp_rp_excess": """
            phot_bp_rp_excess_factor - (
                CASE
                    WHEN bp_rp < 0.5 THEN 1.154360 + 0.033772*bp_rp + 0.032277*bp_rp*bp_rp
                    WHEN bp_rp < 4.0 THEN 1.162004 + 0.011464*bp_rp + 0.049255*bp_rp*bp_rp
                                           - 0.005879*bp_rp*bp_rp*bp_rp
                    ELSE 1.057572 + 0.140537*bp_rp
                END
            )""",
        "pm_significance": """
            SQRT(
                (pmra*pmra*pmdec_error*pmdec_error
                 - 2.0*pmra_pmdec_corr*pmra_error*pmdec_error*pmra*pmdec
                 + pmdec*pmdec*pmra_error*pmra_error)
                / NULLIF(pmra_error*pmra_error*pmdec_error*pmdec_error
                         *(1.0 - pmra_pmdec_corr*pmra_pmdec_corr), 0)
            )""",
        "position_error": """
            SQRT(ra_error*ra_error*dec_error*dec_error
                 *(1.0 - ra_dec_corr*ra_dec_corr))""",
    }
