# Auto-generated friction parameters
# Model: tau_fric = Fc * tanh(TANH_K * qdot) + B * qdot
# Fc/B use the symmetric average of positive/negative sweeps.
# qdot unit: rad/s
import numpy as np

TANH_K = 20.0

FRIC_FC = np.array([
    0.01960758,
    0.02365714,
    0.03601446,
    0.02148591,
    0.01329498,
    0.02775467,
    0.04305190,
    0.02232790,
    0.03620720,
    0.02029402,
    0.03399841,
    0.03115494,
    0.02860342,
    0.02195292,
    0.03828288,
    0.02981830,
    0.02233388,
    0.01593889,
    0.03208451,
    0.02652924,
], dtype=np.float64)

FRIC_B = np.array([
    0.09430760,
    0.09199139,
    0.09315150,
    0.09022948,
    0.10378990,
    0.09078315,
    0.08800658,
    0.09131011,
    0.09509403,
    0.09863172,
    0.09204688,
    0.09335977,
    0.09645739,
    0.09662044,
    0.08869751,
    0.09531121,
    0.09696898,
    0.09501765,
    0.09519648,
    0.09343037,
], dtype=np.float64)

def compute_friction(qdot, scale=1.0):
    qdot = np.asarray(qdot, dtype=np.float64)
    tau = FRIC_FC * np.tanh(TANH_K * qdot) + FRIC_B * qdot
    return scale * tau
