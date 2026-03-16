import numpy as np
from params_dynamic import load_scale
from geometry_dynamic import Phi_mesh, phi_1D, Z_1D

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz


def compute_Fx(P):
    return _trapz(_trapz(P * np.cos(Phi_mesh),
                         phi_1D, axis=1), Z_1D) * load_scale


def compute_Fy(P):
    return _trapz(_trapz(P * np.sin(Phi_mesh),
                         phi_1D, axis=1), Z_1D) * load_scale
