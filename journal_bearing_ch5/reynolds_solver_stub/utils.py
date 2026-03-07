"""
Utility for creating texture (ellipsoidal depressions) on bearing surface.
Drop-in replacement for reynolds_solver.utils.create_H_with_ellipsoidal_depressions.
"""
import numpy as np


def create_H_with_ellipsoidal_depressions(H0, H_p, Phi_mesh, Z_mesh,
                                           phi_centers, Z_centers,
                                           A_tex, B_tex):
    """
    Add ellipsoidal depressions to a film thickness field.

    Parameters
    ----------
    H0 : ndarray – baseline film thickness (N_Z x N_phi)
    H_p : float  – dimensionless dimple depth
    Phi_mesh, Z_mesh : ndarray – coordinate meshes
    phi_centers, Z_centers : 1-D arrays – centres of dimples
    A_tex : float – dimensionless half-axis in Z direction
    B_tex : float – dimensionless half-axis in phi direction

    Returns
    -------
    H : ndarray – modified film thickness with dimples
    """
    H = H0.copy()
    for phi_c, z_c in zip(phi_centers, Z_centers):
        r2 = ((Phi_mesh - phi_c) / B_tex) ** 2 + ((Z_mesh - z_c) / A_tex) ** 2
        mask = r2 <= 1.0
        H[mask] += H_p * (1.0 - r2[mask])
    return H
