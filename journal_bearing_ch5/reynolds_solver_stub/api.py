"""
CPU-based Reynolds equation solver (Red-Black SOR).
Drop-in replacement for reynolds_solver.api.solve_reynolds when the GPU
package is not installed.
"""
import numpy as np


def solve_reynolds(H, d_phi, d_Z, R, L, *,
                   omega=1.5, tol=1e-5, max_iter=50000, check_every=500):
    """
    Solve the static Reynolds equation on a (N_Z x N_phi) grid.

    Returns
    -------
    P : ndarray  – dimensionless pressure field (negative values clipped to 0)
    delta : float – final residual
    iters : int   – iterations performed
    """
    Nz, Nphi = H.shape
    P = np.zeros_like(H)
    DL = (2 * R / L)  # diameter-to-length ratio factor

    delta = 0.0

    for it in range(1, max_iter + 1):
        P_old = P.copy() if (it % check_every == 0) else None

        for j in range(1, Nz - 1):
            for i in range(1, Nphi - 1):
                H3 = H[j, i] ** 3

                a_e = H3 / d_phi**2
                a_w = H[j, i-1]**3 / d_phi**2 if i > 0 else H3 / d_phi**2
                a_n = DL**2 * H3 / d_Z**2
                a_s = DL**2 * H3 / d_Z**2

                a_e = (H[j, i]**3 + H[j, i+1]**3) / 2 / d_phi**2 if i < Nphi-1 else H3 / d_phi**2
                a_w = (H[j, i]**3 + H[j, i-1]**3) / 2 / d_phi**2 if i > 0 else H3 / d_phi**2

                H3_n = (H[j, i]**3 + H[j+1, i]**3) / 2 if j < Nz-1 else H3
                H3_s = (H[j, i]**3 + H[j-1, i]**3) / 2 if j > 0 else H3
                a_n = DL**2 * H3_n / d_Z**2
                a_s = DL**2 * H3_s / d_Z**2

                ap = a_e + a_w + a_n + a_s

                # RHS: dH/dphi
                dH = (H[j, i+1] - H[j, i-1]) / (2 * d_phi) if 0 < i < Nphi-1 else 0.0

                rhs = dH + a_e * P[j, i+1] + a_w * P[j, i-1] + a_n * P[j+1, i] + a_s * P[j-1, i]
                P_new = rhs / ap
                P[j, i] += omega * (P_new - P[j, i])

        # Periodic BC in phi
        P[:, 0] = P[:, -2]
        P[:, -1] = P[:, 1]

        # Boundary in Z: P = 0
        P[0, :] = 0.0
        P[-1, :] = 0.0

        # Cavitation: clip negative pressure
        P[P < 0] = 0.0

        if it % check_every == 0 and P_old is not None:
            delta = np.max(np.abs(P - P_old))
            if delta < tol:
                return P, delta, it

    return P, delta, max_iter
