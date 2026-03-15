"""
Динамический солвер Рейнольдса — писан с нуля (SOR, CPU + numba).

Уравнение:
  ∂/∂φ [H³ ∂P/∂φ] + (2R/L)² ∂/∂Z [H³ ∂P/∂Z] = ∂H/∂φ + 2*(R/U)*squeeze

  squeeze = (xdot*cos(φ) + ydot*sin(φ)) / c
"""

import numpy as np
from numba import njit
from params_dynamic import R, L, U, c, SOR_W, TOL, MAX_ITER, CHECK_EVERY
from geometry_dynamic import d_phi, d_Z, phi_1D


def _precompute(H):
    """Коэффициенты дискретизации (CPU, numpy)."""
    N_Z, N_phi = H.shape
    D_over_L = 2.0 * R / L
    alpha_sq = (D_over_L * d_phi / d_Z) ** 2

    # Полуточки по φ (периодическое)
    H_ip = 0.5 * (H + np.roll(H, -1, axis=1))   # (i, i+1)
    H_im = np.roll(H_ip, 1, axis=1)               # (i-1, i)

    A = H_ip ** 3    # коэфф. для P_{i+1,j}
    B = H_im ** 3    # коэфф. для P_{i-1,j}

    # Полуточки по Z
    H_jp = 0.5 * (H[:-1, :] + H[1:, :])

    C = np.zeros_like(H)
    D = np.zeros_like(H)
    C[1:-1, :] = alpha_sq * H_jp[1:, :] ** 3
    D[1:-1, :] = alpha_sq * H_jp[:-1, :] ** 3

    E = A + B + C + D

    # Статическая часть RHS: d_phi * (H_{i+1/2} - H_{i-1/2})
    F = d_phi * (H_ip - H_im)

    return A, B, C, D, E, F


def solve_dynamic(H, xdot, ydot):
    """
    Решить динамическое уравнение Рейнольдса.

    Parameters
    ----------
    H : ndarray (N_Z, N_phi) — безразмерное поле зазора
    xdot, ydot : float — физические скорости (м/с)

    Returns
    -------
    P : ndarray (N_Z, N_phi) — безразмерное давление
    converged : bool
    n_iter : int
    """
    N_Z, N_phi = H.shape
    A, B, C, D, E, F = _precompute(H)

    # Динамическая добавка к RHS
    squeeze = (xdot * np.cos(phi_1D) + ydot * np.sin(phi_1D)) / c
    dyn_rhs = d_phi ** 2 * 2.0 * (R / U) * squeeze   # (N_phi,)
    F = F + dyn_rhs[np.newaxis, :]

    P = np.zeros((N_Z, N_phi), dtype=np.float64)
    P, delta, n_iter = _sor_loop(P, A, B, C, D, E, F,
                                  SOR_W, MAX_ITER, TOL, CHECK_EVERY,
                                  N_Z, N_phi)
    converged = delta < TOL
    return P, converged, n_iter


@njit(cache=True)
def _sor_loop(P, A, B, C, D, E, F, omega, max_iter, tol, check_every,
              N_Z, N_phi):
    """Gauss-Seidel SOR с кавитацией P>=0, периодичностью по φ."""
    delta = 1.0
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        for j in range(1, N_Z - 1):        # Z: пропуск границ (P=0)
            for i in range(N_phi):          # φ: периодический
                ip = (i + 1) % N_phi
                im = (i - 1) % N_phi

                P_gs = (A[j, i] * P[j, ip] +
                        B[j, i] * P[j, im] +
                        C[j, i] * P[j + 1, i] +
                        D[j, i] * P[j - 1, i] -
                        F[j, i]) / E[j, i]

                P_new = (1.0 - omega) * P[j, i] + omega * P_gs
                if P_new < 0.0:
                    P_new = 0.0
                P[j, i] = P_new

        if iteration % check_every == 0:
            max_P = 0.0
            max_res = 0.0
            for j in range(1, N_Z - 1):
                for i in range(N_phi):
                    ip = (i + 1) % N_phi
                    im = (i - 1) % N_phi
                    res = (A[j, i] * P[j, ip] +
                           B[j, i] * P[j, im] +
                           C[j, i] * P[j + 1, i] +
                           D[j, i] * P[j - 1, i] -
                           E[j, i] * P[j, i] -
                           F[j, i])
                    r_abs = abs(res)
                    if r_abs > max_res:
                        max_res = r_abs
                    p_abs = abs(P[j, i])
                    if p_abs > max_P:
                        max_P = p_abs

            delta = max_res / (max_P + 1e-30)
            n_iter = iteration
            if delta < tol:
                break

    return P, delta, n_iter
