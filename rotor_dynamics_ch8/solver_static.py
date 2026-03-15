"""
Статический солвер Рейнольдса.

Пытается использовать GPU (reynolds_solver.api), при недоступности —
CPU SOR через solver_dynamic._precompute + _sor_loop.
"""

import numpy as np
from params_dynamic import SOR_W, TOL, MAX_ITER, CHECK_EVERY, R, L
from geometry_dynamic import d_phi, d_Z

_USE_GPU = True
try:
    from reynolds_solver.api import solve_reynolds as _solve_gpu
    # Пробный вызов — проверить доступность GPU
    _test_H = np.ones((4, 4), dtype=np.float64)
    _solve_gpu(_test_H, 0.1, 0.1, R, L, max_iter=1, check_every=1)
except Exception:
    _USE_GPU = False


def solve_static(H):
    """Возвращает P, converged, n_iter."""
    if _USE_GPU:
        P, delta, n_iter = _solve_gpu(H, d_phi, d_Z, R, L,
                                       omega=SOR_W, tol=TOL,
                                       max_iter=MAX_ITER,
                                       check_every=CHECK_EVERY)
        converged = delta < TOL
        return P, converged, n_iter

    # CPU fallback: тот же SOR что в solver_dynamic, но без squeeze
    from solver_dynamic import _precompute, _sor_loop
    N_Z, N_phi = H.shape
    A, B, C, D, E, F = _precompute(H)
    P = np.zeros((N_Z, N_phi), dtype=np.float64)
    P, delta, n_iter = _sor_loop(P, A, B, C, D, E, F,
                                  SOR_W, MAX_ITER, TOL, CHECK_EVERY,
                                  N_Z, N_phi)
    converged = delta < TOL
    return P, converged, n_iter
