import numpy as np
from params_bit import SOR_W, TOL, MAX_ITER, CHECK_EVERY, R, L
from params_bit import omega_bit, R_bit, R_cone
from geometry_bit import H_smooth, H_textured, phi_1D, Z_1D, d_phi, d_Z
from postproc_bit import compute_load_bit
from kinematics_bit import compute_U_eq
from reynolds_solver.api import solve_reynolds


def solve_bit(H):
    """Статический решатель — переиспользует GPU-солвер из главы 5."""
    return solve_reynolds(H, d_phi, d_Z, R, L,
                          omega=SOR_W, tol=TOL,
                          max_iter=MAX_ITER, check_every=CHECK_EVERY)


def find_operating_point(F_ext, texture_cfg=None,
                         eps_lo=0.05, eps_hi=0.97,
                         tol_rel=0.01, max_iter=30):
    """
    Найти epsilon равновесия при заданной нагрузке F_ext.

    Parameters
    ----------
    F_ext       : float  целевая нагрузка, Н
    texture_cfg : dict или None  (None = гладкий)
    tol_rel     : float  относительный допуск, по умолчанию 1%

    Returns
    -------
    epsilon : float
    P       : ndarray, shape (N_Z, N_phi)
    H       : ndarray, shape (N_Z, N_phi)

    Raises
    ------
    ValueError — если F_ext недостижима в чисто гидродинамической постановке.
        Это физически допустимый исход для высоконагруженной опоры долота:
        означает, что режим требует смешанной или граничной интерпретации.
    """
    U_eq = compute_U_eq(omega_bit, R_bit, R_cone, R)

    def F_at_eps(eps):
        H = H_smooth(eps) if texture_cfg is None else H_textured(eps, texture_cfg)
        P, _, _ = solve_bit(H)
        F = compute_load_bit(P, phi_1D, Z_1D, U_eq)
        return F, P, H

    F_lo, _, _   = F_at_eps(eps_lo)
    F_hi, P_hi, H_hi = F_at_eps(eps_hi)

    if F_lo > F_ext:
        raise ValueError(
            f"F_ext={F_ext:.0f} Н < F при eps={eps_lo:.2f} ({F_lo:.0f} Н). "
            f"Снизьте F_ext или уменьшите eps_lo.")
    if F_hi < F_ext:
        raise ValueError(
            f"Чисто гидродинамическая грузоподъёмность недостаточна: "
            f"при eps={eps_hi:.2f} F={F_hi:.0f} Н < F_ext={F_ext:.0f} Н. "
            f"Режим работы требует смешанной/граничной интерпретации.")

    # Бисекция с относительным допуском
    eps_mid, P_mid, H_mid = eps_lo, None, None
    for _ in range(max_iter):
        eps_mid = (eps_lo + eps_hi) / 2
        F_mid, P_mid, H_mid = F_at_eps(eps_mid)
        if abs(F_mid - F_ext) / F_ext < tol_rel:
            break
        if F_mid < F_ext:
            eps_lo = eps_mid
        else:
            eps_hi = eps_mid

    return eps_mid, P_mid, H_mid
