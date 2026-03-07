import numpy as np
from params import load_scale, friction_scale, eta, c


def compute_load(P, phi_1D, Z_1D):
    """Несущая сила, Н."""
    cos_m = np.cos(np.meshgrid(phi_1D, Z_1D)[0])
    sin_m = np.sin(np.meshgrid(phi_1D, Z_1D)[0])
    Fx_nd = np.trapz(np.trapz(P * cos_m, phi_1D, axis=1), Z_1D)
    Fy_nd = np.trapz(np.trapz(P * sin_m, phi_1D, axis=1), Z_1D)
    F_nd  = np.sqrt(Fx_nd**2 + Fy_nd**2)
    return F_nd * load_scale


def _dP_dphi(P, d_phi):
    """Производная давления по φ (центральные разности, периодичность)."""
    dPdphi = np.zeros_like(P)
    dPdphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dPdphi[:, 0]    = (P[:, 1]  - P[:, -2])  / (2 * d_phi)
    dPdphi[:, -1]   = dPdphi[:, 0]
    return dPdphi


def compute_friction(P, H, phi_1D, Z_1D, d_phi):
    """Сила трения, Н."""
    dP = _dP_dphi(P, d_phi)
    integrand = 1.0 / H + 3.0 * H * dP
    f_nd = np.trapz(np.trapz(integrand, phi_1D, axis=1), Z_1D)
    return f_nd * friction_scale


def compute_Qout(P, H, phi_1D, Z_1D, d_Z):
    """
    Осевые утечки через оба торца, м³/с.

    Безразмерный поток по Z (из уравнения Рейнольдса):
        q_Z = -(D/L)^2 * H^3 * dP/dZ / (d_phi * d_Z)  [безразм.]

    На торце Z=+1 (строка -1): dP/dZ ≈ (P[-1,:] - P[-2,:]) / d_Z = (0 - P[-2,:]) / d_Z
    На торце Z=-1 (строка  0): dP/dZ ≈ (P[1,:]  - P[0,:])  / d_Z = (P[1,:] - 0)  / d_Z

    Суммарный расход (оба торца одинаковы по модулю при симметрии):
        Q_out = (Q_top + Q_bot) / 2

    Размерный множитель:
        Q_scale = pressure_scale * c^3 / (12 * eta) * R
    """
    from params import pressure_scale, R
    Q_scale = pressure_scale * c**3 / (12 * eta) * R

    # Торец Z = +1
    H_top  = H[-1, :]
    dPdZ_top = (0.0 - P[-2, :]) / d_Z
    q_top_nd = np.trapz(-H_top**3 * dPdZ_top, phi_1D)

    # Торец Z = -1
    H_bot  = H[0, :]
    dPdZ_bot = (P[1, :] - 0.0) / d_Z
    q_bot_nd = np.trapz(H_bot**3 * dPdZ_bot, phi_1D)

    Q_out = (abs(q_top_nd) + abs(q_bot_nd)) / 2 * Q_scale
    return Q_out
