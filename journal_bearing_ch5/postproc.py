import numpy as np
from params import load_scale, friction_scale, eta, c, R, pressure_scale


def compute_load(P, phi_1D, Z_1D):
    """Несущая сила, Н."""
    Phi_m = np.meshgrid(phi_1D, Z_1D)[0]
    Fx_nd = np.trapz(np.trapz(P * np.cos(Phi_m), phi_1D, axis=1), Z_1D)
    Fy_nd = np.trapz(np.trapz(P * np.sin(Phi_m), phi_1D, axis=1), Z_1D)
    return np.sqrt(Fx_nd**2 + Fy_nd**2) * load_scale


def compute_phi_load(P, phi_1D, Z_1D):
    """
    Угол нагружения phi_load, градусы — от оси θ=0.
    НЕ путать с psi = c/R (относительный зазор).
    """
    Phi_m = np.meshgrid(phi_1D, Z_1D)[0]
    Fx_nd = np.trapz(np.trapz(P * np.cos(Phi_m), phi_1D, axis=1), Z_1D)
    Fy_nd = np.trapz(np.trapz(P * np.sin(Phi_m), phi_1D, axis=1), Z_1D)
    phi = np.degrees(np.arctan2(Fy_nd, Fx_nd))
    return (phi + 360) % 360   # приводим к диапазону [0°, 360°]


def _dP_dphi(P, d_phi):
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
    Z ∈ [-1,1], поэтому Z = 2z/L => масштаб Q_scale_end = p_scale * c³ * R / (6*eta*L).
    """
    from params import L
    Q_scale_end = pressure_scale * c**3 * R / (6 * eta * L)

    # Торец Z = +1
    H_top    = H[-1, :]
    dPdZ_top = (0.0 - P[-2, :]) / d_Z
    q_top_nd = np.trapz(-H_top**3 * dPdZ_top, phi_1D)

    # Торец Z = -1
    H_bot    = H[0, :]
    dPdZ_bot = (P[1, :] - 0.0) / d_Z
    q_bot_nd = np.trapz(-H_bot**3 * dPdZ_bot, phi_1D)

    # Суммарные утечки через оба торца
    return (abs(q_top_nd) + abs(q_bot_nd)) * Q_scale_end


def compute_gains(F_tex, F_smooth, mu_tex, mu_smooth, Q_tex, Q_smooth,
                  H_tex, epsilon):
    """
    G_F = F_tex / F_smooth
    G_f = mu_tex / mu_smooth
    G_Q = Q_tex / Q_smooth
    G_h = min(H_tex) / (1 - epsilon)   — изменение минимального зазора
    """
    return {
        "G_F": F_tex / F_smooth,
        "G_f": mu_tex / mu_smooth,
        "G_Q": Q_tex / Q_smooth,
        "G_h": float(np.min(H_tex)) / (1.0 - epsilon),
    }


def compute_coverage(cfg):
    """
    Коэффициент покрытия φ = N * π * A_tex * B_tex / (2 * phi_range).
    phi_range = phi_end - phi_start, радианы.
    Формула справедлива при текстурировании по всей длине Z ∈ [-1,1].
    """
    N         = cfg["N_phi_tex"] * cfg["N_Z_tex"]
    phi_range = np.deg2rad(cfg["phi_end_deg"] - cfg["phi_start_deg"])
    return N * np.pi * cfg["A_tex"] * cfg["B_tex"] / (2 * phi_range)
