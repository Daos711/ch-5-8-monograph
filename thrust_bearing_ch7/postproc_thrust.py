import numpy as np
from geometry_thrust import R_mesh, r_1D, theta_1D


def compute_load(P):
    """W = integral p(r,theta) * r dr dtheta  (один сектор, Н)"""
    return np.trapz(np.trapz(P * R_mesh, r_1D, axis=0), theta_1D)


def compute_friction_moment(H):
    """Вязкий момент трения (модель Куэтта), один сектор, Н·м."""
    from params_thrust import mu, omega
    tau = mu * omega * R_mesh / H
    return np.trapz(np.trapz(tau * R_mesh**2, r_1D, axis=0), theta_1D)


def compute_friction_coeff(W, M_f):
    from params_thrust import R_in, R_out
    R_m = 2.0 / 3.0 * (R_out**3 - R_in**3) / (R_out**2 - R_in**2)
    return M_f / (W * R_m)


def compute_flow_out(P, H):
    from params_thrust import mu
    d_r = r_1D[1] - r_1D[0]
    dPdr_out = (0.0 - P[-2, :]) / d_r
    H_out = H[-1, :]
    q = -H_out**3 / (12.0 * mu) * dPdr_out * r_1D[-1]
    return np.trapz(q, theta_1D)


def compute_hmin(H):
    return float(np.min(H))


def compute_pmax(P):
    return float(np.max(P))


def compute_gains(res_tex, res_smooth, eps=1e-12):
    """G > 1 — улучшение по показателю."""
    return {
        "G_W":    res_tex["W"]       / max(res_smooth["W"],      eps),
        "G_Mf":   res_smooth["M_f"]  / max(res_tex["M_f"],       eps),
        "G_fT":   res_smooth["f_T"]  / max(res_tex["f_T"],       eps),
        "G_Q":    res_smooth["Q"]    / max(res_tex["Q"],          eps),
        "G_hmin": res_tex["h_min"]   / max(res_smooth["h_min"],   eps),
        "G_pmax": res_smooth["p_max"] / max(res_tex["p_max"],     eps),
    }
