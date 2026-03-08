import numpy as np
from params_bit import eta, R, L, c


def make_scales(U_eq):
    """
    Масштабы давления/нагрузки/трения для опоры долота.
    U_eq передаётся явно, не берётся из глобальных переменных.
    """
    pressure_scale = (6 * eta * U_eq * R) / c**2
    load_scale     = pressure_scale * R * L / 2
    friction_scale = (eta * U_eq * R * L) / c
    return pressure_scale, load_scale, friction_scale


def compute_load_bit(P, phi_1D, Z_1D, U_eq):
    """
    Несущая сила, Н.
    P shape: (N_Z, N_phi) — ось 0 = Z, ось 1 = phi.
    """
    _, load_scale, _ = make_scales(U_eq)
    Phi_m = np.meshgrid(phi_1D, Z_1D)[0]  # shape (N_Z, N_phi)
    Fx_nd = np.trapz(np.trapz(P * np.cos(Phi_m), phi_1D, axis=1), Z_1D)
    Fy_nd = np.trapz(np.trapz(P * np.sin(Phi_m), phi_1D, axis=1), Z_1D)
    return np.sqrt(Fx_nd**2 + Fy_nd**2) * load_scale


def compute_friction_bit(P, H, phi_1D, Z_1D, U_eq):
    """
    Сила трения, Н.
    P, H shape: (N_Z, N_phi).
    """
    _, _, friction_scale = make_scales(U_eq)
    d_phi = phi_1D[1] - phi_1D[0]
    dPdphi = np.zeros_like(P)
    dPdphi[:, 1:-1] = (P[:, 2:] - P[:, :-2]) / (2 * d_phi)
    dPdphi[:, 0]    = (P[:, 1]  - P[:, -2])  / (2 * d_phi)
    dPdphi[:, -1]   = dPdphi[:, 0]
    integrand = 1.0 / H + 3.0 * H * dPdphi
    f_nd = np.trapz(np.trapz(integrand, phi_1D, axis=1), Z_1D)
    return f_nd * friction_scale


def compute_phi_load_bit(P, phi_1D, Z_1D):
    """Угол нагружения, градусы."""
    Phi_m = np.meshgrid(phi_1D, Z_1D)[0]
    Fx_nd = np.trapz(np.trapz(P * np.cos(Phi_m), phi_1D, axis=1), Z_1D)
    Fy_nd = np.trapz(np.trapz(P * np.sin(Phi_m), phi_1D, axis=1), Z_1D)
    phi = np.degrees(np.arctan2(Fy_nd, Fx_nd))
    return (phi + 360) % 360


def compute_Qout_bit(P, H, phi_1D, Z_1D, U_eq):
    """
    Осевые утечки через оба торца, м³/с.
    P, H shape: (N_Z, N_phi).
    """
    pressure_scale, _, _ = make_scales(U_eq)
    d_Z = Z_1D[1] - Z_1D[0]
    Q_scale_end = pressure_scale * c**3 * R / (6 * eta * L)

    # Торец Z = +1
    H_top    = H[-1, :]
    dPdZ_top = (0.0 - P[-2, :]) / d_Z
    q_top_nd = np.trapz(-H_top**3 * dPdZ_top, phi_1D)

    # Торец Z = -1
    H_bot    = H[0, :]
    dPdZ_bot = (P[1, :] - 0.0) / d_Z
    q_bot_nd = np.trapz(-H_bot**3 * dPdZ_bot, phi_1D)

    return (abs(q_top_nd) + abs(q_bot_nd)) * Q_scale_end


def full_postproc(epsilon, P, H, phi_1D, Z_1D, d_phi, U_eq, label=""):
    """
    Все выходные характеристики опоры шарошки в одном вызове.
    Параметры phi_1D, Z_1D, d_phi, U_eq передаются явно —
    нет скрытых глобальных зависимостей.

    Returns: dict с ключами:
        label, epsilon, F, phi_load, mu,
        h_min_um, lam, regime, PV, I_wear
    """
    from mixed_lubrication import compute_h_min, compute_lambda, classify_regime
    from wear_bit import compute_PV, compute_wear_severity_index

    F      = compute_load_bit(P, phi_1D, Z_1D, U_eq)
    f      = compute_friction_bit(P, H, phi_1D, Z_1D, U_eq)
    mu     = abs(f) / F if F > 0 else 0.0
    phi_ld = compute_phi_load_bit(P, phi_1D, Z_1D)

    h_min  = compute_h_min(epsilon)
    lam    = compute_lambda(epsilon)
    regime = classify_regime(lam)

    PV     = compute_PV(F, U_eq, R, L)
    I_wear = compute_wear_severity_index(F, U_eq, epsilon, R, L, c)

    return dict(
        label    = label,
        epsilon  = epsilon,
        F        = F,
        phi_load = phi_ld,
        mu       = mu,
        h_min_um = h_min * 1e6,
        lam      = lam,
        regime   = regime,
        PV       = PV,
        I_wear   = I_wear,
    )


def print_results_table(results_dict):
    """Сводная таблица в терминал."""
    hdr = (f"{'':10s} {'ε':>6} {'F,Н':>8} {'φ,°':>7} {'μ':>8} "
           f"{'h_min,мкм':>10} {'λ':>6} {'режим':>15} {'PV':>12} {'I_wear':>12}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in results_dict.items():
        print(f"{name:10s} {r['epsilon']:>6.3f} {r['F']:>8.1f} {r['phi_load']:>7.1f} "
              f"{r['mu']:>8.5f} {r['h_min_um']:>10.2f} {r['lam']:>6.2f} "
              f"{r['regime']:>15s} {r['PV']:>12.1f} {r['I_wear']:>12.1e}")
