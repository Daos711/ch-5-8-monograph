import numpy as np
from params_dynamic import N_phi, N_Z, R, c

# ВАЖНО: endpoint=False — точки 0 и 2π не дублируются
phi_1D = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
Z_1D   = np.linspace(-1, 1, N_Z)
Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
d_phi = phi_1D[1] - phi_1D[0]
d_Z   = Z_1D[1] - Z_1D[0]


def H_base(x, y):
    """Гладкий зазор, безразмерный. x, y — физические смещения (м)."""
    return 1.0 + (x * np.cos(Phi_mesh) + y * np.sin(Phi_mesh)) / c


def build_texture_centers(cfg):
    """
    Шахматная раскладка лунок.
    Логика скопирована из journal_bearing_ch5/geometry.py.
    """
    phi_start = np.deg2rad(cfg["phi_start_deg"])
    phi_end   = np.deg2rad(cfg["phi_end_deg"])
    A_tex, B_tex     = cfg["A_tex"], cfg["B_tex"]
    N_phi_tex, N_Z_tex = cfg["N_phi_tex"], cfg["N_Z_tex"]

    phi_in_start = phi_start + B_tex
    phi_in_end   = phi_end   - B_tex
    Lphi = phi_in_end - phi_in_start
    if Lphi <= 0:
        raise ValueError(f"phi-range слишком мал для B_tex={B_tex:.4f}")

    sphi = Lphi / N_phi_tex
    phi_even = phi_in_start + 0.5 * sphi + sphi * np.arange(N_phi_tex)
    phi_odd  = phi_even + 0.5 * sphi

    delta_Z_gap    = (2 - 2 * N_Z_tex * A_tex) / (N_Z_tex - 1)
    delta_Z_center = 2 * A_tex + delta_Z_gap
    Z_c_values     = (-1 + A_tex) + delta_Z_center * np.arange(N_Z_tex)

    phi_c_list, Z_c_list = [], []
    for j, Zc in enumerate(Z_c_values):
        row = phi_odd if (j % 2 == 1) else phi_even
        for phic in row:
            phi_c_list.append(phic)
            Z_c_list.append(Zc)

    return np.array(phi_c_list), np.array(Z_c_list)


def _add_ellipsoidal_dimples(H0, cfg):
    """Добавить эллипсоидальные лунки (логика из ch5 reynolds_solver.utils)."""
    H = H0.copy()
    H_p   = cfg["H_p"]
    A_tex = cfg["A_tex"]
    B_tex = cfg["B_tex"]
    phi_c, Z_c = build_texture_centers(cfg)

    for k in range(len(phi_c)):
        delta_phi = np.arctan2(np.sin(Phi_mesh - phi_c[k]),
                               np.cos(Phi_mesh - phi_c[k]))
        expr = (delta_phi / B_tex) ** 2 + ((Z_mesh - Z_c[k]) / A_tex) ** 2
        inside = expr <= 1
        H[inside] += H_p * np.sqrt(1 - expr[inside])

    return H


def H_textured(x, y, cfg):
    """Зазор с текстурой. x, y — физические смещения (м)."""
    return _add_ellipsoidal_dimples(H_base(x, y), cfg)
