import numpy as np
from params import N_phi, N_Z, R, c
from reynolds_solver.utils import create_H_with_ellipsoidal_depressions

phi_1D = np.linspace(0, 2 * np.pi, N_phi)
Z_1D   = np.linspace(-1, 1, N_Z)
Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
d_phi = phi_1D[1] - phi_1D[0]
d_Z   = Z_1D[1] - Z_1D[0]


def build_texture_centers(cfg: dict):
    """
    Шахматная раскладка лунок.
    Нечётные ряды сдвинуты на полшага по φ.
    Ни один центр не выходит за (phi_start+B_tex, phi_end-B_tex).
    """
    phi_start = np.deg2rad(cfg["phi_start_deg"])
    phi_end   = np.deg2rad(cfg["phi_end_deg"])
    A_tex, B_tex     = cfg["A_tex"], cfg["B_tex"]
    N_phi_tex, N_Z_tex = cfg["N_phi_tex"], cfg["N_Z_tex"]

    # Центры по φ: равномерно внутри допустимого диапазона
    phi_in_start = phi_start + B_tex
    phi_in_end   = phi_end   - B_tex
    Lphi  = phi_in_end - phi_in_start
    if Lphi <= 0:
        raise ValueError(f"phi-range слишком мал для B_tex={B_tex:.4f}: Lphi={Lphi:.4f}")
    if N_phi_tex < 1 or N_Z_tex < 1:
        raise ValueError("N_phi_tex и N_Z_tex должны быть >= 1")
    sphi  = Lphi / N_phi_tex

    phi_even = phi_in_start + 0.5 * sphi + sphi * np.arange(N_phi_tex)
    phi_odd  = phi_even + 0.5 * sphi   # сдвиг нечётных рядов

    # Центры по Z: равномерно
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


def H_smooth(epsilon):
    return 1.0 + epsilon * np.cos(Phi_mesh)


def H_textured(epsilon, cfg: dict):
    H0 = H_smooth(epsilon)
    phi_c, Z_c = build_texture_centers(cfg)
    return create_H_with_ellipsoidal_depressions(
        H0, cfg["H_p"], Phi_mesh, Z_mesh, phi_c, Z_c, cfg["A_tex"], cfg["B_tex"]
    )
