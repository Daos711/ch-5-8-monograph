import numpy as np
from params import N_phi, N_Z, R, c, H_p, A_tex, B_tex
from params import N_phi_tex, N_Z_tex, phi_start_deg, phi_end_deg

try:
    from reynolds_solver.utils import create_H_with_ellipsoidal_depressions
except ImportError:
    from reynolds_solver_stub.utils import create_H_with_ellipsoidal_depressions

# Сетка
phi_1D = np.linspace(0, 2 * np.pi, N_phi)
Z_1D   = np.linspace(-1, 1, N_Z)
Phi_mesh, Z_mesh = np.meshgrid(phi_1D, Z_1D)
d_phi = phi_1D[1] - phi_1D[0]
d_Z   = Z_1D[1] - Z_1D[0]


def build_texture_centers():
    """Координаты центров лунок в шахматной раскладке."""
    phi_start = np.deg2rad(phi_start_deg)
    phi_end   = np.deg2rad(phi_end_deg)

    # Центры по φ
    delta_phi_gap    = (phi_end - phi_start - 2 * N_phi_tex * B_tex) / (N_phi_tex - 1)
    delta_phi_center = 2 * B_tex + delta_phi_gap
    phi_c_values     = (phi_start + B_tex) + delta_phi_center * np.arange(N_phi_tex)

    # Центры по Z
    delta_Z_gap    = (2 - 2 * N_Z_tex * A_tex) / (N_Z_tex - 1)
    delta_Z_center = 2 * A_tex + delta_Z_gap
    Z_c_values     = (-1 + A_tex) + delta_Z_center * np.arange(N_Z_tex)

    phi_c_grid, Z_c_grid = np.meshgrid(phi_c_values, Z_c_values)
    return phi_c_grid.flatten(), Z_c_grid.flatten()


phi_c_flat, Z_c_flat = build_texture_centers()


def H_smooth(epsilon):
    return 1.0 + epsilon * np.cos(Phi_mesh)


def H_textured(epsilon):
    H0 = H_smooth(epsilon)
    return create_H_with_ellipsoidal_depressions(
        H0, H_p, Phi_mesh, Z_mesh, phi_c_flat, Z_c_flat, A_tex, B_tex
    )
