import numpy as np
from params_thrust import R_in, R_out, beta, N_r, N_theta, h_out

r_1D     = np.linspace(R_in, R_out, N_r)
theta_1D = np.linspace(0, beta, N_theta)
R_mesh, Theta_mesh = np.meshgrid(r_1D, theta_1D, indexing='ij')
# R_mesh.shape = (N_r, N_theta)

d_r     = r_1D[1] - r_1D[0]
d_theta = theta_1D[1] - theta_1D[0]


def build_texture_centers(cfg):
    A_tex, B_tex = cfg["A_tex"], cfg["B_tex"]
    N_r_tex, N_theta_tex = cfg["N_r_tex"], cfg["N_theta_tex"]
    r_start, r_end = cfg["r_start"], cfg["r_end"]
    theta_start = cfg["theta_start_frac"] * beta
    theta_end   = cfg["theta_end_frac"]   * beta

    a_r     = A_tex * (R_out - R_in)
    a_theta = B_tex * beta

    r_lo = r_start + a_r
    r_hi = r_end   - a_r
    t_lo = theta_start + a_theta
    t_hi = theta_end   - a_theta

    s_r = (r_hi - r_lo) / N_r_tex
    s_t = (t_hi - t_lo) / N_theta_tex

    r_even = r_lo + 0.5 * s_r + s_r * np.arange(N_r_tex)
    t_even = t_lo + 0.5 * s_t + s_t * np.arange(N_theta_tex)
    t_odd  = t_even + 0.5 * s_t

    r_list, t_list = [], []
    for i, rc in enumerate(r_even):
        row = t_odd if (i % 2 == 1) else t_even
        for tc in row:
            r_list.append(rc)
            t_list.append(tc)

    return np.array(r_list), np.array(t_list)


def H_smooth(K):
    """Гладкий клиновой зазор, м."""
    h_in = K * h_out
    return h_out + (h_in - h_out) * (1.0 - Theta_mesh / beta)


def add_ellipsoidal_dimples(H_base, cfg):
    """Добавить эллипсоидальные лунки к H_base."""
    H = H_base.copy()
    r_c, t_c = build_texture_centers(cfg)
    a_r     = cfg["A_tex"] * (R_out - R_in)
    a_theta = cfg["B_tex"] * beta
    H_p     = cfg["H_p"] * h_out   # физическая глубина, м

    for rc, tc in zip(r_c, t_c):
        xi_r = (R_mesh - rc) / a_r
        xi_t = (Theta_mesh - tc) / a_theta
        mask = xi_r**2 + xi_t**2 <= 1.0
        H[mask] += H_p * (1.0 - xi_r[mask]**2 - xi_t[mask]**2)

    return H


def H_textured(K, cfg):
    return add_ellipsoidal_dimples(H_smooth(K), cfg)
