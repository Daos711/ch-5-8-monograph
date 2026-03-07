import numpy as np

# --- Геометрия ---
R   = 0.035
c   = 0.00005
L   = 0.056

# --- Смазка ---
eta = 0.01105

# --- Режим работы ---
n_rpm  = 2980
omega  = 2 * np.pi * n_rpm / 60
U      = omega * R

# --- Сетка ---
N_phi = 500
N_Z   = 500

# --- Солвер ---
SOR_W   = 1.5
MAX_ITER    = 50000
TOL         = 1e-5
CHECK_EVERY = 500

# --- Масштабирование ---
pressure_scale = (6 * eta * U * R) / c**2
load_scale     = pressure_scale * R * L / 2
friction_scale = (eta * U * R * L) / c

# --- Конфигурации текстуры ---
# H_p            — безразмерная глубина (h_p / c)
# A_tex          — безразмерная полуось по Z  (2*a / L,  т.к. Z ∈ [-1,1])
# B_tex          — безразмерная полуось по φ  (b / R)
# phi_start_deg, phi_end_deg — зона нанесения, градусы
# N_phi_tex, N_Z_tex — число лунок по φ и Z

TEXTURE_CONFIGS = {
    "T1": dict(
        H_p=0.2,
        A_tex=2 * 0.00241 / 0.056,
        B_tex=0.002214 / 0.035,
        phi_start_deg=90,
        phi_end_deg=270,
        N_phi_tex=8,
        N_Z_tex=11,
    ),
    "T2": dict(
        H_p=0.4,
        A_tex=2 * 0.00241 / 0.056,
        B_tex=0.002214 / 0.035,
        phi_start_deg=90,
        phi_end_deg=270,
        N_phi_tex=8,
        N_Z_tex=11,
    ),
    "T3": dict(
        H_p=0.2,
        A_tex=2 * 0.00241 / 0.056,
        B_tex=0.002214 / 0.035,
        phi_start_deg=0,
        phi_end_deg=180,
        N_phi_tex=8,
        N_Z_tex=11,
    ),
}

# --- Расчётные точки ---
epsilon_nom    = 0.6
epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
