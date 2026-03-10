import sys
import numpy as np

# --- Режим расчёта ---
DRAFT = "--draft" in sys.argv

# --- Геометрия ---
R_in   = 0.030       # м
R_out  = 0.060       # м
N_pads = 6
beta   = 2 * np.pi / N_pads

# --- Клин ---
h_out  = 50e-6       # м — зазор на выходной кромке
K_nom  = 2.0         # основной параметр клина h_in/h_out

# --- Смазка ---
mu     = 0.020       # Па·с

# --- Режим ---
n_rpm  = 3000
omega  = 2 * np.pi * n_rpm / 60   # рад/с

# --- Сетка ---
if DRAFT:
    N_r     = 50
    N_theta = 75
else:
    N_r     = 200
    N_theta = 300

# --- Солвер (SOR) ---
SOR_W       = 1.5
MAX_ITER    = 50000
TOL         = 1e-5
CHECK_EVERY = 50 if DRAFT else 500

# --- Sweep ---
K_values = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# --- Конфигурации текстуры ---
# H_p   — безразмерная глубина (h_p / h_out)
# A_tex — безразмерная полуось по r     (a_r / (R_out - R_in))
# B_tex — безразмерная полуось по θ     (a_theta / beta)
# theta_start_frac, theta_end_frac — зона по θ (доли от β)
# N_r_tex, N_theta_tex — число лунок по r и θ
TEXTURE_CONFIGS = {
    "T1": dict(
        H_p=0.2, A_tex=0.060, B_tex=0.040,
        r_start=R_in, r_end=R_out,
        theta_start_frac=0.1, theta_end_frac=0.7,
        N_r_tex=5, N_theta_tex=8,
    ),
    "T2": dict(
        H_p=0.4, A_tex=0.060, B_tex=0.040,
        r_start=R_in, r_end=R_out,
        theta_start_frac=0.1, theta_end_frac=0.7,
        N_r_tex=5, N_theta_tex=8,
    ),
    "T3": dict(
        H_p=0.2, A_tex=0.060, B_tex=0.040,
        r_start=R_in, r_end=R_out,
        theta_start_frac=0.3, theta_end_frac=0.9,
        N_r_tex=5, N_theta_tex=8,
    ),
}
# T1 и T2: зона θ ∈ [0.1β, 0.7β] — входная/средняя часть клина
# T3:       зона θ ∈ [0.3β, 0.9β] — средняя/выходная часть
