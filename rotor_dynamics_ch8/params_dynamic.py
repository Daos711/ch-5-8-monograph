import sys
import numpy as np

DRAFT = "--draft" in sys.argv

# --- Геометрия (те же что в главе 5) ---
R   = 0.035       # м
c   = 50e-6       # м
L   = 0.056       # м

# --- Смазка ---
mu  = 0.01105     # Па·с

# --- Режим ---
n_rpm  = 2980
omega  = 2 * np.pi * n_rpm / 60
U      = omega * R

# --- Масштабирование ---
pressure_scale = (6 * mu * U * R) / c**2
load_scale     = pressure_scale * R * L / 2

# --- Сетка ---
if DRAFT:
    N_phi = 90
    N_Z   = 30
else:
    N_phi = 360
    N_Z   = 120
# ВАЖНО: endpoint=False для периодической координаты φ

# --- Солвер ---
SOR_W       = 1.5
MAX_ITER    = 30000 if not DRAFT else 5000
TOL         = 1e-5
CHECK_EVERY = 500 if not DRAFT else 100

# --- Шаги конечных разностей ---
dx = 0.005 * c    # по x и y (одинаковый)
dv = 0.005 * U    # по xdot и ydot (одинаковый)

# --- Sweep ---
if DRAFT:
    epsilon_values = [0.2, 0.4, 0.6, 0.8]
else:
    epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
epsilon_orbit  = 0.6

# --- Ротор ---
m_rotor  = 10.0
m_unb    = 0.001
e_unb    = 0.001

# --- Текстура T2 из главы 5 ---
TEXTURE_CONFIG = dict(
    H_p=0.4,
    A_tex=2 * 0.00241 / 0.056,
    B_tex=0.002214 / 0.035,
    phi_start_deg=90,
    phi_end_deg=270,
    N_phi_tex=8,
    N_Z_tex=11,
)
