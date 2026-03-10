import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from params_thrust import (
    R_in, R_out, N_pads, beta, h_out, K_nom,
    mu, omega, n_rpm, N_r, N_theta,
    SOR_W, MAX_ITER, TOL, CHECK_EVERY,
    K_values, TEXTURE_CONFIGS, DRAFT,
)
from geometry_thrust import (
    r_1D, theta_1D, R_mesh, Theta_mesh,
    H_smooth, H_textured,
)
from solver_thrust import solve_reynolds_thrust
from postproc_thrust import (
    compute_load, compute_friction_moment, compute_friction_coeff,
    compute_flow_out, compute_hmin, compute_pmax, compute_gains,
)

os.makedirs("plots", exist_ok=True)

if DRAFT:
    print(f"*** DRAFT-режим: сетка {N_r}×{N_theta} (быстрая проверка) ***\n")
else:
    print(f"*** FINAL-режим: сетка {N_r}×{N_theta} ***\n")

# ============================================================
# Вспомогательная функция
# ============================================================

def solve_with_fallback(H, P_init=None):
    P, conv, n_iter = solve_reynolds_thrust(
        H, mu, omega, r_1D, theta_1D,
        SOR_W=SOR_W, tol=TOL, max_iter=MAX_ITER,
        check_every=CHECK_EVERY, P_init=P_init)
    if not conv:
        print("  [!] SOR_W=1.5 не сошлось, повтор с SOR_W=1.2")
        P, conv, n_iter = solve_reynolds_thrust(
            H, mu, omega, r_1D, theta_1D,
            SOR_W=1.2, tol=TOL, max_iter=MAX_ITER * 2,
            check_every=CHECK_EVERY, P_init=None)
    if not conv:
        print("  [!] ПРЕДУПРЕЖДЕНИЕ: решение не сошлось!")
    return P, conv, n_iter


def full_postproc(P, H):
    """Вычислить все показатели для одного сектора, вернуть dict."""
    W   = compute_load(P)
    M_f = compute_friction_moment(H)
    f_T = compute_friction_coeff(W, M_f)
    Q   = compute_flow_out(P, H)
    h_min = compute_hmin(H)
    p_max = compute_pmax(P)
    return dict(W=W, M_f=M_f, f_T=f_T, Q=Q, h_min=h_min, p_max=p_max,
                P=P, H=H)


# ============================================================
# Режим 1 — расчёт при K_nom
# ============================================================
print(f"=== Расчёт при K_nom = {K_nom} ===\n")

results_nominal = {}

# Гладкий
H = H_smooth(K_nom)
P, conv, n_iter = solve_with_fallback(H)
status = f"сошлось за {n_iter} итераций" if conv else "НЕ СОШЛОСЬ"
print(f"  Гладкий: {status}")
res = full_postproc(P, H)
# Масштаб на полный подшипник
res["W"]   *= N_pads
res["M_f"] *= N_pads
res["Q"]   *= N_pads
results_nominal["smooth"] = res

# Текстурированные
for tag, cfg in TEXTURE_CONFIGS.items():
    H = H_textured(K_nom, cfg)
    P, conv, n_iter = solve_with_fallback(H)
    status = f"сошлось за {n_iter} итераций" if conv else "НЕ СОШЛОСЬ"
    print(f"  {tag}: {status}")
    res = full_postproc(P, H)
    res["W"]   *= N_pads
    res["M_f"] *= N_pads
    res["Q"]   *= N_pads
    results_nominal[tag] = res

# Таблица результатов
print(f"\n=== Таблица: результаты при K_nom = {K_nom} (полный подшипник) ===")
print(f"{'Вариант':<10} {'W, кН':>8} {'M_f, Н·м':>10} {'f_T':>8} "
      f"{'Q, мл/с':>9} {'h_min, мкм':>11} {'p_max, МПа':>11}")
for tag in ["smooth", "T1", "T2", "T3"]:
    r = results_nominal[tag]
    label = "Гладкий" if tag == "smooth" else tag
    print(f"{label:<10} {r['W']/1e3:>8.2f} {r['M_f']:>10.4f} {r['f_T']:>8.5f} "
          f"{r['Q']*1e6:>9.3f} {r['h_min']*1e6:>11.1f} {r['p_max']/1e6:>11.3f}")

# Коэффициенты улучшения
print(f"\n=== Коэффициенты улучшения при K_nom = {K_nom} ===")
print(f"{'Вариант':<10} {'G_W':>7} {'G_Mf':>7} {'G_fT':>7} "
      f"{'G_Q':>7} {'G_hmin':>7} {'G_pmax':>7}")
gains_nominal = {}
for tag in ["T1", "T2", "T3"]:
    g = compute_gains(results_nominal[tag], results_nominal["smooth"])
    gains_nominal[tag] = g
    print(f"{tag:<10} {g['G_W']:>7.4f} {g['G_Mf']:>7.4f} {g['G_fT']:>7.4f} "
          f"{g['G_Q']:>7.4f} {g['G_hmin']:>7.4f} {g['G_pmax']:>7.4f}")

# ============================================================
# Режим 2 — sweep по K_values
# ============================================================
print(f"\n=== Sweep по K = {K_values} ===\n")

results_sweep = {}

# Гладкий sweep
sw = {k: [] for k in ["K", "W", "f_T", "Q", "h_min", "p_max"]}
P_prev = None
for K in K_values:
    H = H_smooth(K)
    P, conv, n_iter = solve_with_fallback(H, P_init=P_prev)
    P_prev = P.copy()
    status = f"сошлось за {n_iter}" if conv else "НЕ СОШЛОСЬ"
    print(f"  Гладкий K={K}: {status}")
    r = full_postproc(P, H)
    sw["K"].append(K)
    sw["W"].append(r["W"] * N_pads)
    sw["f_T"].append(r["f_T"])
    sw["Q"].append(r["Q"] * N_pads)
    sw["h_min"].append(r["h_min"])
    sw["p_max"].append(r["p_max"])
results_sweep["smooth"] = sw

# Текстурированные sweep
for tag, cfg in TEXTURE_CONFIGS.items():
    sw = {k: [] for k in ["K", "W", "f_T", "Q", "h_min", "p_max"]}
    P_prev = None
    for K in K_values:
        H = H_textured(K, cfg)
        P, conv, n_iter = solve_with_fallback(H, P_init=P_prev)
        P_prev = P.copy()
        status = f"сошлось за {n_iter}" if conv else "НЕ СОШЛОСЬ"
        print(f"  {tag} K={K}: {status}")
        r = full_postproc(P, H)
        sw["K"].append(K)
        sw["W"].append(r["W"] * N_pads)
        sw["f_T"].append(r["f_T"])
        sw["Q"].append(r["Q"] * N_pads)
        sw["h_min"].append(r["h_min"])
        sw["p_max"].append(r["p_max"])
    results_sweep[tag] = sw

# ============================================================
# Графики
# ============================================================
print("\n=== Построение графиков ===")

COLORS = {"smooth": "blue", "T1": "red", "T2": "green", "T3": "purple"}
LABELS = {"smooth": "Гладкий", "T1": "T1", "T2": "T2", "T3": "T3"}

r_mm    = r_1D * 1e3
theta_deg = theta_1D * 180.0 / np.pi
R_mm, Theta_deg = np.meshgrid(r_mm, theta_deg, indexing='ij')

# --- 3D поле p: гладкий ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
P_smooth = results_nominal["smooth"]["P"]
ax.plot_surface(R_mm, Theta_deg, P_smooth / 1e6, cmap='plasma')
ax.set_xlabel('r, мм')
ax.set_ylabel('θ, °')
ax.set_zlabel('p, МПа')
plt.tight_layout()
fig.savefig("plots/fig_P3D_smooth_thrust.png", dpi=300)
plt.close(fig)
print("  -> fig_P3D_smooth_thrust.png")

# --- 3D поле p: T2 ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
P_T2 = results_nominal["T2"]["P"]
ax.plot_surface(R_mm, Theta_deg, P_T2 / 1e6, cmap='plasma')
ax.set_xlabel('r, мм')
ax.set_ylabel('θ, °')
ax.set_zlabel('p, МПа')
plt.tight_layout()
fig.savefig("plots/fig_P3D_T2_thrust.png", dpi=300)
plt.close(fig)
print("  -> fig_P3D_T2_thrust.png")

# --- Карта толщины плёнки: T2 ---
H_T2 = results_nominal["T2"]["H"]
fig, ax = plt.subplots(figsize=(8, 5))
Theta_plot, R_plot = np.meshgrid(theta_deg, r_mm)
im = ax.pcolormesh(Theta_plot, R_plot, H_T2 * 1e6,
                   cmap='viridis', shading='auto')
cb = fig.colorbar(im, ax=ax)
cb.set_label('h, мкм')
ax.set_xlabel('θ, °')
ax.set_ylabel('r, мм')
plt.tight_layout()
fig.savefig('plots/fig_texture_map_T2.png', dpi=300)
plt.close(fig)
print("  -> fig_texture_map_T2.png")

# --- Sweep графики ---
def plot_sweep(ylabel, key, fname, scale=1.0):
    fig, ax = plt.subplots(figsize=(8, 5))
    for tag in ["smooth", "T1", "T2", "T3"]:
        sw = results_sweep[tag]
        vals = np.array(sw[key]) * scale
        ax.plot(sw["K"], vals, 'o-', color=COLORS[tag], label=LABELS[tag])
    ax.set_xlabel('K')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"plots/{fname}", dpi=300)
    plt.close(fig)
    print(f"  -> {fname}")

plot_sweep('W, кН',      'W',     'fig_W_vs_K.png',     scale=1e-3)
plot_sweep('f_T',        'f_T',   'fig_fT_vs_K.png',    scale=1.0)
plot_sweep('Q, мл/с',   'Q',     'fig_Q_vs_K.png',     scale=1e6)
plot_sweep('p_max, МПа', 'p_max', 'fig_pmax_vs_K.png',  scale=1e-6)

# --- Bar-chart коэффициентов улучшения ---
fig, ax = plt.subplots(figsize=(10, 6))
gain_names = ["G_W", "G_Mf", "G_fT", "G_Q", "G_hmin", "G_pmax"]
x = np.arange(len(gain_names))
width = 0.25
for idx, tag in enumerate(["T1", "T2", "T3"]):
    vals = [gains_nominal[tag][gn] for gn in gain_names]
    ax.bar(x + idx * width, vals, width, label=tag, color=COLORS[tag])
ax.axhline(y=1.0, color='black', linestyle='--', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(gain_names)
ax.set_ylabel('G')
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig("plots/fig_gains_K_nom.png", dpi=300)
plt.close(fig)
print("  -> fig_gains_K_nom.png")

print("\nГотово.")
