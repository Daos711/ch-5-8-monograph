import os
import numpy as np
import matplotlib.pyplot as plt

from params import *
from geometry import phi_1D, Z_1D, d_phi, d_Z, Phi_mesh, Z_mesh, H_smooth, H_textured
from postproc import (compute_load, compute_phi_load, compute_friction,
                      compute_Qout, compute_gains, compute_coverage)
from reynolds_solver.api import solve_reynolds

os.makedirs("plots", exist_ok=True)

def solve(H):
    return solve_reynolds(H, d_phi, d_Z, R, L,
                          omega=SOR_W,        # SOR relaxation, не угловая скорость
                          tol=TOL, max_iter=MAX_ITER, check_every=CHECK_EVERY)

# ── 1. Эталон: гладкий при epsilon_nom ───────────────────────────────────────
H_s = H_smooth(epsilon_nom)
P_s, _, iter_s = solve(H_s)
print(f"Гладкий: сошлось за {iter_s} итераций")

F_s_nom      = compute_load(P_s, phi_1D, Z_1D)
f_s_nom      = compute_friction(P_s, H_s, phi_1D, Z_1D, d_phi)
mu_s_nom     = f_s_nom / F_s_nom
Q_s_nom      = compute_Qout(P_s, H_s, phi_1D, Z_1D, d_Z)
phi_s_nom    = compute_phi_load(P_s, phi_1D, Z_1D)

# ── 2. Три конфигурации при epsilon_nom ──────────────────────────────────────
results_nom    = {}
results_curves = {}

# Кривые гладкого
F_s_curves, mu_s_curves, Q_s_curves = [], [], []
for eps in epsilon_values:
    H_s_e = H_smooth(eps)
    P_s_e, _, _ = solve(H_s_e)
    F_e  = compute_load(P_s_e, phi_1D, Z_1D)
    f_e  = compute_friction(P_s_e, H_s_e, phi_1D, Z_1D, d_phi)
    Q_e  = compute_Qout(P_s_e, H_s_e, phi_1D, Z_1D, d_Z)
    F_s_curves.append(F_e); mu_s_curves.append(f_e/F_e); Q_s_curves.append(Q_e*1e6)

for name, cfg in TEXTURE_CONFIGS.items():
    # --- при epsilon_nom ---
    H_t = H_textured(epsilon_nom, cfg)
    P_t, _, iter_t = solve(H_t)
    print(f"{name}: сошлось за {iter_t} итераций")

    F_t  = compute_load(P_t, phi_1D, Z_1D)
    f_t  = compute_friction(P_t, H_t, phi_1D, Z_1D, d_phi)
    mu_t = f_t / F_t
    Q_t  = compute_Qout(P_t, H_t, phi_1D, Z_1D, d_Z)
    phi_t = compute_phi_load(P_t, phi_1D, Z_1D)
    gains = compute_gains(F_t, F_s_nom, mu_t, mu_s_nom, Q_t, Q_s_nom,
                          H_t, epsilon_nom)
    varphi = compute_coverage(cfg)

    results_nom[name] = dict(F=F_t, mu=mu_t, Q=Q_t, phi_load=phi_t,
                              gains=gains, varphi=varphi, P=P_t, H=H_t)

    # --- кривые от ε ---
    F_list, mu_list, Q_list = [], [], []
    for eps in epsilon_values:
        H_t_e = H_textured(eps, cfg)
        P_t_e, _, _ = solve(H_t_e)
        F_e  = compute_load(P_t_e, phi_1D, Z_1D)
        f_e  = compute_friction(P_t_e, H_t_e, phi_1D, Z_1D, d_phi)
        Q_e  = compute_Qout(P_t_e, H_t_e, phi_1D, Z_1D, d_Z)
        F_list.append(F_e); mu_list.append(f_e/F_e); Q_list.append(Q_e*1e6)
        print(f"  ε={eps:.2f}: F={F_e:.1f}N  μ={f_e/F_e:.5f}")

    results_curves[name] = dict(F=F_list, mu=mu_list, Q=Q_list)

# ── 3. Сводная таблица в терминал ────────────────────────────────────────────
print(f"\n=== Таблица 5.2: параметры текстуры ===")
print(f"{'':4s} {'H_p':>6} {'A_tex':>8} {'B_tex':>8} {'phi_start':>10} {'phi_end':>8} {'phi':>6}")
for name, cfg in TEXTURE_CONFIGS.items():
    v = results_nom[name]["varphi"]
    print(f"{name:4s} {cfg['H_p']:>6.2f} {cfg['A_tex']:>8.4f} {cfg['B_tex']:>8.4f} "
          f"{cfg['phi_start_deg']:>10d} {cfg['phi_end_deg']:>8d} {v:>6.3f}")

print(f"\n=== Таблица 5.3: результаты при ε = {epsilon_nom} ===")
print(f"{'':12s} {'F, Н':>10} {'phi_load,°':>12} {'μ':>10} {'Q, мл/с':>10} "
      f"{'G_F':>6} {'G_f':>6} {'G_Q':>6} {'G_h':>6}")
print(f"{'Гладкий':12s} {F_s_nom:>10.1f} {phi_s_nom:>12.1f} {mu_s_nom:>10.5f} "
      f"{Q_s_nom*1e6:>10.4f}  {'---':>5}  {'---':>5}  {'---':>5}  {'---':>5}")
for name, res in results_nom.items():
    g = res["gains"]
    print(f"{name:12s} {res['F']:>10.1f} {res['phi_load']:>12.1f} {res['mu']:>10.5f} "
          f"{res['Q']*1e6:>10.4f} {g['G_F']:>6.3f} {g['G_f']:>6.3f} "
          f"{g['G_Q']:>6.3f} {g['G_h']:>6.3f}")

# ── 4. Графики ───────────────────────────────────────────────────────────────
STYLES = {
    "smooth": ("b-",  "o", "Гладкий"),
    "T1":     ("r--", "s", "T1"),
    "T2":     ("g-.", "^", "T2"),
    "T3":     ("m:",  "D", "T3"),
}
eps_arr = np.array(epsilon_values)
Z_idx = np.argmin(np.abs(Z_1D))


def save(fname):
    plt.savefig(f'plots/{fname}.png', dpi=300)
    plt.close()


# График: P(φ) при Z=0, гладкий + T1
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(phi_1D, P_s[Z_idx, :], 'b-',  linewidth=1.5, label='Гладкий')
ax.plot(phi_1D, results_nom["T1"]["P"][Z_idx, :], 'r--', linewidth=1.5, label='T1')
ax.set_xlabel('φ, рад'); ax.set_ylabel('P')
ax.legend(); ax.grid(True); plt.tight_layout(); save('fig_P_phi_comparison')

# Графики 3D полей давления
def plot_3d(P, fname):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Phi_mesh, Z_mesh, P, cmap='plasma', linewidth=0, antialiased=False)
    ax.set_xlabel('φ, рад'); ax.set_ylabel('Z'); ax.set_zlabel('P')
    plt.tight_layout(); save(fname)

plot_3d(P_s, 'fig_P3D_smooth')
for name in TEXTURE_CONFIGS:
    plot_3d(results_nom[name]["P"], f'fig_P3D_{name}')

# Карты кавитации: cav_mask = (P <= 0)
def plot_cav(P, fname):
    cav = (P <= 0).astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(Phi_mesh, Z_mesh, cav, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('φ, рад'); ax.set_ylabel('Z')
    plt.tight_layout(); save(fname)

plot_cav(P_s, 'fig_cav_smooth')
for name in TEXTURE_CONFIGS:
    plot_cav(results_nom[name]["P"], f'fig_cav_{name}')

# Кривые F(ε), μ(ε), Q(ε)
for metric, ylabel, fname, s_data in [
    ("F",  "F, Н",   "fig_F_vs_epsilon",  F_s_curves),
    ("mu", "μ",      "fig_mu_vs_epsilon",  mu_s_curves),
    ("Q",  "Q, мл/с","fig_Q_vs_epsilon",   Q_s_curves),
]:
    fig, ax = plt.subplots(figsize=(7, 5))
    ls, mk, lbl = STYLES["smooth"]
    ax.plot(eps_arr, s_data, ls, marker=mk, linewidth=1.5, markersize=5, label=lbl)
    for name in TEXTURE_CONFIGS:
        ls, mk, lbl = STYLES[name]
        ax.plot(eps_arr, results_curves[name][metric], ls, marker=mk,
                linewidth=1.5, markersize=5, label=lbl)
    ax.set_xlabel('ε'); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(True); plt.tight_layout(); save(fname)

# Сводный bar-chart показателей G при epsilon_nom
names  = list(TEXTURE_CONFIGS.keys())
G_keys = ["G_F", "G_f", "G_Q", "G_h"]
G_labels = ["$G_F$", "$G_f$", "$G_Q$", "$G_h$"]
x = np.arange(len(names))
width = 0.18

fig, ax = plt.subplots(figsize=(8, 5))
for i, (gk, gl) in enumerate(zip(G_keys, G_labels)):
    vals = [results_nom[n]["gains"][gk] for n in names]
    ax.bar(x + i*width, vals, width, label=gl)
ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(names)
ax.set_ylabel('G')
ax.legend(); ax.grid(True, axis='y'); plt.tight_layout(); save('fig_gains_nom')

print("\nГотово. Графики сохранены в plots/")
