import os
import numpy as np
import matplotlib.pyplot as plt

from params import *
from geometry import phi_1D, Z_1D, d_phi, d_Z, Phi_mesh, Z_mesh, H_smooth, H_textured
from postproc import compute_load, compute_friction, compute_Qout

try:
    from reynolds_solver.api import solve_reynolds
except ImportError:
    from reynolds_solver_stub.api import solve_reynolds

os.makedirs("plots", exist_ok=True)

# ── Вспомогательная функция вызова солвера ────────────────────────────────────
def solve(H):
    """Статический решатель (xprime=yprime=0 → GPU static path)."""
    return solve_reynolds(
        H, d_phi, d_Z, R, L,
        omega=SOR_OMEGA, tol=TOL, max_iter=MAX_ITER, check_every=CHECK_EVERY
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Поля давления при epsilon_nom
# ═══════════════════════════════════════════════════════════════════════════════
H_s = H_smooth(epsilon_nom)
H_t = H_textured(epsilon_nom)

P_s, delta_s, iter_s = solve(H_s)
P_t, delta_t, iter_t = solve(H_t)
print(f"Гладкий:        сошлось за {iter_s} итераций, delta={delta_s:.2e}")
print(f"Текстурированный: сошлось за {iter_t} итераций, delta={delta_t:.2e}")

Z_idx = np.argmin(np.abs(Z_1D))  # сечение Z = 0

# График 1: P(φ) при Z=0 — гладкий и текстурированный
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(phi_1D, P_s[Z_idx, :], 'b-',  linewidth=1.5)
ax.plot(phi_1D, P_t[Z_idx, :], 'r--', linewidth=1.5)
ax.set_xlabel('φ, рад')
ax.set_ylabel('P')
ax.grid(True)
plt.tight_layout()
plt.savefig('plots/fig_P_phi_comparison.pdf', dpi=300)
plt.savefig('plots/fig_P_phi_comparison.png', dpi=300)
plt.close()

# График 2: 3D-поле давления, гладкий
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Phi_mesh, Z_mesh, P_s, cmap='plasma', linewidth=0, antialiased=False)
ax.set_xlabel('φ, рад')
ax.set_ylabel('Z')
ax.set_zlabel('P')
plt.tight_layout()
plt.savefig('plots/fig_P3D_smooth.pdf', dpi=300)
plt.savefig('plots/fig_P3D_smooth.png', dpi=300)
plt.close()

# График 3: 3D-поле давления, текстурированный
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Phi_mesh, Z_mesh, P_t, cmap='plasma', linewidth=0, antialiased=False)
ax.set_xlabel('φ, рад')
ax.set_ylabel('Z')
ax.set_zlabel('P')
plt.tight_layout()
plt.savefig('plots/fig_P3D_textured.pdf', dpi=300)
plt.savefig('plots/fig_P3D_textured.png', dpi=300)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Кривые F(ε), μ(ε), Q_out(ε)
# ═══════════════════════════════════════════════════════════════════════════════
F_s_list, mu_s_list, Q_s_list = [], [], []
F_t_list, mu_t_list, Q_t_list = [], [], []

for eps in epsilon_values:
    H_s_e = H_smooth(eps)
    H_t_e = H_textured(eps)

    P_s_e, _, _ = solve(H_s_e)
    P_t_e, _, _ = solve(H_t_e)

    F_s = compute_load(P_s_e, phi_1D, Z_1D)
    F_t = compute_load(P_t_e, phi_1D, Z_1D)

    f_s = compute_friction(P_s_e, H_s_e, phi_1D, Z_1D, d_phi)
    f_t = compute_friction(P_t_e, H_t_e, phi_1D, Z_1D, d_phi)

    Q_s = compute_Qout(P_s_e, H_s_e, phi_1D, Z_1D, d_Z)
    Q_t = compute_Qout(P_t_e, H_t_e, phi_1D, Z_1D, d_Z)

    F_s_list.append(F_s);   mu_s_list.append(f_s / F_s);  Q_s_list.append(Q_s * 1e6)
    F_t_list.append(F_t);   mu_t_list.append(f_t / F_t);  Q_t_list.append(Q_t * 1e6)
    print(f"ε={eps:.2f}: F_s={F_s:.1f}N  F_t={F_t:.1f}N  μ_s={f_s/F_s:.5f}  μ_t={f_t/F_t:.5f}")

eps_arr = np.array(epsilon_values)

# График 4: F(ε)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(eps_arr, F_s_list, 'bo-', linewidth=1.5, markersize=5)
ax.plot(eps_arr, F_t_list, 'rs-', linewidth=1.5, markersize=5)
ax.set_xlabel('ε')
ax.set_ylabel('F, Н')
ax.grid(True)
plt.tight_layout()
plt.savefig('plots/fig_F_vs_epsilon.pdf', dpi=300)
plt.savefig('plots/fig_F_vs_epsilon.png', dpi=300)
plt.close()

# График 5: μ(ε)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(eps_arr, mu_s_list, 'bo-', linewidth=1.5, markersize=5)
ax.plot(eps_arr, mu_t_list, 'rs-', linewidth=1.5, markersize=5)
ax.set_xlabel('ε')
ax.set_ylabel('μ')
ax.grid(True)
plt.tight_layout()
plt.savefig('plots/fig_mu_vs_epsilon.pdf', dpi=300)
plt.savefig('plots/fig_mu_vs_epsilon.png', dpi=300)
plt.close()

# График 6: Q_out(ε)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(eps_arr, Q_s_list, 'bo-', linewidth=1.5, markersize=5)
ax.plot(eps_arr, Q_t_list, 'rs-', linewidth=1.5, markersize=5)
ax.set_xlabel('ε')
ax.set_ylabel('Q, мл/с')
ax.grid(True)
plt.tight_layout()
plt.savefig('plots/fig_Q_vs_epsilon.pdf', dpi=300)
plt.savefig('plots/fig_Q_vs_epsilon.png', dpi=300)
plt.close()

print("Готово. Графики сохранены в plots/")
