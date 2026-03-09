import os
import numpy as np
import matplotlib.pyplot as plt

from params_bit import (R, L, c, eta, omega_bit, R_bit, R_cone,
                        F_bit_radial, sigma_eq, TEXTURE_CONFIGS,
                        SOR_W, TOL, MAX_ITER, CHECK_EVERY,
                        WOB, k_load)
from geometry_bit import (phi_1D, Z_1D, d_phi, d_Z, Phi_mesh, Z_mesh,
                          H_smooth, H_textured)
from kinematics_bit import compute_U_eq
from mixed_lubrication import compute_h_min, compute_lambda, classify_regime
from wear_bit import compute_PV, compute_wear_severity_index
from postproc_bit import (compute_load_bit, compute_friction_bit,
                          compute_phi_load_bit, full_postproc,
                          print_results_table)
from operating_point import find_operating_point, solve_bit

os.makedirs("plots", exist_ok=True)


def save(fname):
    plt.savefig(f'plots/{fname}.png', dpi=300)
    plt.close()


# ── 1. Эквивалентная скорость ────────────────────────────────────────────────
U_eq = compute_U_eq(omega_bit, R_bit, R_cone, R)
print(f"U_eq = {U_eq:.4f} м/с")

F_ext = F_bit_radial
print(f"F_ext = {F_ext:.0f} Н (k_load={k_load} × WOB/3 = {k_load}×{WOB/3:.0f})")

# ── 1a. Предварительный расчёт F_max при eps=0.97 ────────────────────────────
print("\n=== Грузоподъёмность при ε = 0.97 (расчётный предел) ===")
EPS_MAX = 0.97

H_test = H_smooth(EPS_MAX)
P_test, _, _ = solve_bit(H_test)
F_max_smooth = compute_load_bit(P_test, phi_1D, Z_1D, U_eq)

F_max_configs = {}
F_max_best_name = "Гладкий"
F_max_best_val  = F_max_smooth
for name, cfg in TEXTURE_CONFIGS.items():
    H_test = H_textured(EPS_MAX, cfg)
    P_test, _, _ = solve_bit(H_test)
    F_max_configs[name] = compute_load_bit(P_test, phi_1D, Z_1D, U_eq)
    if F_max_configs[name] > F_max_best_val:
        F_max_best_val = F_max_configs[name]
        F_max_best_name = name

# Вывод таблицы F_max
max_mark = lambda n: " <- максимум" if n == F_max_best_name else ""
print(f"  Гладкий:  F(ε = 0.97) = {F_max_smooth:.0f} Н{max_mark('Гладкий')}")
for name in TEXTURE_CONFIGS:
    print(f"  {name}:       F(ε = 0.97) = {F_max_configs[name]:.0f} Н{max_mark(name)}")

# Научное наблюдение
F_wob3 = WOB / 3
print(f"\n  Примечание: при нагрузке WOB/3 = {F_wob3:.0f} Н чисто гидродинамическая")
print(f"  постановка недостаточна (F(ε = 0.97),textured ~ {F_max_best_val:.0f} Н, что составляет")
print(f"  около {F_max_best_val/F_wob3*100:.0f}% от грубой оценки WOB/3 = {F_wob3/1e3:.0f} кН).")
print(f"  Поэтому расчёт рабочей точки ведётся не по WOB/3, а по сниженной")
print(f"  нагрузке на цапфу: F_journal = k_load × WOB/3 = {F_ext:.0f} Н.")
print(f"  Это физически ожидаемый результат для низкоскоростной опоры")
print(f"  шарошечного долота (U_eq ~ {U_eq:.2f} м/с против ~10 м/с у насоса).")

F_max_global = max(F_max_smooth, *F_max_configs.values())
if F_max_global < F_ext:
    print(f"\n  ⚠ F_ext={F_ext:.0f} Н > F(ε = 0.97)={F_max_global:.0f} Н — ")
    print(f"    sweep будет адаптирован к диапазону [0, {F_max_global:.0f}] Н.")

# ── 2. Рабочие точки: гладкий + T1/T2/T3 ────────────────────────────────────
print("\n--- Поиск рабочих точек ---")
results = {}

# Гладкий
try:
    eps_s, P_s, H_s = find_operating_point(F_ext, texture_cfg=None)
    res_s = full_postproc(eps_s, P_s, H_s, phi_1D, Z_1D, d_phi, U_eq, label="Гладкий")
    res_s["P"] = P_s
    res_s["H"] = H_s
    results["Гладкий"] = res_s
    print(f"Гладкий: ε={eps_s:.4f}, F={res_s['F']:.0f} Н, "
          f"h_min={res_s['h_min_um']:.2f} мкм, λ={res_s['lam']:.2f}")
except ValueError as e:
    print(f"[Гладкий] {e}")

# Текстурированные
for name, cfg in TEXTURE_CONFIGS.items():
    try:
        eps_t, P_t, H_t = find_operating_point(F_ext, texture_cfg=cfg)
        res_t = full_postproc(eps_t, P_t, H_t, phi_1D, Z_1D, d_phi, U_eq, label=name)
        res_t["P"] = P_t
        res_t["H"] = H_t
        results[name] = res_t
        print(f"{name}: ε={eps_t:.4f}, F={res_t['F']:.0f} Н, "
              f"h_min={res_t['h_min_um']:.2f} мкм, λ={res_t['lam']:.2f}")
    except ValueError as e:
        print(f"[{name}] {e}")
        continue

# ── 3. Сводная таблица ───────────────────────────────────────────────────────
if results:
    print(f"\n=== Таблица 6.x: результаты при F_ext = {F_ext:.0f} Н ===")
    print_results_table(results)
    print("  Примечание: μ — гидродинамическая составляющая. "
          "В смешанном режиме реальное трение узла выше за счёт "
          "контактного взаимодействия шероховатостей.")

# ── 4. Коэффициенты улучшения ────────────────────────────────────────────────
gains = {}
if "Гладкий" in results:
    r_s = results["Гладкий"]
    for name in TEXTURE_CONFIGS:
        if name not in results:
            continue
        r_t = results[name]
        gains[name] = {
            "G_hmin":   r_t["h_min_um"] / r_s["h_min_um"],
            "G_lambda": r_t["lam"]      / r_s["lam"],
            "G_PV":     r_s["PV"]       / r_t["PV"] if r_t["PV"] > 0 else float('inf'),
            "G_wear":   r_s["I_wear"]   / r_t["I_wear"] if r_t["I_wear"] > 0 else float('inf'),
        }
        g = gains[name]
        print(f"{name}: G_hmin={g['G_hmin']:.3f}  G_λ={g['G_lambda']:.3f}  "
              f"G_PV={g['G_PV']:.3f}  G_wear={g['G_wear']:.3f}")

# ── 5. Графики ───────────────────────────────────────────────────────────────

STYLES = {
    "Гладкий": ("b-",  "o", "Гладкий"),
    "T1":      ("r--", "s", "T1"),
    "T2":      ("g-.", "^", "T2"),
    "T3":      ("m:",  "D", "T3"),
}

# --- fig_operating_points: F(epsilon) кривые + F_ext линия + маркеры ---
print("\n--- Построение F(ε) кривых (20 точек × 4 конфигурации) ---")
eps_scan = np.linspace(0.05, 0.97, 20)

fig, ax = plt.subplots(figsize=(8, 5))

# Scan для каждой конфигурации
scan_configs = [("Гладкий", None)] + [(n, c) for n, c in TEXTURE_CONFIGS.items()]
for cfg_name, cfg in scan_configs:
    F_scan = []
    for i, eps in enumerate(eps_scan):
        H_e = H_smooth(eps) if cfg is None else H_textured(eps, cfg)
        P_e, _, _ = solve_bit(H_e)
        F_scan.append(compute_load_bit(P_e, phi_1D, Z_1D, U_eq))
    ls, mk, lbl = STYLES[cfg_name]
    ax.plot(eps_scan, F_scan, ls, linewidth=1.5, label=lbl)
    print(f"  {cfg_name}: F_max(scan) = {max(F_scan):.0f} Н")

ax.axhline(F_ext, color='k', linewidth=0.8, linestyle='--', label=f'F_ext={F_ext/1e3:.0f} кН')

# Маркеры рабочих точек
for name, r in results.items():
    ls, mk, lbl = STYLES[name]
    ax.plot(r["epsilon"], r["F"], mk, color=ls[0], markersize=10, zorder=5)

ax.set_xlabel('ε')
ax.set_ylabel('F, Н')
ax.legend()
ax.grid(True)
plt.tight_layout()
save('fig_operating_points')

# --- Барчарты (только если есть результаты) ---
if results:
    names_plot = [n for n in results]
    x = np.arange(len(names_plot))

    # fig_lambda_comparison
    lam_vals = [results[n]["lam"] for n in names_plot]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, lam_vals, color=['b', 'r', 'g', 'm'][:len(names_plot)])
    ax.axhline(1.0, color='orange', linewidth=1.2, linestyle='--')
    ax.axhline(3.0, color='green',  linewidth=1.2, linestyle='--')
    # Зоны режимов — подписи слева, между линиями
    ax.axhspan(0.0, 1.0, color='orange', alpha=0.05)
    ax.axhspan(1.0, 3.0, color='yellow', alpha=0.05)
    ax.axhspan(3.0, ax.get_ylim()[1] if max(lam_vals) > 3 else 4.0,
               color='green', alpha=0.05)
    ax.text(-0.45, 0.5, 'граничный', fontsize=8, color='orange', ha='left', va='center', style='italic')
    ax.text(-0.45, 2.0, 'смешанный', fontsize=8, color='goldenrod', ha='left', va='center', style='italic')
    ax.text(-0.45, 3.3, 'гидродин.', fontsize=8, color='green', ha='left', va='center', style='italic')
    ax.set_xticks(x)
    ax.set_xticklabels(names_plot)
    ax.set_ylabel('λ')
    ax.set_ylim(0, max(max(lam_vals) * 1.15, 3.5))
    ax.grid(True, axis='y')
    plt.tight_layout()
    save('fig_lambda_comparison')



# --- 3D поля давления ---
Z_idx = np.argmin(np.abs(Z_1D))


def plot_3d(P, fname):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Phi_mesh, Z_mesh, P, cmap='plasma', linewidth=0, antialiased=False)
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    ax.set_zlabel('P')
    plt.tight_layout()
    save(fname)


if "Гладкий" in results:
    plot_3d(results["Гладкий"]["P"], 'fig_P3D_smooth_bit')
if "T2" in results:
    plot_3d(results["T2"]["P"], 'fig_P3D_T2_bit')

# --- Карты кавитации ---


def plot_cav(P, fname):
    cav = (P <= 0).astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(Phi_mesh, Z_mesh, cav, cmap='Blues', vmin=0, vmax=1)
    ax.set_xlabel('φ, рад')
    ax.set_ylabel('Z')
    plt.tight_layout()
    save(fname)


if "Гладкий" in results:
    plot_cav(results["Гладкий"]["P"], 'fig_cav_smooth_bit')

# ── 6. Sweep B — рабочая зона текстур (T1/T2/T3) ─────────────────────────
F_sweep_B_lo = F_max_smooth * 1.1
F_sweep_B_hi = F_max_best_val * 0.9
N_sweep_B = 11

SWEEP_B_STYLES = {
    "T1": ("r--", "s", "T1"),
    "T2": ("g-.", "^", "T2"),
    "T3": ("m:",  "D", "T3"),
}

if F_sweep_B_hi > F_sweep_B_lo:
    F_ext_sweep_B = np.linspace(F_sweep_B_lo, F_sweep_B_hi, N_sweep_B)
    F_ext_B_kN = F_ext_sweep_B / 1e3

    sweep_B = {name: {"eps": [], "lam": [], "hmin": []} for name in TEXTURE_CONFIGS}

    print(f"\n=== Sweep B: {F_sweep_B_lo:.0f} – {F_sweep_B_hi:.0f} Н "
          f"({N_sweep_B} точек, только T1/T2/T3) ===")
    for i, F_e in enumerate(F_ext_sweep_B):
        print(f"  [{i+1}/{N_sweep_B}] F={F_e:.0f} Н ...", end=" ", flush=True)
        for name, cfg in TEXTURE_CONFIGS.items():
            try:
                eps, _, _ = find_operating_point(F_e, texture_cfg=cfg)
                sweep_B[name]["eps"].append(eps)
                sweep_B[name]["lam"].append(compute_lambda(eps))
                sweep_B[name]["hmin"].append(compute_h_min(eps) * 1e6)
                print(f"{name} ε={eps:.4f}", end=" ")
            except ValueError:
                sweep_B[name]["eps"].append(np.nan)
                sweep_B[name]["lam"].append(np.nan)
                sweep_B[name]["hmin"].append(np.nan)
                print(f"{name}: N/A", end=" ")
        print()

    # fig_sweep_B_epsilon
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in TEXTURE_CONFIGS:
        ls, mk, lbl = SWEEP_B_STYLES[name]
        ax.plot(F_ext_B_kN, sweep_B[name]["eps"], ls, marker=mk,
                linewidth=1.5, markersize=5, label=lbl)
    ax.set_xlabel('F_ext, кН')
    ax.set_ylabel('ε')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save('fig_sweep_B_epsilon')

    # fig_sweep_B_lambda
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in TEXTURE_CONFIGS:
        ls, mk, lbl = SWEEP_B_STYLES[name]
        ax.plot(F_ext_B_kN, sweep_B[name]["lam"], ls, marker=mk,
                linewidth=1.5, markersize=5, label=lbl)
    ax.axhline(1.0, color='orange', linewidth=0.8, linestyle='--', label='λ = 1')
    ax.axhline(3.0, color='green',  linewidth=0.8, linestyle='--', label='λ = 3')
    ax.set_xlabel('F_ext, кН')
    ax.set_ylabel('λ')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save('fig_sweep_B_lambda')

    # fig_sweep_B_hmin
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in TEXTURE_CONFIGS:
        ls, mk, lbl = SWEEP_B_STYLES[name]
        ax.plot(F_ext_B_kN, sweep_B[name]["hmin"], ls, marker=mk,
                linewidth=1.5, markersize=5, label=lbl)
    ax.set_xlabel('F_ext, кН')
    ax.set_ylabel('h_min, мкм')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    save('fig_sweep_B_hmin')
else:
    print(f"\n⚠ Sweep B пропущен: F_hi={F_sweep_B_hi:.0f} <= F_lo={F_sweep_B_lo:.0f}")

# ── 7. Коэффициенты улучшения при F_common ─────────────────────────────────
F_common = F_max_smooth * 0.85

print(f"\n=== Коэффициенты улучшения при F = {F_common:.0f} Н (общая достижимая нагрузка) ===")

results_common = {}
for cfg_name, cfg in [("Гладкий", None)] + list(TEXTURE_CONFIGS.items()):
    try:
        eps_c, P_c, H_c = find_operating_point(F_common, texture_cfg=cfg)
        results_common[cfg_name] = full_postproc(
            eps_c, P_c, H_c, phi_1D, Z_1D, d_phi, U_eq, label=cfg_name)
    except ValueError:
        pass

gains_common = {}
if "Гладкий" in results_common:
    r_s = results_common["Гладкий"]
    for name in TEXTURE_CONFIGS:
        if name in results_common:
            r_t = results_common[name]
            gains_common[name] = {
                "G_hmin":   r_t["h_min_um"] / r_s["h_min_um"],
                "G_lambda": r_t["lam"]      / r_s["lam"],
                "G_PV":     r_s["PV"]       / r_t["PV"] if r_t["PV"] > 0 else float('inf'),
                "G_wear":   r_s["I_wear"]   / r_t["I_wear"] if r_t["I_wear"] > 0 else float('inf'),
            }
            g = gains_common[name]
            print(f"  {name}: G_hmin={g['G_hmin']:.3f}  G_λ={g['G_lambda']:.3f}  "
                  f"G_PV={g['G_PV']:.3f}  G_wear={g['G_wear']:.3f}")

# fig_gains_common
if gains_common:
    g_names  = list(gains_common.keys())
    G_keys   = ["G_hmin", "G_lambda", "G_PV", "G_wear"]
    G_labels = ["$G_{h_{min}}$", "$G_{\\lambda}$", "$G_{PV}$", "$G_{wear}$"]
    x_g = np.arange(len(g_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (gk, gl) in enumerate(zip(G_keys, G_labels)):
        vals = [gains_common[n][gk] for n in g_names]
        ax.bar(x_g + i * width, vals, width, label=gl)
    ax.axhline(1.0, color='k', linewidth=0.8, linestyle='--')
    ax.set_xticks(x_g + width * 1.5)
    ax.set_xticklabels(g_names)
    ax.set_ylabel('G')
    ax.legend()
    ax.grid(True, axis='y')
    plt.tight_layout()
    save('fig_gains_common')

print("\nГотово. Графики сохранены в plots/")
