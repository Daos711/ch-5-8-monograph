import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from params_dynamic import (epsilon_values, epsilon_orbit, m_rotor,
                             m_unb, e_unb, omega, TEXTURE_CONFIG, c)
from operating_point import get_operating_point
from coefficients import compute_8coeffs
from stability import compute_stability
from rotor_ode import integrate_orbit

os.makedirs("plots", exist_ok=True)

CACHE_PATH = "plots/cache.pkl"
PLOT_ONLY = "--plot-only" in sys.argv

# ── Запятая вместо точки на осях ──
_comma_fmt = mticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))


def _apply_comma(ax, axes3d=False):
    """Применить формат с запятой ко всем числовым осям."""
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))
    if axes3d and hasattr(ax, 'zaxis'):
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}".replace(".", ",")))


# ═════════════════════════════════════════════════════════════════════════════
#  РАСЧЁТНАЯ ЧАСТЬ  (пропускается при --plot-only)
# ═════════════════════════════════════════════════════════════════════════════

if PLOT_ONLY:
    if not os.path.exists(CACHE_PATH):
        print("Ошибка: кэш не найден. Сначала запустите без --plot-only.")
        sys.exit(1)
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    results_by_eps = cache["results_by_eps"]
    results        = cache["results"]
    orbits         = cache["orbits"]
    print(f"Кэш загружен из {CACHE_PATH}. Строю только графики...")
else:
    results_by_eps = {"smooth": {}, "T2": {}}

    results = {
        "smooth": {"eps": [], "Kxx": [], "Kxy": [], "Kyx": [], "Kyy": [],
                   "Cxx": [], "Cxy": [], "Cyx": [], "Cyy": [],
                   "Re_max": [], "stable": []},
        "T2":     {"eps": [], "Kxx": [], "Kxy": [], "Kyx": [], "Kyy": [],
                   "Cxx": [], "Cxy": [], "Cyx": [], "Cyy": [],
                   "Re_max": [], "stable": []},
    }

    orbits = {"smooth": {}, "T2": {}}

    # ============================================================
    # Режим 1 — коэффициенты vs epsilon
    # ============================================================
    for variant in ["smooth", "T2"]:
        textured = (variant == "T2")
        print(f"\n=== Коэффициенты ({variant}) ===")
        for eps in epsilon_values:
            coeffs = compute_8coeffs(eps, textured=textured,
                                     cfg=TEXTURE_CONFIG if textured else None)
            stab = compute_stability(coeffs, m_rotor)
            results_by_eps[variant][eps] = {**coeffs, **stab}

            for key in ["Kxx", "Kxy", "Kyx", "Kyy",
                         "Cxx", "Cxy", "Cyx", "Cyy"]:
                results[variant][key].append(coeffs[key])
            results[variant]["eps"].append(eps)
            results[variant]["Re_max"].append(stab["Re_max"])
            results[variant]["stable"].append(stab["stable"])

    # ============================================================
    # Таблица коэффициентов
    # ============================================================
    for variant in ["smooth", "T2"]:
        print(f"\n=== Таблица ({variant}) ===")
        print(f"{'eps':>5s}  {'Kxx':>10s} {'Kxy':>10s} {'Kyx':>10s} {'Kyy':>10s}"
              f"  {'Cxx':>9s} {'Cxy':>9s} {'Cyx':>9s} {'Cyy':>9s}"
              f"  {'Re_max':>8s} {'stable':>6s}")
        r = results[variant]
        for i in range(len(r["eps"])):
            print(f"{r['eps'][i]:5.2f}"
                  f"  {r['Kxx'][i]:10.0f} {r['Kxy'][i]:10.0f}"
                  f" {r['Kyx'][i]:10.0f} {r['Kyy'][i]:10.0f}"
                  f"  {r['Cxx'][i]:9.2f} {r['Cxy'][i]:9.2f}"
                  f" {r['Cyx'][i]:9.2f} {r['Cyy'][i]:9.2f}"
                  f"  {r['Re_max'][i]:8.2f} {str(r['stable'][i]):>6s}")

    # ============================================================
    # Режим 2 — орбита при epsilon_orbit
    # ============================================================
    print(f"\n=== Орбиты при eps={epsilon_orbit} ===")
    for variant in ["smooth", "T2"]:
        coeffs = results_by_eps[variant][epsilon_orbit]
        x0, y0 = get_operating_point(epsilon_orbit)
        t, xi, eta, x_tot, y_tot = integrate_orbit(
            coeffs, m_rotor, m_unb, e_unb, omega, x0, y0)
        orbits[variant] = dict(t=t, xi=xi, eta=eta,
                               x_total=x_tot, y_total=y_tot)
        print(f"  {variant}: max|xi|={np.max(np.abs(xi))*1e6:.2f} мкм, "
              f"max|eta|={np.max(np.abs(eta))*1e6:.2f} мкм")

    # ── Сохранение кэша ──
    cache = dict(results_by_eps=results_by_eps, results=results, orbits=orbits)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nКэш сохранён в {CACHE_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
#  ГРАФИКИ  (всегда выполняется)
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Построение графиков ===")

LABELS = {"smooth": "Гладкий", "T2": "T2"}

# Уникальный стиль для каждой комбинации (вариант, коэффициент)
_STYLES = {
    ("smooth", "Kxx"): dict(color="blue",    linestyle="-",  marker="o"),
    ("smooth", "Kyy"): dict(color="blue",    linestyle="--", marker="s"),
    ("smooth", "Kxy"): dict(color="blue",    linestyle="-",  marker="o"),
    ("smooth", "Kyx"): dict(color="blue",    linestyle="--", marker="s"),
    ("smooth", "Cxx"): dict(color="blue",    linestyle="-",  marker="o"),
    ("smooth", "Cyy"): dict(color="blue",    linestyle="--", marker="s"),
    ("smooth", "Cxy"): dict(color="blue",    linestyle="-",  marker="o"),
    ("smooth", "Cyx"): dict(color="blue",    linestyle="--", marker="s"),
    ("T2",     "Kxx"): dict(color="green",   linestyle="-",  marker="^"),
    ("T2",     "Kyy"): dict(color="green",   linestyle="--", marker="D"),
    ("T2",     "Kxy"): dict(color="green",   linestyle="-",  marker="^"),
    ("T2",     "Kyx"): dict(color="green",   linestyle="--", marker="D"),
    ("T2",     "Cxx"): dict(color="green",   linestyle="-",  marker="^"),
    ("T2",     "Cyy"): dict(color="green",   linestyle="--", marker="D"),
    ("T2",     "Cxy"): dict(color="green",   linestyle="-",  marker="^"),
    ("T2",     "Cyx"): dict(color="green",   linestyle="--", marker="D"),
}


def plot_coeffs_vs_eps(keys, ylabel, fname):
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in ["smooth", "T2"]:
        eps_arr = results[variant]["eps"]
        for key in keys:
            vals = results[variant][key]
            st = _STYLES[(variant, key)]
            ax.plot(eps_arr, vals,
                    color=st["color"], linestyle=st["linestyle"],
                    marker=st["marker"], markersize=5,
                    label=f"{LABELS[variant]} {key}")
    ax.set_xlabel('ε')
    ax.set_ylabel(ylabel)
    _apply_comma(ax)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"plots/{fname}", dpi=300)
    plt.close(fig)
    print(f"  -> {fname}")


# fig 1: Kxx, Kyy vs eps
plot_coeffs_vs_eps(["Kxx", "Kyy"], 'K, Н/м', 'fig_8_1.png')

# fig 2: Kxy, Kyx vs eps
plot_coeffs_vs_eps(["Kxy", "Kyx"], 'K, Н/м', 'fig_8_2.png')

# fig 3: Cxx, Cyy vs eps
plot_coeffs_vs_eps(["Cxx", "Cyy"], 'C, Н·с/м', 'fig_8_3.png')

# fig 4: Cxy, Cyx vs eps
plot_coeffs_vs_eps(["Cxy", "Cyx"], 'C, Н·с/м', 'fig_8_3a.png')

# fig 5: Re_max vs eps
fig, ax = plt.subplots(figsize=(8, 5))
for variant in ["smooth", "T2"]:
    ax.plot(results[variant]["eps"], results[variant]["Re_max"],
            'o-', color=("blue" if variant == "smooth" else "green"), label=LABELS[variant])
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('ε')
ax.set_ylabel('Re_max')
_apply_comma(ax)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("plots/fig_8_4.png", dpi=300)
plt.close(fig)
print("  -> fig_8_4.png")

# fig 6: orbit x(t), y(t)
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for variant in ["smooth", "T2"]:
    orb = orbits[variant]
    t_ms = orb["t"] * 1e3
    axes[0].plot(t_ms, orb["x_total"] * 1e6,
                 color=("blue" if variant == "smooth" else "green"), label=LABELS[variant])
    axes[1].plot(t_ms, orb["y_total"] * 1e6,
                 color=("blue" if variant == "smooth" else "green"), label=LABELS[variant])
axes[0].set_ylabel('x, мкм')
axes[1].set_ylabel('y, мкм')
axes[1].set_xlabel('t, мс')
for ax in axes:
    _apply_comma(ax)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("plots/fig_8_5.png", dpi=300)
plt.close(fig)
print("  -> fig_8_5.png")

# fig 7: фазовый портрет y vs x
fig, ax = plt.subplots(figsize=(7, 3.5))
for variant in ["smooth", "T2"]:
    orb = orbits[variant]
    ax.plot(orb["x_total"] * 1e6, orb["y_total"] * 1e6,
            color=("blue" if variant == "smooth" else "green"), label=LABELS[variant], linewidth=0.7)
ax.set_xlabel('x, мкм')
ax.set_ylabel('y, мкм')
_apply_comma(ax)
ax.set_aspect('equal')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig("plots/fig_8_6.png", dpi=300)
plt.close(fig)
print("  -> fig_8_6.png")

print("\nГотово.")
