"""
verify_ch8.py — верификация численной устойчивости коэффициентов главы 8.

Проверка 1: чувствительность к шагу dx/dv  (порог 20%)
Проверка 2: чувствительность к сетке         (порог 10%)

Запуск:  cd rotor_dynamics_ch8 && python verify_ch8.py
"""

import importlib
import sys
import math
import numpy as np

# Убираем --draft, чтобы params_dynamic загрузился в полном режиме
sys.argv = [a for a in sys.argv if a != "--draft"]

import params_dynamic as P

# Сохраняем исходные значения для восстановления
_ORIG = dict(dx=P.dx, dv=P.dv, N_phi=P.N_phi, N_Z=P.N_Z,
             MAX_ITER=P.MAX_ITER, TOL=P.TOL, CHECK_EVERY=P.CHECK_EVERY,
             SOR_W=P.SOR_W)

COEFF_NAMES = ["Kxx", "Kxy", "Kyx", "Kyy", "Cxx", "Cxy", "Cyx", "Cyy"]


def _restore_params():
    for k, v in _ORIG.items():
        setattr(P, k, v)


def run_with_params(eps, variant, dx_val, dv_val, N_phi_val, N_Z_val):
    """Запуск compute_8coeffs + compute_stability с подменёнными параметрами."""
    P.dx = dx_val
    P.dv = dv_val
    P.N_phi = N_phi_val
    P.N_Z = N_Z_val
    # Для крупных сеток увеличиваем лимит итераций
    P.MAX_ITER = 50000
    P.TOL = 1e-5
    P.CHECK_EVERY = 500
    P.SOR_W = 1.5

    import geometry_dynamic, solver_static, solver_dynamic
    import forces, operating_point, coefficients, stability
    for mod in [geometry_dynamic, solver_static, solver_dynamic,
                forces, operating_point, coefficients, stability]:
        importlib.reload(mod)

    from coefficients import compute_8coeffs
    from stability import compute_stability

    textured = (variant == "T2")
    cfg = P.TEXTURE_CONFIG if textured else None
    coeffs = compute_8coeffs(eps, textured=textured, cfg=cfg)
    stab = compute_stability(coeffs, P.m_rotor)
    return {**coeffs, **stab}


def _has_bad(result):
    """Проверка NaN/inf в результатах."""
    for name in COEFF_NAMES + ["Re_max"]:
        v = result.get(name, None)
        if v is None or not np.isfinite(v):
            return True
    return False


def _pct(a, b):
    """Процентное отклонение b относительно a."""
    if a == 0 and b == 0:
        return 0.0
    denom = max(abs(a), abs(b), 1e-30)
    return abs(b - a) / denom * 100.0


def _sign_changed(a, b):
    return (a > 0 and b < 0) or (a < 0 and b > 0)


def _fmt(v):
    """Форматирование числа."""
    if v is None or not np.isfinite(v):
        return "NaN/inf"
    av = abs(v)
    if av == 0:
        return "0.00"
    if av >= 1e6:
        return f"{v:.2e}"
    if av >= 1:
        return f"{v:.1f}"
    return f"{v:.4f}"


def compare_and_report(label_a, result_a, label_b, result_b, threshold,
                       eps, variant):
    """Сравнивает два прогона, выводит таблицу, возвращает список предупреждений."""
    warnings = []

    if _has_bad(result_a):
        warnings.append(f"eps={eps}, {variant}: прогон {label_a} содержит NaN/inf — unreliable")
    if _has_bad(result_b):
        warnings.append(f"eps={eps}, {variant}: прогон {label_b} содержит NaN/inf — unreliable")

    print(f"\n  eps={eps}, {variant}:")
    print(f"  {'Коэфф':<8} {'прогон_A':>14} {'прогон_B':>14} {'отклонение_%':>14}")

    for name in COEFF_NAMES:
        va = result_a.get(name, float("nan"))
        vb = result_b.get(name, float("nan"))
        if not np.isfinite(va) or not np.isfinite(vb):
            pct_str = "NaN/inf"
            warnings.append(f"eps={eps}, {variant}: {name} NaN/inf")
        else:
            pct = _pct(va, vb)
            pct_str = f"{pct:.1f}%"
            if pct > threshold:
                pct_str += " !"
                warnings.append(
                    f"eps={eps}, {variant}: {name} отклонение {pct:.0f}%"
                )
            if _sign_changed(va, vb):
                pct_str += " SIGN"
                warnings.append(
                    f"eps={eps}, {variant}: {name} смена знака"
                )
        print(f"  {name:<8} {_fmt(va):>14} {_fmt(vb):>14} {pct_str:>14}")

    # Re_max
    ra = result_a.get("Re_max", float("nan"))
    rb = result_b.get("Re_max", float("nan"))
    if np.isfinite(ra) and np.isfinite(rb):
        pct_re = _pct(ra, rb)
        abs_diff = rb - ra
        print(f"  {'Re_max':<8} {_fmt(ra):>14} {_fmt(rb):>14} "
              f"{pct_re:.1f}%  abs_diff={abs_diff:+.3f}")
    else:
        print(f"  {'Re_max':<8} {_fmt(ra):>14} {_fmt(rb):>14} {'NaN/inf':>14}")
        warnings.append(f"eps={eps}, {variant}: Re_max NaN/inf")

    return warnings


def main():
    all_warnings_1 = []
    all_warnings_2 = []

    cases_mandatory = [
        (0.6, "smooth"),
        (0.8, "smooth"),
    ]
    cases_extra = [
        (0.8, "T2"),
    ]

    # =====================================================================
    # ПРОВЕРКА 1: чувствительность к dx/dv (порог 20%)
    # =====================================================================
    print("=" * 70)
    print("=== ПРОВЕРКА 1: чувствительность к dx/dv ===")
    print("=" * 70)

    for eps, variant in cases_mandatory + cases_extra:
        tag = f"eps={eps}, {variant}"
        print(f"\n--- {tag} ---")

        # Прогон A: стандартные шаги
        print(f"  Прогон A (dx=0.005*c, dv=0.005*U) ...")
        res_a = run_with_params(eps, variant,
                                dx_val=0.005 * P.c,
                                dv_val=0.005 * P.U,
                                N_phi_val=360, N_Z_val=120)

        # Прогон B: вдвое меньше
        print(f"  Прогон B (dx=0.0025*c, dv=0.0025*U) ...")
        res_b = run_with_params(eps, variant,
                                dx_val=0.0025 * P.c,
                                dv_val=0.0025 * P.U,
                                N_phi_val=360, N_Z_val=120)

        ws = compare_and_report("A", res_a, "B", res_b, 20.0, eps, variant)
        all_warnings_1.extend(ws)

    # =====================================================================
    # ПРОВЕРКА 2: чувствительность к сетке (порог 10%)
    # =====================================================================
    print("\n" + "=" * 70)
    print("=== ПРОВЕРКА 2: чувствительность к сетке ===")
    print("=" * 70)

    for eps, variant in cases_mandatory + cases_extra:
        tag = f"eps={eps}, {variant}"
        print(f"\n--- {tag} ---")

        # Сетка A: 360×120
        print(f"  Сетка A (360×120) ...")
        res_a = run_with_params(eps, variant,
                                dx_val=0.005 * P.c,
                                dv_val=0.005 * P.U,
                                N_phi_val=360, N_Z_val=120)

        # Сетка B: 480×160
        print(f"  Сетка B (480×160) ...")
        res_b = run_with_params(eps, variant,
                                dx_val=0.005 * P.c,
                                dv_val=0.005 * P.U,
                                N_phi_val=480, N_Z_val=160)

        ws = compare_and_report("A", res_a, "B", res_b, 10.0, eps, variant)
        all_warnings_2.extend(ws)

    # =====================================================================
    # ИТОГ
    # =====================================================================
    print("\n" + "=" * 70)
    print("=== ИТОГ ===")
    print("=" * 70)

    if all_warnings_1:
        print("\nПроверка 1 (dx/dv): WARNING")
        print("  Проблемные случаи:")
        for w in all_warnings_1:
            print(f"  - {w}")
    else:
        print("\nПроверка 1 (dx/dv): PASS")

    if all_warnings_2:
        print("\nПроверка 2 (сетка): WARNING")
        print("  Проблемные случаи:")
        for w in all_warnings_2:
            print(f"  - {w}")
    else:
        print("\nПроверка 2 (сетка): PASS")

    # Восстановление исходных параметров
    _restore_params()
    print("\nПараметры восстановлены к исходным значениям.")
    print("Готово.")


if __name__ == "__main__":
    main()
