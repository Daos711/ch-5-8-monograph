#!/usr/bin/env python3
"""
Верификационный скрипт для главы 8.
Проверяет чувствительность 8-коэффициентной модели к:
  1) шагу конечных разностей dx, dv
  2) разрешению сетки N_phi, N_Z

Запуск:  cd rotor_dynamics_ch8 && python verify_ch8.py
"""
import sys
import importlib
import io
import math
import numpy as np

# ---------- Инициализация ----------
import params_dynamic as P

# Запомнить исходные значения для восстановления
_ORIG_dx = P.dx
_ORIG_dv = P.dv
_ORIG_N_phi = P.N_phi
_ORIG_N_Z = P.N_Z

COEFF_NAMES = ["Kxx", "Kxy", "Kyx", "Kyy", "Cxx", "Cxy", "Cyx", "Cyy"]

# ---------- Механизм смены параметров ----------

def run_with_params(eps, variant, dx_val, dv_val, N_phi_val, N_Z_val):
    """Запуск расчёта с переопределёнными параметрами через monkeypatch + reload."""
    # Переопределить параметры
    P.dx = dx_val
    P.dv = dv_val
    P.N_phi = N_phi_val
    P.N_Z = N_Z_val

    # Перезагрузить зависимые модули в правильном порядке
    import geometry_dynamic, solver_static, solver_dynamic
    import forces, operating_point, coefficients, stability
    for mod in [geometry_dynamic, solver_static, solver_dynamic,
                forces, operating_point, coefficients, stability]:
        importlib.reload(mod)

    # Импорт после reload
    from coefficients import compute_8coeffs
    from stability import compute_stability

    textured = (variant == "T2")
    cfg = P.TEXTURE_CONFIG if textured else None

    # Захват stdout для обнаружения WARNING от внутренних модулей
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    try:
        coeffs = compute_8coeffs(eps, textured=textured, cfg=cfg)
    finally:
        sys.stdout = old_stdout
    warnings_text = captured.getvalue()

    stab = compute_stability(coeffs, P.m_rotor)

    # Проверка надёжности результата
    has_nan_inf = False
    for name in COEFF_NAMES:
        v = coeffs[name]
        if not np.isfinite(v):
            has_nan_inf = True
            break
    if not np.isfinite(stab["Re_max"]):
        has_nan_inf = True

    has_internal_warning = "WARNING" in warnings_text

    unreliable = has_nan_inf or has_internal_warning

    return {**coeffs, "Re_max": stab["Re_max"], "stable": stab["stable"],
            "unreliable": unreliable, "has_nan_inf": has_nan_inf,
            "internal_warnings": warnings_text.strip()}


def restore_params():
    """Восстановить исходные параметры."""
    P.dx = _ORIG_dx
    P.dv = _ORIG_dv
    P.N_phi = _ORIG_N_phi
    P.N_Z = _ORIG_N_Z


# ---------- Вычисление отклонений ----------

def pct_dev(a, b):
    """Процентное отклонение B относительно A."""
    if a == 0 and b == 0:
        return 0.0
    denom = max(abs(a), abs(b))
    if denom == 0:
        return 0.0
    return 100.0 * (b - a) / denom


def sign_changed(a, b):
    """Проверка смены знака."""
    if a == 0 or b == 0:
        return False
    return (a > 0) != (b > 0)


# ---------- Форматированный вывод ----------

def print_comparison(label, res_a, res_b, threshold_pct):
    """Вывод таблицы сравнения двух прогонов. Возвращает список проблем."""
    issues = []

    if res_a["unreliable"] or res_b["unreliable"]:
        print(f"  {label}:")
        if res_a["unreliable"]:
            print(f"    прогон_A: UNRELIABLE", end="")
            if res_a["has_nan_inf"]:
                print(" (NaN/inf)", end="")
            if res_a["internal_warnings"]:
                print(f"\n      {res_a['internal_warnings']}", end="")
            print()
        if res_b["unreliable"]:
            print(f"    прогон_B: UNRELIABLE", end="")
            if res_b["has_nan_inf"]:
                print(" (NaN/inf)", end="")
            if res_b["internal_warnings"]:
                print(f"\n      {res_b['internal_warnings']}", end="")
            print()
        issues.append(f"{label}: unreliable результат")
        return issues

    print(f"  {label}:")
    print(f"    {'Коэфф':<8} {'прогон_A':>14} {'прогон_B':>14} {'отклонение_%':>14}")

    for name in COEFF_NAMES:
        va = res_a[name]
        vb = res_b[name]
        dev = pct_dev(va, vb)
        flag = ""

        if not np.isfinite(va) or not np.isfinite(vb):
            flag = " WARNING(NaN/inf)"
            issues.append(f"{label}: {name} NaN/inf")
        elif sign_changed(va, vb):
            flag = " WARNING(знак)"
            issues.append(f"{label}: {name} смена знака")
        elif abs(dev) > threshold_pct:
            flag = " WARNING"
            issues.append(f"{label}: {name} отклонение {dev:+.1f}%")

        print(f"    {name:<8} {va:>14.3e} {vb:>14.3e} {dev:>+13.1f}%{flag}")

    # Re_max
    re_a = res_a["Re_max"]
    re_b = res_b["Re_max"]
    abs_diff = re_b - re_a
    dev_re = pct_dev(re_a, re_b)
    flag_re = ""
    if not np.isfinite(re_a) or not np.isfinite(re_b):
        flag_re = " WARNING(NaN/inf)"
        issues.append(f"{label}: Re_max NaN/inf")
    print(f"    {'Re_max':<8} {re_a:>14.1f} {re_b:>14.1f}"
          f"     abs_diff={abs_diff:+.1f}{flag_re}")

    return issues


# ---------- Главный блок ----------

def main():
    all_issues_check1 = []
    all_issues_check2 = []

    # Значения параметров
    dx_A = 0.005 * P.c
    dv_A = 0.005 * P.U
    dx_B = 0.0025 * P.c
    dv_B = 0.0025 * P.U

    N_phi_A, N_Z_A = 360, 120
    N_phi_B, N_Z_B = 480, 160

    cases_mandatory = [(0.6, "smooth"), (0.8, "smooth")]
    cases_extra = [(0.8, "T2"), (0.6, "T2")]

    cases = cases_mandatory + cases_extra

    # ===== ПРОВЕРКА 1: чувствительность к dx/dv =====
    print("=" * 60)
    print("=== ПРОВЕРКА 1: чувствительность к dx/dv ===")
    print(f"    прогон_A: dx=0.005*c, dv=0.005*U")
    print(f"    прогон_B: dx=0.0025*c, dv=0.0025*U")
    print(f"    Сетка: N_phi={N_phi_A}, N_Z={N_Z_A}")
    print("=" * 60)

    for eps, variant in cases:
        label = f"eps={eps}, {variant}"
        print(f"\n  --- {label} ---")
        print(f"  Прогон A...")
        res_a = run_with_params(eps, variant, dx_A, dv_A, N_phi_A, N_Z_A)
        print(f"  Прогон B...")
        res_b = run_with_params(eps, variant, dx_B, dv_B, N_phi_A, N_Z_A)
        issues = print_comparison(label, res_a, res_b, threshold_pct=20.0)
        all_issues_check1.extend(issues)

    # ===== ПРОВЕРКА 2: чувствительность к сетке =====
    print("\n" + "=" * 60)
    print("=== ПРОВЕРКА 2: чувствительность к сетке ===")
    print(f"    Сетка_A: N_phi={N_phi_A}, N_Z={N_Z_A}")
    print(f"    Сетка_B: N_phi={N_phi_B}, N_Z={N_Z_B}")
    print(f"    dx=0.005*c, dv=0.005*U")
    print("=" * 60)

    for eps, variant in cases:
        label = f"eps={eps}, {variant}"
        print(f"\n  --- {label} ---")
        print(f"  Прогон A (сетка {N_phi_A}x{N_Z_A})...")
        res_a = run_with_params(eps, variant, dx_A, dv_A, N_phi_A, N_Z_A)
        print(f"  Прогон B (сетка {N_phi_B}x{N_Z_B})...")
        res_b = run_with_params(eps, variant, dx_A, dv_A, N_phi_B, N_Z_B)
        issues = print_comparison(label, res_a, res_b, threshold_pct=10.0)
        all_issues_check2.extend(issues)

    # ===== ИТОГ =====
    print("\n" + "=" * 60)
    print("=== ИТОГ ===")
    print("=" * 60)

    if all_issues_check1:
        print("Проверка 1 (dx/dv): WARNING")
        print("  Проблемные случаи:")
        for iss in all_issues_check1:
            print(f"    - {iss}")
    else:
        print("Проверка 1 (dx/dv): PASS")

    if all_issues_check2:
        print("Проверка 2 (сетка): WARNING")
        print("  Проблемные случаи:")
        for iss in all_issues_check2:
            print(f"    - {iss}")
    else:
        print("Проверка 2 (сетка): PASS")

    print()

    # Восстановить исходные параметры
    restore_params()
    # Перезагрузить модули с исходными параметрами
    import geometry_dynamic, solver_static, solver_dynamic
    import forces, operating_point, coefficients, stability
    for mod in [geometry_dynamic, solver_static, solver_dynamic,
                forces, operating_point, coefficients, stability]:
        importlib.reload(mod)

    print("Параметры восстановлены к исходным значениям.")


if __name__ == "__main__":
    main()
