import numpy as np


def compute_U_eq(omega_bit, R_bit, R_cone, R_journal):
    """
    Эквивалентная скорость скольжения в журнальном подшипнике шарошки.

    Физика: цапфа жёстко связана с корпусом долота (omega_bit).
    Шарошка перекатывается по забою с абсолютной угловой скоростью
    ≈ omega_bit × R_bit / R_cone.
    Относительная скорость в подшипнике:
        omega_rel = omega_bit × (R_bit / R_cone - 1)
        U_eq = omega_rel × R_journal

    При n=80 об/мин, D_bit=215 мм, R_cone=72 мм, R_journal=30 мм
    результат: U_eq ≈ 0.124 м/с (в ~90 раз меньше насоса главы 5).
    """
    omega_rel = omega_bit * (R_bit / R_cone - 1.0)
    return abs(omega_rel) * R_journal
