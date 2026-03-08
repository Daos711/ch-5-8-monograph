import numpy as np


def compute_PV(F, U_eq, R, L):
    """
    PV-параметр через номинальное давление.

    p_nom = F / (2 * R * L)  — номинальное контактное давление, Па
    PV    = p_nom * U_eq     — Па·м/с

    Используется номинальное p_nom (не среднее по карте давления),
    чтобы метрика не зависела от деталей дискретизации и кавитационной маски.
    """
    p_nom = F / (2 * R * L)
    return p_nom * U_eq


def compute_wear_severity_index(F, U_eq, epsilon, R, L, c):
    """
    Суррогатный индекс интенсивности износа:
        I ~ PV / h_min  (Па/с)

    ВАЖНО: это НЕ строгий закон Арчарда (для него нужны коэффициент
    износа, твёрдость, путь трения). Это сравнительный безразмерный
    индекс для сопоставления гладкой и текстурированной опоры
    в рамках одной главы.
    """
    from mixed_lubrication import compute_h_min
    PV    = compute_PV(F, U_eq, R, L)
    h_min = compute_h_min(epsilon)
    return PV / h_min if h_min > 0 else float('inf')
