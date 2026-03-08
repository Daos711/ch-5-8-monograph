import numpy as np
from params_bit import c, sigma_eq


def compute_h_min(epsilon):
    """
    Минимальная толщина плёнки, м.

    Определение: минимальный зазор БАЗОВОЙ ГЛАДКОЙ формы в зоне сужения.
        h_min = c * (1 - epsilon)

    Намеренно НЕ используется H.min() * c, чтобы лунки текстуры
    не искажали оценку — их локальные впадины не являются
    «минимальным зазором» в смысле несущей плёнки.
    """
    return c * (1.0 - epsilon)


def compute_lambda(epsilon):
    """
    Параметр Стрибека: lambda = h_min / sigma_eq

    Классификация (по Stribeck / Hamrock & Dowson):
        lambda < 1          — граничный режим
        1 <= lambda <= 3    — смешанный режим
        lambda > 3          — гидродинамический режим
    """
    return compute_h_min(epsilon) / sigma_eq


def classify_regime(lam):
    if lam < 1.0:
        return "граничный"
    elif lam <= 3.0:
        return "смешанный"
    else:
        return "гидродинамический"
