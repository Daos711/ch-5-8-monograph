from params_dynamic import c


def get_operating_point(epsilon):
    """
    Возвращает рабочую точку (x0, y0) для заданного epsilon = e/c.

    В данной версии модели точка задаётся параметрически,
    а НЕ ищется из уравнения равновесия.
    Направление смещения — вдоль оси x: x0 = epsilon*c, y0 = 0.
    """
    return epsilon * c, 0.0
