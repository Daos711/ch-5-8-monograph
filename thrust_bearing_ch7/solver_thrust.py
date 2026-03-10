import numpy as np


def solve_reynolds_thrust(H, mu, omega, r_1D, theta_1D,
                          SOR_W=1.5, tol=1e-5, max_iter=50000,
                          check_every=500, P_init=None):
    """
    SOR-решение уравнения Рейнольдса в цилиндрических координатах.

    Уравнение:
      d/dr[r * h^3 * dp/dr] + d/dtheta[h^3/r * dp/dtheta] = 6*mu*omega*r * dh/dtheta

    P_init — начальное приближение (None -> нули).
    Возвращает: P (N_r x N_theta), converged (bool), n_iter (int)
    """
    N_r = len(r_1D)
    N_theta = len(theta_1D)
    d_r = r_1D[1] - r_1D[0]
    d_theta = theta_1D[1] - theta_1D[0]

    if P_init is not None:
        P = P_init.copy()
    else:
        P = np.zeros((N_r, N_theta), dtype=np.float64)

    # Граничные условия: P=0 на всех границах (уже нули)

    converged = False
    n_iter = 0

    for iteration in range(1, max_iter + 1):
        n_iter = iteration

        if iteration % check_every == 0:
            P_old = P.copy()

        for i in range(1, N_r - 1):
            r_i = r_1D[i]
            r_e = r_i + 0.5 * d_r
            r_w = r_i - 0.5 * d_r

            for j in range(1, N_theta - 1):
                h_e = 0.5 * (H[i + 1, j] + H[i, j])
                h_w = 0.5 * (H[i, j]     + H[i - 1, j])
                h_n = 0.5 * (H[i, j + 1] + H[i, j])
                h_s = 0.5 * (H[i, j]     + H[i, j - 1])

                aE = r_e * h_e**3 / d_r**2
                aW = r_w * h_w**3 / d_r**2
                aN = h_n**3 / (r_i * d_theta**2)
                aS = h_s**3 / (r_i * d_theta**2)
                aP = -(aE + aW + aN + aS)

                RHS = 6.0 * mu * omega * r_i * (H[i, j + 1] - H[i, j - 1]) / (2.0 * d_theta)

                p_new = (RHS - aE * P[i + 1, j] - aW * P[i - 1, j]
                             - aN * P[i, j + 1] - aS * P[i, j - 1]) / aP

                P[i, j] = P[i, j] + SOR_W * (p_new - P[i, j])

                # Условие кавитации Рейнольдса
                if P[i, j] < 0.0:
                    P[i, j] = 0.0

        # Проверка сходимости
        if iteration % check_every == 0:
            diff = np.max(np.abs(P - P_old))
            scale = np.max(np.abs(P)) + 1e-10
            if diff / scale < tol:
                converged = True
                break

    return P, converged, n_iter
