import numpy as np
from scipy.integrate import solve_ivp
from params_dynamic import c


def integrate_orbit(coeffs, m, m_unb, e_unb, omega, x0, y0):
    """
    ОДУ для малых возмущений ξ = x - x0, η = y - y0:

      m*ξ'' + Cxx*ξ' + Cxy*η' + Kxx*ξ + Kxy*η = m_unb*e_unb*ω²*cos(ωt)
      m*η'' + Cyx*ξ' + Cyy*η' + Kyx*ξ + Kyy*η = m_unb*e_unb*ω²*sin(ωt)

    НУ: ξ(0) = 0.01*c, η(0) = 0, ξ'(0) = 0, η'(0) = 0
    """
    Kxx, Kxy = coeffs['Kxx'], coeffs['Kxy']
    Kyx, Kyy = coeffs['Kyx'], coeffs['Kyy']
    Cxx, Cxy = coeffs['Cxx'], coeffs['Cxy']
    Cyx, Cyy = coeffs['Cyx'], coeffs['Cyy']

    F0 = m_unb * e_unb * omega ** 2

    def rhs(t, state):
        xi, eta, xi_d, eta_d = state
        fx = F0 * np.cos(omega * t)
        fy = F0 * np.sin(omega * t)
        xi_dd  = (fx - Cxx * xi_d - Cxy * eta_d - Kxx * xi - Kxy * eta) / m
        eta_dd = (fy - Cyx * xi_d - Cyy * eta_d - Kyx * xi - Kyy * eta) / m
        return [xi_d, eta_d, xi_dd, eta_dd]

    t_end = 20 * 2 * np.pi / omega
    y0_vec = [0.01 * c, 0.0, 0.0, 0.0]

    sol = solve_ivp(rhs, [0, t_end], y0_vec,
                    method='RK45', max_step=2 * np.pi / omega / 50,
                    rtol=1e-8, atol=1e-12)

    t_arr   = sol.t
    xi_arr  = sol.y[0]
    eta_arr = sol.y[1]
    x_total = x0 + xi_arr
    y_total = y0 + eta_arr

    return t_arr, xi_arr, eta_arr, x_total, y_total
