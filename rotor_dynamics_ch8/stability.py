import numpy as np


def compute_stability(coeffs, m):
    K = np.array([[coeffs['Kxx'], coeffs['Kxy']],
                  [coeffs['Kyx'], coeffs['Kyy']]])
    C = np.array([[coeffs['Cxx'], coeffs['Cxy']],
                  [coeffs['Cyx'], coeffs['Cyy']]])
    Minv = np.eye(2) / m
    A = np.block([[np.zeros((2, 2)),  np.eye(2)],
                  [-Minv @ K,        -Minv @ C]])
    eigs   = np.linalg.eigvals(A)
    Re_max = float(np.max(np.real(eigs)))
    stable = Re_max < 0
    print(f"    Re_max={Re_max:.3f}  stable={stable}")
    return dict(Re_max=Re_max, stable=stable, eigenvalues=eigs)
