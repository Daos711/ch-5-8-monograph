import numpy as np
from params_dynamic import dx, dv, c, TEXTURE_CONFIG
from geometry_dynamic import H_base, H_textured
from solver_static import solve_static
from solver_dynamic import solve_dynamic
from forces import compute_Fx, compute_Fy
from operating_point import get_operating_point


def compute_8coeffs(epsilon, textured=False, cfg=None):
    x0, y0 = get_operating_point(epsilon)

    def H(x, y):
        H_ = H_textured(x, y, cfg) if textured else H_base(x, y)
        if np.min(H_) <= 0:
            print(f"  WARNING: min(H)={np.min(H_):.4f} <= 0 при x={x:.2e}, y={y:.2e}")
            return None
        return H_

    def safe_static(x, y):
        H_ = H(x, y)
        if H_ is None:
            return None, False, 0
        return solve_static(H_)

    def safe_dynamic(x, y, xdot, ydot):
        H_ = H(x, y)
        if H_ is None:
            return None, False, 0
        return solve_dynamic(H_, xdot, ydot)

    nan8 = dict(Kxx=np.nan, Kxy=np.nan, Kyx=np.nan, Kyy=np.nan,
                Cxx=np.nan, Cxy=np.nan, Cyx=np.nan, Cyy=np.nan)

    # Жёсткости (статический solver)
    P_xp, _, _ = safe_static(x0+dx, y0)
    if P_xp is None: return nan8
    P_xm, _, _ = safe_static(x0-dx, y0)
    if P_xm is None: return nan8
    Kxx = -(compute_Fx(P_xp) - compute_Fx(P_xm)) / (2*dx)
    Kyx = -(compute_Fy(P_xp) - compute_Fy(P_xm)) / (2*dx)

    P_yp, _, _ = safe_static(x0, y0+dx)
    if P_yp is None: return nan8
    P_ym, _, _ = safe_static(x0, y0-dx)
    if P_ym is None: return nan8
    Kxy = -(compute_Fx(P_yp) - compute_Fx(P_ym)) / (2*dx)
    Kyy = -(compute_Fy(P_yp) - compute_Fy(P_ym)) / (2*dx)

    # Демпфирование (динамический solver)
    P_vxp, _, _ = safe_dynamic(x0, y0, xdot=+dv, ydot=0)
    if P_vxp is None: return nan8
    P_vxm, _, _ = safe_dynamic(x0, y0, xdot=-dv, ydot=0)
    if P_vxm is None: return nan8
    Cxx = -(compute_Fx(P_vxp) - compute_Fx(P_vxm)) / (2*dv)
    Cyx = -(compute_Fy(P_vxp) - compute_Fy(P_vxm)) / (2*dv)

    P_vyp, _, _ = safe_dynamic(x0, y0, xdot=0, ydot=+dv)
    if P_vyp is None: return nan8
    P_vym, _, _ = safe_dynamic(x0, y0, xdot=0, ydot=-dv)
    if P_vym is None: return nan8
    Cxy = -(compute_Fx(P_vyp) - compute_Fx(P_vym)) / (2*dv)
    Cyy = -(compute_Fy(P_vyp) - compute_Fy(P_vym)) / (2*dv)

    # Вывод без коррекции
    print(f"  eps={epsilon:.2f}: "
          f"Kxx={Kxx:.1f} Kxy={Kxy:.1f} Kyx={Kyx:.1f} Kyy={Kyy:.1f}")
    print(f"           "
          f"Cxx={Cxx:.3f} Cxy={Cxy:.3f} Cyx={Cyx:.3f} Cyy={Cyy:.3f}")

    # Warnings — не исправления
    if Kxx < 0:
        print(f"  WARNING: Kxx={Kxx:.1f} < 0")
    if Cyy < 0:
        print(f"  WARNING: Cyy={Cyy:.3f} < 0")
    asym = abs(Kxy + Kyx) / (abs(Kxy) + abs(Kyx) + 1e-10)
    if asym > 0.3:
        print(f"  WARNING: несимметрия Kxy/Kyx = {asym:.0%}")

    return dict(Kxx=Kxx, Kxy=Kxy, Kyx=Kyx, Kyy=Kyy,
                Cxx=Cxx, Cxy=Cxy, Cyx=Cyx, Cyy=Cyy)
