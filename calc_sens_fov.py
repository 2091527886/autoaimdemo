import math


def calc_px_per_360(dpi: float,
                    camera_sens: float,
                    pf_mouse_sens: float,
                    pf_aim_mult: float,
                    inches_per_360_base: float = 3.0) -> float:
    """
    计算每360°所需鼠标计数（counts/360）。
    基于项目默认：CS=1, PF=1, Aim=1 时约 3 in/360；1200 DPI 对应 3600 counts/360。
    px_per_360 = dpi * inches_per_360_base / (camera_sens * pf_mouse_sens * pf_aim_mult)
    """
    scale = (camera_sens * pf_mouse_sens * pf_aim_mult)
    if scale <= 0:
        raise ValueError("sens scale must be > 0")
    return dpi * inches_per_360_base / scale


def vfov_to_hfov_deg(vfov_deg: float, width: int, height: int) -> float:
    """
    垂直FOV(vfov) -> 水平FOV(hfov)（度）。
    hfov = 2 * atan( tan(vfov/2) * aspect ), aspect = width / height
    """
    aspect = width / float(height)
    vfov_rad = math.radians(vfov_deg)
    hfov_rad = 2.0 * math.atan(math.tan(vfov_rad / 2.0) * aspect)
    return math.degrees(hfov_rad)


if __name__ == "__main__":
    # 默认使用你提供的参数
    width, height = 1920, 1200
    dpi = 1200.0
    vfov_deg = 60.0
    camera_sens = 1.0
    pf_mouse_sens = 1.0
    pf_aim_mult = 1.0

    px360 = calc_px_per_360(dpi, camera_sens, pf_mouse_sens, pf_aim_mult)
    hfov = vfov_to_hfov_deg(vfov_deg, width, height)

    print(f"px_per_360={px360:.6f}")
    print(f"hfov_deg={hfov:.6f}")
