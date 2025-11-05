import math


def compute_hfov_deg(vfov_deg: float, width: int, height: int) -> float:
    """Compute horizontal FOV (degrees) from vertical FOV and resolution."""
    aspect = width / height
    return math.degrees(2.0 * math.atan(math.tan(math.radians(vfov_deg) / 2.0) * aspect))


def compute_px_per_360(
    camera_sens: float,
    mouse_sens: float,
    aim_mult: float,
    deg_per_count_base: float = 0.1,
) -> float:
    """
    Estimate mouse counts per 360 degrees.
    Assumes deg_per_count scales linearly with the product of sensitivities.
    deg_per_count_base is the base degrees-per-count at sensitivities=1.
    """
    eff = camera_sens * mouse_sens * aim_mult
    deg_per_count = deg_per_count_base * eff
    return 360.0 / max(1e-12, deg_per_count)


if __name__ == "__main__":
    # Your parameters
    width, height = 1920, 1200
    vfov_deg = 80.0
    camera_sens = 1.0
    mouse_sens = 1.0
    aim_mult = 1.0

    hfov = compute_hfov_deg(vfov_deg, width, height)
    px360 = compute_px_per_360(camera_sens, mouse_sens, aim_mult, deg_per_count_base=0.1)

    print(f"hfov_deg = {hfov:.6f}")
    print(f"px_per_360 = {px360:.6f}")
