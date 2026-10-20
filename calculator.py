#!/usr/bin/env python3
"""
High-altitude balloon helium calculator (1 or 2 balloons)
==========================================================

Method
------
* Free-lift method: the helium must provide a NET lift equal to
  (total dry weight) x (1 + f), where f is the free-lift fraction
  (14 % baseline -> ~3-4 m/s ascent).
* Atmosphere: U.S. Standard Atmosphere 1976 (pressure & temperature),
  densities from the ideal gas law (eq. 4.10 / 4.11).
* Two balloons: the parachute balloon is sized so that, after the carrier
  bursts, the payload descends at the desired rate AT GROUND LEVEL
  (eq. 4.12 - 4.16 evaluated with ground-level air/helium density; the
  helium mass is fixed, so its size at altitude follows from that).  Its own net buoyancy is counted during the ascent,
  so the carrier only has to supply the remaining lift.
* The carrier must burst FIRST while the parachute balloon survives. Burst
  diameters are not known, so the carrier is simply given more helium than
  the parachute balloon: its helium mass is the larger of (a) the helium
  needed for lift and (b) CARRIER_HELIUM_RATIO x the parachute balloon's helium.

Important physics note
----------------------
Net lift of a helium balloon = m_He * (M_air / M_He - 1).  It depends only on
the helium MASS, not on altitude.  So the helium mass (and ground-level volume)
does not change with burst altitude; altitude only changes the balloon's size
in the air and, for two balloons, the parachute-balloon size.

All inputs are in grams / metres / m/s.  All lift, helium-volume and size
results are given for GROUND LEVEL (diameter at altitude is shown as well).
"""

import math
from collections import namedtuple

# ------------------------------------------------------------------ constants
M_AIR = 0.0289652        # kg/mol  molar mass of dry air
M_HE = 0.0040026         # kg/mol  molar mass of helium
R_GAS = 8.31446          # J/(mol K)
G = 9.80665              # m/s^2

CD = 0.47                # drag coefficient of the parachute balloon (sphere)
FREE_LIFT_FRACTION = 0.14   # 0.10 slow (~2 m/s), 0.14 standard (~3-4 m/s), 0.20 fast (~5 m/s)
GROUND_ALTITUDE_M = 1445.0     # launch-site elevation above sea level (set to your site)
CARRIER_HELIUM_RATIO = 1.5  # carrier helium mass >= this x parachute-balloon helium mass

# ---- U.S. Standard Atmosphere 1976 (geopotential layers: base h, base T, lapse K/m)
_R_STAR = 8.31432
_M0 = 0.0289644
_R_EARTH = 6356766.0
_P0 = 101325.0
_LAYERS = [
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.0010),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.0020),
]
_H_MAX = 84852.0
MAX_ALTITUDE_M = 50000.0

Atmosphere = namedtuple("Atmosphere", "p t rho_air rho_he")


def atmosphere(altitude_m):
    """Pressure, temperature and air/helium density at a geometric altitude."""
    h = _R_EARTH * altitude_m / (_R_EARTH + altitude_m)   # geopotential height
    p_base = _P0
    p = t = None
    for i, (hb, tb, lapse) in enumerate(_LAYERS):
        h_top = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else _H_MAX
        dh = min(h, h_top) - hb
        if lapse == 0.0:
            t = tb
            p = p_base * math.exp(-G * _M0 * dh / (_R_STAR * tb))
        else:
            t = tb + lapse * dh
            p = p_base * (tb / t) ** (G * _M0 / (_R_STAR * lapse))
        if h <= h_top:
            break
        p_base = p
    rho_air = p * M_AIR / (R_GAS * t)    # eq. 4.10
    rho_he = p * M_HE / (R_GAS * t)      # eq. 4.11
    return Atmosphere(p, t, rho_air, rho_he)


# -------------------------------------------------------------------- helpers
def sphere_diameter(volume):
    return (6.0 * volume / math.pi) ** (1.0 / 3.0)


def helium_mass_for_lift(lift_kg):
    """Helium mass (kg) whose net lift equals lift_kg (independent of altitude)."""
    return lift_kg * M_HE / (M_AIR - M_HE)


def balloon_from_helium_mass(m_he, ground, alt):
    v0 = m_he / ground.rho_he      # helium volume at ground level
    vh = m_he / alt.rho_he         # same helium at burst altitude
    return {
        "m_he_kg": m_he,
        "v_ground_m3": v0,
        "v_alt_m3": vh,
        "d_ground_m": sphere_diameter(v0),
        "d_alt_m": sphere_diameter(vh),
    }


def solve_parachute_diameter(m_payload, m_chute, alt, v_desc):
    """
    Solve eq. 4.12-4.15 together.  Meff depends on the balloon volume, which
    depends on the area, which depends on Meff, so solve for d numerically:
        0.5*rho*CD*(pi d^2/4)*v^2 = g*(Mp + Md - k*pi d^3/6),  k = rho_air - rho_he
    """
    k = alt.rho_air - alt.rho_he

    def residual(d):
        area = math.pi * d * d / 4.0
        vol = math.pi * d ** 3 / 6.0
        return 0.5 * alt.rho_air * CD * area * v_desc ** 2 - G * (m_payload + m_chute - k * vol)

    lo, hi = 0.0, 1.0
    while residual(hi) < 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- calculations
def calc_single(total_g, alt_m):
    ground = atmosphere(GROUND_ALTITUDE_M)
    alt = atmosphere(alt_m)
    free_g = total_g * FREE_LIFT_FRACTION
    lift_g = total_g + free_g
    balloon = balloon_from_helium_mass(helium_mass_for_lift(lift_g / 1000.0), ground, alt)
    return {"total_g": total_g, "free_g": free_g, "lift_g": lift_g, "balloon": balloon}


def calc_double(train_g, carrier_g, chute_g, alt_m, v_desc):
    ground = atmosphere(GROUND_ALTITUDE_M)
    alt = atmosphere(alt_m)
    mp, mc, md = train_g / 1000.0, carrier_g / 1000.0, chute_g / 1000.0

    # --- parachute balloon sized for the desired descent rate at GROUND level
    d_0 = solve_parachute_diameter(mp, md, ground, v_desc)
    v_0 = math.pi * d_0 ** 3 / 6.0
    k0 = ground.rho_air - ground.rho_he
    lift_d = k0 * v_0                                 # net buoyancy of parachute balloon (kg)
    m_eff = mp + md - lift_d                          # eq. 4.12 (same at any altitude)
    m_he_d = ground.rho_he * v_0                      # eq. 4.16 (helium mass is conserved)
    chute = balloon_from_helium_mass(m_he_d, ground, alt)

    # approximate descent speed at burst altitude (same balloon, thin air, expanded)
    a_alt = math.pi * chute["d_alt_m"] ** 2 / 4.0
    v_alt = math.sqrt(2.0 * m_eff * G / (alt.rho_air * CD * a_alt)) if m_eff > 0 else float("nan")

    # --- carrier balloon
    total_dry = mp + md + mc
    free_req = total_dry * FREE_LIFT_FRACTION
    lift_total_req = total_dry + free_req
    lift_c_req = lift_total_req - lift_d              # what the carrier must add for lift
    m_lift = helium_mass_for_lift(lift_c_req) if lift_c_req > 0 else 0.0
    m_min = CARRIER_HELIUM_RATIO * m_he_d             # carrier must hold more helium than the chute
    m_c = max(m_lift, m_min)
    carrier = balloon_from_helium_mass(m_c, ground, alt)
    lift_c = m_c * (M_AIR / M_HE - 1.0)
    lift_total = lift_c + lift_d

    return {
        "total_g": total_dry * 1000, "free_req_g": free_req * 1000,
        "lift_total_req_g": lift_total_req * 1000, "lift_c_req_g": lift_c_req * 1000,
        "lift_c_g": lift_c * 1000, "lift_d_g": lift_d * 1000, "lift_total_g": lift_total * 1000,
        "free_actual_g": (lift_total - total_dry) * 1000,
        "free_actual_pct": (lift_total - total_dry) / total_dry * 100.0,
        "m_eff_g": m_eff * 1000, "chute": chute, "carrier": carrier, "v_ground": v_desc, "v_alt": v_alt,
        "ratio_limited": m_min > m_lift,
        "size_ratio": carrier["d_alt_m"] / chute["d_alt_m"],
    }


# ------------------------------------------------------------------- output
def print_balloon(title, lift_g, b, hang_g=None):
    print(f"\n  {title}")
    print(f"    Helium at ground level : {b['v_ground_m3'] * 1000:10.0f} L   |  {b['m_he_kg'] * 1000:8.1f} g")
    print(f"    Balloon size (ground)  : {b['d_ground_m']:10.2f} m diameter (sphere approx.)")
    print(f"    Balloon size (altitude): {b['d_alt_m']:10.2f} m diameter  <- compare with balloon burst diameter")
    print(f"    Net lift at ground     : {lift_g:10.0f} g")
    if hang_g is not None:
        print(f"    Weight to hang to fill : {hang_g:10.0f} g  (lift minus fill pipe/nozzle)")


def print_single(r, hardware_g):
    print("\n" + "=" * 62)
    print(" RESULT - 1 BALLOON (ground-level values)")
    print("=" * 62)
    print(f"  Total weight (no free lift) : {r['total_g']:8.0f} g")
    print(f"  Free-lift margin ({FREE_LIFT_FRACTION * 100:.0f} %)    : {r['free_g']:8.0f} g")
    print(f"  Weight the balloon must lift: {r['lift_g']:8.0f} g")
    print_balloon("Balloon", r["lift_g"], r["balloon"], max(0.0, r["lift_g"] - hardware_g))
    print("=" * 62)


def print_double(r, hardware_g):
    print("\n" + "=" * 62)
    print(" RESULT - 2 BALLOONS (ground-level values)")
    print("=" * 62)
    print(f"  Total dry weight (both balloons incl.) : {r['total_g']:8.0f} g")
    print(f"  Free-lift margin target ({FREE_LIFT_FRACTION * 100:.0f} %)      : {r['free_req_g']:8.0f} g")
    print(f"  Total net lift required                : {r['lift_total_req_g']:8.0f} g")
    print(f"  Parachute balloon net buoyancy         : {r['lift_d_g']:8.0f} g")
    print(f"  Carrier lift needed for the {FREE_LIFT_FRACTION * 100:.0f} % target: {max(r['lift_c_req_g'], 0):8.0f} g")

    print_balloon("CARRIER balloon (must burst first)", r["lift_c_g"], r["carrier"],
                  max(0.0, r["lift_c_g"] - hardware_g))
    if r["ratio_limited"]:
        print(f"    Helium set by the carrier-burst rule ({CARRIER_HELIUM_RATIO:g} x parachute-balloon helium)")
    else:
        print("    Helium set by the lift requirement (already above the carrier-burst rule)")
    print(f"    Carrier is {r['size_ratio']:.2f} x the parachute balloon's diameter at altitude")

    print_balloon("PARACHUTE balloon (must stay intact)", r["lift_d_g"], r["chute"])

    print(f"\n  Actual total net lift   : {r['lift_total_g']:8.0f} g")
    print(f"  Actual free lift        : {r['free_actual_g']:8.0f} g  ({r['free_actual_pct']:.0f} % of dry weight)")
    if r["free_actual_pct"] > 20.0:
        print("  !  Above 20 %: expect an ascent faster than ~5 m/s.")
    print(f"\n  Descent after carrier burst (effective mass {r['m_eff_g']:.0f} g):")
    print(f"    At ground level        : {r['v_ground']:.1f} m/s  (your target)")
    print(f"    At burst altitude      : ~{r['v_alt']:.0f} m/s  (thin air - much faster)")
    print("    (sphere approximation; ignores latex elasticity, slack envelope and the")
    print("     burst carrier's remains)")
    print("=" * 62)


# -------------------------------------------------------------------- input
def ask_float(prompt, positive=True, maximum=None, default=None):
    while True:
        raw = input(prompt).strip().replace(",", ".")
        if raw == "" and default is not None:
            return default
        try:
            v = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        if (positive and v <= 0) or (not positive and v < 0):
            print("  Value must be " + ("greater than 0." if positive else "0 or more."))
            continue
        if maximum is not None and v > maximum:
            print(f"  Value must be at most {maximum:g}.")
            continue
        return v


def main():
    print("=" * 62)
    print(" HIGH-ALTITUDE BALLOON HELIUM CALCULATOR")
    print(f" Free-lift fraction: {FREE_LIFT_FRACTION * 100:.0f} %   |  Launch elevation: {GROUND_ALTITUDE_M:g} m")
    print("=" * 62)

    while True:
        n = ""
        while n not in ("1", "2"):
            n = input("\nHow many balloons are used? (1 or 2): ").strip()

        if n == "1":
            total = ask_float("Total payload weight incl. balloon, parachute, cord, "
                              "NO free lift (g): ")
            alt = ask_float(f"Desired altitude (m, max {MAX_ALTITUDE_M:g}): ", maximum=MAX_ALTITUDE_M)
            hw = ask_float("Fill pipe + nozzle weight (g) [Enter = 0]: ", positive=False, default=0.0)
            print_single(calc_single(total, alt), hw)
        else:
            train = ask_float("Total payload weight EXCLUDING both balloons "
                              "(payload, cord, rigging, no free lift) (g): ")
            carrier = ask_float("Carrier balloon weight, uninflated (g): ")
            chute = ask_float("Parachute balloon weight incl. its rigging, uninflated (g): ")
            alt = ask_float(f"Desired altitude (m, max {MAX_ALTITUDE_M:g}): ", maximum=MAX_ALTITUDE_M)
            vd = ask_float("Desired descent rate at ground level (m/s): ")
            hw = ask_float("Fill pipe + nozzle weight (g) [Enter = 0]: ", positive=False, default=0.0)
            print_double(calc_double(train, carrier, chute, alt, vd), hw)

        if input("\nRun another calculation? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()