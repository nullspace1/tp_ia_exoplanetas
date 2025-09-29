import numpy as np
import batman
import pandas as pd
import matplotlib.pyplot as plt

R_EARTH_TO_RSUN = 1.0 / 109.1 
BKJD_ZERO = 2454833.0

G_cgs = 6.6743e-8
R_sun_cgs = 6.957e10
M_sun_cgs = 1.9885e33

def _to_bkjd_days(time_like):
    t = np.asarray(time_like, dtype=float)
    if np.nanmin(t) > 2e6:  # JD scale
        t = t - BKJD_ZERO
    return t

def batman_phase_model(star_radius_Rsun, star_mass_Msun, t0_bkjd, impact_b,
                       planet_radius_Rearth, planet_period_days, bin_count):
    
    
    RpRs = (planet_radius_Rearth * R_EARTH_TO_RSUN) / star_radius_Rsun
    P_days = float(planet_period_days)
    b = float(impact_b)

    P_sec = P_days * 86400.0
    M_star = star_mass_Msun * M_sun_cgs
    R_star = star_radius_Rsun * R_sun_cgs
    a_cm = (G_cgs * M_star * P_sec**2 / (4*np.pi**2))**(1/3)
    a_over_Rs = a_cm / R_star
    
    cosi = b / a_over_Rs
    inc_deg = np.degrees(np.arccos(cosi))
    
    phase_interval = np.linspace(-0.5, 0.5, bin_count + 1)
    time_grid = t0_bkjd + phase_interval*P_days

    params = batman.TransitParams()
    params.t0 = float(t0_bkjd)
    params.per = P_days
    params.rp = RpRs
    params.a = a_over_Rs
    params.inc = inc_deg
    params.ecc = 0.0
    params.w = 90.0
    params.limb_dark = "quadratic"
    params.u = [0.1, 0.3]

    m = batman.TransitModel(params, time_grid)
    flux = m.light_curve(params)

    return time_grid, flux
