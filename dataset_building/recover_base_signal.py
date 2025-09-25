


import numpy as np
import batman
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import brentq

R_EARTH_TO_RSUN = 1.0 / 109.1  # Earth radii → solar radii
BKJD_ZERO = 2454833.0          # Kepler zero-point

def _to_bkjd_days(time_like):
    """Convert JD/BJD to BKJD if needed."""
    t = np.asarray(time_like, dtype=float)
    if np.nanmin(t) > 2e6:  # JD scale
        t = t - BKJD_ZERO
    return t

def build_base_signal(
    base_light_curve,           # DataFrame with columns "time","flux"
    star_radius_Rsun,           # koi_srad [R☉]
    t0_bkjd,                    # koi_time0bk [BKJD days]
    impact_b,                   # koi_impact
    planet_radius_Rearth,       # koi_prad [R⊕]
    planet_period_days,         # koi_period [days]
    transit_duration_hours      # koi_duration [hours]
):
    # Extract time in BKJD days
    time = _to_bkjd_days(base_light_curve["time"].to_numpy())

    # Convert inputs
    RpRs = (planet_radius_Rearth * R_EARTH_TO_RSUN) / star_radius_Rsun
    P = float(planet_period_days)                 # days
    T_days = float(transit_duration_hours) / 24.0 # hours → days
    b = float(impact_b)

    # Quick geometry check: transit possible?
    if RpRs <= 0 or b >= 1.0 + RpRs:
        return time, np.ones_like(time)  # flat light curve

    # Duration function
    def duration_from_aRs(aRs):
        cosi = b / aRs
        if cosi >= 1.0:
            return 0.0
        sini = np.sqrt(1.0 - cosi**2)
        arg = np.sqrt((1.0 + RpRs)**2 - b**2) / (aRs * sini)
        arg = np.clip(arg, -1.0, 1.0)
        return (P / np.pi) * np.arcsin(arg)

    # Solve a/Rs for given duration
    try:
        a_over_Rs = brentq(lambda a: duration_from_aRs(a) - T_days,
                           1.01, 1e3)
    except ValueError:
        # No solution → inconsistent inputs
        return time, np.ones_like(time)

    inc_deg = np.degrees(np.arccos(b / a_over_Rs))

    # Build batman parameters
    params = batman.TransitParams()
    params.t0 = float(t0_bkjd)
    params.per = P
    params.rp  = RpRs
    params.a   = a_over_Rs
    params.inc = inc_deg
    params.ecc = 0.0
    params.w   = 90.0
    params.limb_dark = "quadratic"
    params.u = [0.1, 0.3]  # placeholder LD coefficients

    flux_model = batman.TransitModel(params, time).light_curve(params)
    return time, flux_model


def sanity_check_lightcurves(time, flux_obs, flux_model, t0, period, dur_hours):
    dur_days = dur_hours / 24
   
    time = np.asarray(time)
    flux_obs = np.asarray(flux_obs)
    flux_model = np.asarray(flux_model)
    
    # Residuals
    res = flux_obs - flux_model
    rms_res = np.nanstd(res)

    # Phase fold
    phase = ((time - t0 + 0.5*period) % period) / period - 0.5
    in_transit = np.abs(phase) < (dur_days / (2*period))

    # Depth
    depth_obs = 1 - np.nanmedian(flux_obs[in_transit])
    depth_model = 1 - np.nanmedian(flux_model[in_transit])

    # Approx SNR = depth / (rms_out / sqrt(N))
    out_of_transit = ~in_transit
    rms_out = np.nanstd(flux_obs[out_of_transit])
    N_in = np.count_nonzero(in_transit)
    snr = depth_obs / (rms_out / np.sqrt(max(N_in,1)))

    print("Sanity check results")
    print(f"  Depth (obs):   {depth_obs:.6f}")
    print(f"  Depth (model): {depth_model:.6f}")
    print(f"  Duration (days): {dur_days:.4f} (~{dur_days*24:.2f} h)")
    print(f"  RMS residuals: {rms_res:.6e}")
    print(f"  Out-of-transit RMS: {rms_out:.6e}")
    print(f"  Approx SNR:    {snr:.2f}")

    return {
        "depth_obs": depth_obs,
        "depth_model": depth_model,
        "duration_days": dur_days,
        "residual_rms": rms_res,
        "snr": snr
    }

if __name__ == "__main__":
    
    kepid =  10872983
    
    candidates_table = pd.read_csv("data/candidates_lookup_table.csv")
    candidates = candidates_table[candidates_table["kepid"] == kepid]

    # Load base signal
    signal = pd.read_csv(f"data/light_curves/base/{kepid}.csv")

    # Start with flat baseline
    combined_flux = np.ones_like(signal["time"].to_numpy(), dtype=float)

    for _, cand in candidates.iterrows():
        star_radius = cand["koi_srad"]
        time_of_first_transit = cand["koi_time0bk"]
        min_distance_to_star_center = cand["koi_impact"]
        planet_radius = cand["koi_prad"]
        planet_period = cand["koi_period"]
        transit_duration = cand["koi_duration"]  # hours

        time, flux_model = build_base_signal(
            signal,
            star_radius,
            time_of_first_transit,
            min_distance_to_star_center,
            planet_radius,
            planet_period,
            transit_duration,
        )

        combined_flux *= flux_model  # multiplicative effect of multiple planets

    new_signal = signal.copy()
    new_signal["flux"] = combined_flux

    # Optional: run sanity check on combined model
    results = sanity_check_lightcurves(
        time,
        signal["flux"].to_numpy(),
        combined_flux,
        candidates.iloc[0]["koi_time0bk"],
        candidates.iloc[0]["koi_period"],
        candidates.iloc[0]["koi_duration"],
    )

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(new_signal["time"], new_signal["flux"], label="Model Transit(s)", alpha=0.8)
    plt.plot(signal["time"], signal["flux"], label="Original Signal", alpha=0.5)
    plt.xlabel("Time [BKJD days]")
    plt.ylabel("Normalized Flux")
    plt.title(f"Light Curve for KIC {kepid} ({len(candidates)} candidates)")
    plt.legend()
    plt.show()
    
    
    
    
    