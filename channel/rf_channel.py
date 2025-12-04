# ============================================================
# rf_channel_20GHz.py (distance = 800 km fixed, 19 GHz model)
# RF減衰 + SNR + 容量モデル（19GHz専用、SciPyなし）
# ============================================================

import numpy as np

C_LIGHT = 299_792_458.0  # [m/s]

# =============================
# 固定距離 800 km
# =============================
FIXED_DISTANCE_KM = 800.0

def distance_km(p1_km, p2_km):
    """距離は常に 800 km とする"""
    return FIXED_DISTANCE_KM

# =============================
# FSPL
# =============================
def free_space_loss_dB(dist_km, freq_ghz):
    """自由空間損失 FSPL [dB]"""
    d_m = dist_km * 1e3
    f_hz = freq_ghz * 1e9
    return 20*np.log10(d_m) + 20*np.log10(f_hz) - 147.55


# =============================
# 雨減衰 (ITU-R P.838, 19GHz)
# =============================
def rain_specific_k_alpha_19ghz(angle_deg=0.0):
    """
    あなたの旧コードの ↓ に対応
      kh = 0.08084, kv = 0.08642
      αh = 1.0691, αv = 0.9930
    """

    kh = 0.08084
    kv = 0.08642
    alpha_h = 1.0691
    alpha_v = 0.9930

    # elevation angle の依存を削除して angle=0° とみなす
    angle = np.radians(angle_deg)
    cos_term = np.cos(angle)**2 * np.cos(np.pi)

    k = (kh + kv + (kh - kv)*cos_term) / 2
    alpha = (kh*alpha_h + kv*alpha_v + (kh*alpha_h - kv*alpha_v)*cos_term) / (2*k)

    return k, alpha


def rain_attenuation_dB(rain_rate, path_km=5.0, angle_deg=0.0):
    """A_rain = k * R^α * path"""
    k, alpha = rain_specific_k_alpha_19ghz(angle_deg)
    gamma = k * (rain_rate ** alpha)
    return gamma * path_km


# =============================
# 雲減衰（旧 specific_attenuation_coefficient）
# =============================
def cloud_specific_attenuation_dB_per_km(freq_ghz, cloud_density, temperature):
    """
    specific_attenuation_coefficient(freq, temperature, w)
    を数学的に簡略化したもの
    """
    f = freq_ghz * 1e9
    T = temperature

    # 旧コードの係数を再現する近似式（SciPyなし）
    eps0 = 77.66 + 103.3 * (300 / T - 1)
    eps1 = 0.0671 * eps0
    eps2 = 3.52
    fp = 20.20 - 146*(300/T - 1) + 316*(300/T - 1)**2
    fs = 39.8 * fp

    eps_prime = eps0 - (eps0 - eps1)/(1 + (f/fp)**2) - (eps1 - eps2)/(1 + (f/fs)**2) + eps2
    eps_double_prime = (f*(eps0-eps1)/fp)/(1+(f/fp)**2) + (f*(eps1-eps2)/fs)/(1+(f/fs)**2)

    eta = (2 + eps_prime) / eps_double_prime
    kl = 0.819 * f / (eps_double_prime*(1 + eta**2)) * cloud_density  # [dB/km]
    return kl


def cloud_attenuation_dB(freq_ghz, cloud_density, temperature, path_km=5.0):
    kl = cloud_specific_attenuation_dB_per_km(freq_ghz, cloud_density, temperature)
    return kl * path_km


# =============================
# Shadowed–Rician フェージング (固定)
# =============================
H1_SQUARED = 0.6233226259481273  # E[h']^2 の固定値


# =============================
# 総減衰
# =============================
def total_rf_attenuation_dB(dist_km,
                            rain_rate,
                            cloud_density,
                            freq_ghz,
                            params):
    fspl = free_space_loss_dB(dist_km, freq_ghz)

    A_rain = rain_attenuation_dB(
        rain_rate,
        path_km=params["rain_path_km"],
        angle_deg=0.0
    )

    A_cloud = cloud_attenuation_dB(
        freq_ghz,
        cloud_density,
        params.get("temperature", 273.15),
        path_km=params["cloud_path_km"]
    )


    A_atm = params["atm_const_dB"]

    return fspl + A_rain + A_cloud + A_atm


# =============================
# 容量計算
# =============================
def rf_capacity_bps(tx_pos_km,
                    rx_pos_km,
                    rain_rate,
                    cloud_density,
                    freq_ghz,
                    bw_hz,
                    pt_dbw,
                    gt_dbi,
                    gr_dbi,
                    N0_dbw_per_hz,
                    params):
    dist = distance_km(tx_pos_km, rx_pos_km)

    L_tot = total_rf_attenuation_dB(
        dist, rain_rate, cloud_density, freq_ghz, params
    )

    # フェージング利得を送信電力に吸収したので h1² は使わない
    snr_db = pt_dbw + gt_dbi + gr_dbi \
             - L_tot \
             - (N0_dbw_per_hz + 10*np.log10(bw_hz))

    snr_lin = 10 ** (snr_db / 10.0)

    return bw_hz * np.log2(1 + snr_lin)


# =============================
# 晴天/降雨 capacity
# =============================
def rf_capacity_clear_and_rain(tx_pos_km, rx_pos_km,
                               rain_rate, cloud_density,
                               freq_ghz, bw_hz,
                               pt_dbw, gt_dbi, gr_dbi,
                               N0_dbw_per_hz,
                               params):

    C_clear = rf_capacity_bps(
        tx_pos_km, rx_pos_km,
        rain_rate=0.0,
        cloud_density=0.0,
        freq_ghz=freq_ghz,
        bw_hz=bw_hz,
        pt_dbw=pt_dbw,
        gt_dbi=gt_dbi,
        gr_dbi=gr_dbi,
        N0_dbw_per_hz=N0_dbw_per_hz,
        params=params
    )

    C_rain = rf_capacity_bps(
        tx_pos_km, rx_pos_km,
        rain_rate=rain_rate,
        cloud_density=cloud_density,
        freq_ghz=freq_ghz,
        bw_hz=bw_hz,
        pt_dbw=pt_dbw,
        gt_dbi=gt_dbi,
        gr_dbi=gr_dbi,
        N0_dbw_per_hz=N0_dbw_per_hz,
        params=params
    )

    return C_clear, C_rain


# =============================
# 19 GHz 用のパラメータセット（旧コード互換）
# =============================
params_19ghz = {
    "rain_path_km": 5.0,
    "cloud_path_km": 5.0,
    "atm_const_dB": 0.5,
    "temperature": 273.15
}
