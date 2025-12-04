# ============================================================
# fso_channel.py
# FSO減衰 + SNR + 容量モデル（雨の影響なし）
# ============================================================

import numpy as np

C_LIGHT = 299_792_458.0   # [m/s]
LAMBDA = 1550e-9          # 波長 1550 nm を想定（一般的な地上⇔衛星FSO）

# ----------------------------
# 基本関数
# ----------------------------
def distance_km(p1_km, p2_km):
    """2点 [km] のユークリッド距離 [km]"""
    return float(np.linalg.norm(np.asarray(p1_km) - np.asarray(p2_km)))


# ----------------------------
# 幾何損失（光学ビーム広がり）
# ----------------------------
def geometric_loss_dB(dist_km, tx_diameter_m, divergence_rad):
    """
    幾何損失 [dB]
    divergence_rad: ビーム広がり角（例：100 µrad）
    """
    dist_m = dist_km * 1000

    beam_radius = dist_m * divergence_rad
    aperture_radius = tx_diameter_m / 2

    # 受光面積に対するビーム面積比
    ratio = (aperture_radius / beam_radius) ** 2
    ratio = max(ratio, 1e-20)

    return -10 * np.log10(ratio)


# ----------------------------
# 大気減衰（霧・霞・エアロゾル：雨は入れない）
# ----------------------------
def atmospheric_attenuation_dB(dist_km, visibility_km):
    """
    Beer-Lambert の簡易モデル
      γ = 3.91 / visibility_km  [dB/km]
    """
    if visibility_km <= 0.1:
        visibility_km = 0.1

    gamma = 3.91 / visibility_km
    return gamma * dist_km


# ----------------------------
# FSO容量
# ----------------------------
def fso_capacity_bps(tx_pos_km,
                     rx_pos_km,
                     visibility_km,
                     bw_hz,
                     pt_dbw,
                     G_tx_dB,
                     G_rx_dB,
                     divergence_rad,
                     aperture_m,
                     N0_dbw_per_hz):
    """
    FSO容量 [bps]
    - visibility_km: 視程（霧などによる光減衰）
    - 雨の影響は入れない
    """

    dist = distance_km(tx_pos_km, rx_pos_km)

    # 幾何損失
    L_geo = geometric_loss_dB(dist, aperture_m, divergence_rad)

    # 大気減衰（雨の項なし！）
    L_atm = atmospheric_attenuation_dB(dist, visibility_km)

    # 総減衰
    L_tot = L_geo + L_atm

    # SNR計算
    snr_db = pt_dbw + G_tx_dB + G_rx_dB \
             - L_tot \
             - (N0_dbw_per_hz + 10 * np.log10(bw_hz))

    snr_lin = 10 ** (snr_db / 10)
    C = bw_hz * np.log2(1 + snr_lin)
    return C
