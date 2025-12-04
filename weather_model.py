# weather_model.py
# ============================================================
# SimpleWeatherModel
# ------------------------------------------------------------
# ・衛星位置 sat_pos_km と GS 位置 gs_pos_km を入力に
#   雨量 Rain [mm/h] と Cloud 量（雲水量）を返す。
# ・今回は「呼ばれたタイミングでだけ」ランダム生成。
#   → MultiStationEnv 側で HO のときだけ呼ぶので，
#      それ以外のステップでは天気は変わらない。
# ============================================================

import numpy as np


class SimpleWeatherModel:
    """
    get_weather() の返す値：
        rain_rate: [mm/h] 降雨量
        cloud     : [g/m^3] 雲水量（RFrev2で使用）
    """

    def __init__(
        self,
        seed: int = 0,
        rain_min: float = 0.0,
        rain_max: float = 10.0,
        cloud_min: float = 0.0,
        cloud_max: float = 1.0,
    ):
        self.rng = np.random.default_rng(seed)

        self.rain_min = rain_min
        self.rain_max = rain_max
        self.cloud_min = cloud_min
        self.cloud_max = cloud_max

    def get_weather(self, sat_pos_km, gs_pos_km):
        """
        入力：
            sat_pos_km: np.array([x,y,z]) [km]
            gs_pos_km : np.array([x,y,z]) [km]
        出力：
            (rain_rate, cloud)
        ※ 今回は位置には依存させず，HO 時に呼ばれる度にランダムに決める。
        """
        if sat_pos_km is None or gs_pos_km is None:
            # 可視外などの場合は晴天扱い
            return 0.0, 0.0

        rain = float(self.rng.uniform(self.rain_min, self.rain_max))
        cloud = float(self.rng.uniform(self.cloud_min, self.cloud_max))
        return rain, cloud
