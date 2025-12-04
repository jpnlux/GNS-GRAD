#!/home/anaconda3/bin/python
# ==============================================================================
# 衛星通信ネットワーク トポロジ・ルーティング最適化プログラム (Ver. 6.6.1 コメント強化版)
# - 目的：各トポロジ構築手法を、統一された事前計算型経路制御アルゴリズムの下で評価・比較する。
# - 変更点：コード全体に、各処理の目的を説明する詳細なコメントを追記。
# ==============================================================================

# ### ライブラリのインポート ###
import numpy as np  # 数値計算ライブラリ
import networkx as nx  # ネットワークグラフ操作ライブラリ
from RFrev2 import S2U_RF_throughput, S2U_RF_throughput_for29GHz  # 外部のRFスループット計算モジュール
import csv  # CSVファイル読み書き
import os
import random  # 乱数生成
import itertools  # 繰り返し処理の組み合わせ生成
import pandas as pd  # データ分析ライブラリ
import time
import sys


# ==============================================================================
# 1. 設定管理クラス
# ==============================================================================
class SimulationConfig:
    """
    シミュレーション全体のパラメータや設定を一元管理するクラス。
    物理定数、衛星の仕様、通信シナリオなどを定義する。
    """
    # --- 物理定数 ---
    EARTH_RADIUS_KM = 6371.0  # 地球の半径 (km)
    SPEED_OF_LIGHT_KM_S = 299792.458  # 光速 (km/s)
    
    # --- 衛星コンステレーション設定 ---
    DEFAULT_NUM_ORBITS = 6  # 軌道面の数
    DEFAULT_NUM_SATS_PER_ORBIT = 16  # 1軌道面あたりの衛星数
    DEFAULT_INCLINATION_DEG = 80.0  # 軌道傾斜角 (度)
    SATELLITE_ALTITUDE_KM = 1000.0  # 衛星の高度 (km)
    SATELLITE_ORBIT_RADIUS_KM = EARTH_RADIUS_KM + SATELLITE_ALTITUDE_KM  # 衛星の軌道半径 (km)
    
    # --- 通信パラメータ ---
    FSO_THROUGHPUT_GBPS = 10.0  # 衛星間光通信(FSO)の伝送容量 (Gbps)
    UPLINK_FREQUENCY_GHZ = 29.0  # 地上から衛星へのアップリンク周波数 (GHz)
    DOWNLINK_FREQUENCY_GHZ = 19.0  # 衛星から地上へのダウンリンク周波数 (GHz)
    
    # --- シミュレーションシナリオ設定 ---
    # 考慮する天候パターンのリスト (降雨率 [mm/h], 雲量)
    WEATHER_CONDITIONS = [
        (0.0, 0.0),   # 晴天
        (3.0, 0.5),   # 雨天
        (10.0, 0.5)   # 強い雨
    ]

    # --- トポロジ探索に用いる通信シナリオの定義 ---
    # シナリオ1: London宛の大容量データが中心の通信パターン
    # _SEARCH_DATA_SCENARIO_1 = [
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    # ]
    # # シナリオ2: 欧州各地への大容量データが分散する通信パターン
    # _SEARCH_DATA_SCENARIO_2 = [
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0}, 
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0}, 
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    # ]

    # _SEARCH_DATA_SCENARIO_1 = [
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0}, 
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0}, 
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    # ]
    # _SEARCH_DATA_SCENARIO_2 = [
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1}, 
    #     {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1}, 
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    #     {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    # ]
    _SEARCH_DATA_SCENARIO_1 = [
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 10.0},
    ]
    _SEARCH_DATA_SCENARIO_2 = [
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'London', 'size_mb': 10.0},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Paris', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
        {'source': 'Tokyo', 'dest': 'Glasgow', 'size_mb': 0.1},
    ]

    # --- 検証用データの自動生成パラメータ ---
    ALPHA_AVG_SIZE_MB = 0.08  # 小サイズデータ（α）の平均値
    ALPHA_STD_DEV_MB = 0.04   # 小サイズデータ（α）の標準偏差
    BETA_AVG_SIZE_MB = 10.0   # 大サイズデータ（β）の平均値
    BETA_STD_DEV_MB = 1.5     # 大サイズデータ（β）の標準偏差

    # --- その他 ---
    K_SHORTEST_PATHS = 5  # 経路探索時に考慮するk-shortest pathの数
    GROUND_STATION_CSV = 'cities_sample.csv'  # 地上局データファイル名

    OUTPUT_FILENAME_BASE = '20250827results'

    def __init__(self, scenario_id=1):
        """
        設定クラスの初期化メソッド。
        引数 `scenario_id` に応じて、使用する通信シナリオを決定する。
        """
        if scenario_id == 1:
            self.SEARCH_DATA_TRANSFERS = self._SEARCH_DATA_SCENARIO_1
        elif scenario_id == 2:
            self.SEARCH_DATA_TRANSFERS = self._SEARCH_DATA_SCENARIO_2
        else:
            # 不正なIDが指定された場合は、デフォルトでシナリオ1を使用
            self.SEARCH_DATA_TRANSFERS = self._SEARCH_DATA_SCENARIO_1



# ==============================================================================
# 2. 地上局クラス
# ==============================================================================
class GroundStation:
    """個々の地上局の属性（位置座標、天候など）を管理するクラス。"""
    def __init__(self, name, lat_deg, lon_deg, earth_radius_km):
        self.name = name
        self.latitude_deg = float(lat_deg)
        self.longitude_deg = float(lon_deg)
        # 地理座標（緯度経度）を三次元の地心固定座標系 (ECEF) に変換
        self.ecef_coords = self._geodetic_to_ecef(earth_radius_km)
        # 各地上局の天候状態を初期化
        self.rain_rate = 0.0
        self.cloud_density = 0.0

    def _geodetic_to_ecef(self, earth_radius_km):
        """緯度・経度をECEF座標に変換する。"""
        lat_rad, lon_rad = np.deg2rad(self.latitude_deg), np.deg2rad(self.longitude_deg)
        x = earth_radius_km * np.cos(lat_rad) * np.cos(lon_rad)
        y = earth_radius_km * np.cos(lat_rad) * np.sin(lon_rad)
        z = earth_radius_km * np.sin(lat_rad)
        return np.array([x, y, z])

    def set_weather(self, rain_rate, cloud_density):
        """この地上局の天候状態を更新する。"""
        self.rain_rate, self.cloud_density = rain_rate, cloud_density

# ==============================================================================
# 3. 衛星コンステレーションクラス
# ==============================================================================
class SatelliteConstellation:
    """衛星コンステレーション全体（全衛星の位置座標、衛星間リンク）を生成・管理するクラス。"""
    def __init__(self, config):
        self.config = config
        self.num_sats = config.DEFAULT_NUM_ORBITS * config.DEFAULT_NUM_SATS_PER_ORBIT
        self.sats_coords, self.id_map1, self.id_map2 = {}, {}, {}
        self.graph = nx.Graph()  # 衛星間リンクを表現するグラフ構造
        self._generate_constellation()  # 全衛星の座標を生成
        self._create_network_graph()  # 衛星間をリンクで接続

    def _generate_constellation(self):
        """設定に基づき、軌道力学的な計算を行い、全衛星のECEF座標を配置する。"""
        inclination_rad = np.deg2rad(self.config.DEFAULT_INCLINATION_DEG)
        sat_counter = 0
        for i in range(self.config.DEFAULT_NUM_ORBITS):
            raan_rad = np.deg2rad(360.0 * i / self.config.DEFAULT_NUM_ORBITS) # 昇交点赤経(RAAN)の計算
            for k in range(self.config.DEFAULT_NUM_SATS_PER_ORBIT):
                phase = (360.0/(2.0*self.config.DEFAULT_NUM_SATS_PER_ORBIT)) * (i % 2) if self.config.DEFAULT_NUM_ORBITS > 1 else 0.0
                anomaly = np.deg2rad((360.0 * k / self.config.DEFAULT_NUM_SATS_PER_ORBIT) + phase) # 真近点角(True Anomaly)の計算
                # 軌道座標系からECEF座標系へ変換
                x_o, y_o = self.config.SATELLITE_ORBIT_RADIUS_KM * np.cos(anomaly), self.config.SATELLITE_ORBIT_RADIUS_KM * np.sin(anomaly)
                x_r, y_r, z_r = x_o, y_o * np.cos(inclination_rad), y_o * np.sin(inclination_rad)
                x_e, y_e = x_r * np.cos(raan_rad) - y_r * np.sin(raan_rad), x_r * np.sin(raan_rad) + y_r * np.cos(raan_rad)
                # 計算結果を格納
                self.sats_coords[sat_counter] = np.array([x_e, y_e, z_r])
                self.id_map1[sat_counter] = (i, k)
                self.id_map2[(i, k)] = sat_counter
                sat_counter += 1

    def _create_network_graph(self):
        """隣接する衛星同士をリンクで接続し、ネットワークグラフを作成する。"""
        for sat_id in self.sats_coords:
            self.graph.add_node(sat_id)
        for cid, (orb, sat) in self.id_map1.items():
            p1 = self.sats_coords[cid]
            # 同一軌道面内(intra-orbit)の次の衛星と接続
            next_intra = self.id_map2[(orb, (sat + 1) % self.config.DEFAULT_NUM_SATS_PER_ORBIT)]
            self.graph.add_edge(cid, next_intra, weight=np.linalg.norm(p1 - self.sats_coords[next_intra]))
            # 隣の軌道面(inter-orbit)上の衛星と接続
            if self.config.DEFAULT_NUM_ORBITS > 1:
                next_inter = self.id_map2[((orb + 1) % self.config.DEFAULT_NUM_ORBITS, sat)]
                self.graph.add_edge(cid, next_inter, weight=np.linalg.norm(p1 - self.sats_coords[next_inter]))

# ==============================================================================
# 4. 経路探索エンジンクラス
# ==============================================================================
class RoutingEngine:
    """通信容量の計算や遅延評価など、ルーティング関連の計算機能を提供するクラス。"""
    def __init__(self, config):
        self.config = config

    def _get_access_link_capacity_gbps(self, u, v, all_gs, constellation, is_uplink):
        """地上局-衛星間の通信容量(スループット)を計算する。"""
        gs_node, sat_node = (u, v) if isinstance(u, str) else (v, u)
        gs = all_gs[gs_node]
        freq = self.config.UPLINK_FREQUENCY_GHZ if is_uplink else self.config.DOWNLINK_FREQUENCY_GHZ
        throughput_func = S2U_RF_throughput_for29GHz if is_uplink else S2U_RF_throughput
        # 外部の物理モデル(RFrev.py)を呼び出してスループットを計算
        return (1e-6) * throughput_func(gs.ecef_coords, constellation.sats_coords[int(sat_node)], 
                                         gs.rain_rate, gs.cloud_density, freq, 1)

    def _calculate_total_delay_new_formula(self, data_transfers, paths_dict, G, all_gs, constellation):
        """指定された経路リストに基づき、データ転送の総遅延時間を計算する（ボトルネックモデル）。"""
        total_delay_ms = 0
        # データ転送（フロー）ごとに遅延を計算し、それらを合計する
        for i, path in paths_dict.items():
            # --- 1. このフローの「伝搬遅延」を計算 (距離 ÷ 光速) ---
            prop_delay_s = sum(G.edges[path[j], path[j + 1]]['weight'] for j in range(len(path) - 1))
            prop_delay_ms = prop_delay_s * 1000

            # --- 2. このフローの「伝送遅延」をボトルネック基準で計算 ---
            tx_delay_ms = 0
            data_size_mb = data_transfers[i]['size_mb']

            # 経路の中からアップリンクとダウンリンクの地上-衛星間アクセス区間を特定
            uplink_gs, uplink_sat, downlink_gs, downlink_sat = None, None, None, None
            for j in range(len(path) - 1):
                if isinstance(path[j], str) and isinstance(path[j + 1], int):
                    uplink_gs, uplink_sat = path[j], path[j + 1]; break
            for j in range(len(path) - 1, 0, -1):
                if isinstance(path[j], str) and isinstance(path[j - 1], int):
                    downlink_gs, downlink_sat = path[j], path[j - 1]; break
            
            # アップリンクとダウンリンクの容量を比較し、より低い方（ボトルネック）を特定
            bottleneck_capacity_gbps = float('inf')
            if uplink_gs:
                cap_up = self._get_access_link_capacity_gbps(uplink_gs, uplink_sat, all_gs, constellation, is_uplink=True)
                bottleneck_capacity_gbps = min(bottleneck_capacity_gbps, cap_up if cap_up > 0 else 0)
            if downlink_gs:
                cap_down = self._get_access_link_capacity_gbps(downlink_gs, downlink_sat, all_gs, constellation, is_uplink=False)
                bottleneck_capacity_gbps = min(bottleneck_capacity_gbps, cap_down if cap_down > 0 else 0)

            # ボトルネック容量に基づき、伝送遅延を計算
            if 0 < bottleneck_capacity_gbps < float('inf'):
                tx_delay_ms = (data_size_mb * 8) / bottleneck_capacity_gbps # T[ms] = (S[MB]*8)/C[Gbps]
            elif bottleneck_capacity_gbps == 0:
                tx_delay_ms = float('inf') # 容量が0なら通信不可

            # このフローの合計遅延（伝搬＋伝送）を全体の遅延に加算
            total_delay_ms += (prop_delay_ms + tx_delay_ms)
        return total_delay_ms

    def find_best_routing_for_search(self, gs_sat_links, data_transfers, all_gs, constellation):
        """提案手法で用いる評価指標（1MBあたり平均遅延）を計算する。"""
        G = SimulationRunner._build_full_network_graph_static(gs_sat_links, all_gs, constellation, self.config)
        final_paths = {}
        # 各データ転送を1つずつ、最適な経路に割り当てていく（Greedy法）
        for i, transfer in enumerate(data_transfers):
            source, dest, best_path = transfer['source'], transfer['dest'], None
            min_overall_delay, temp_paths = float('inf'), final_paths.copy()
            try:
                # 複数の経路候補（k-shortest paths）を探索
                k_shortest_paths = list(itertools.islice(nx.shortest_simple_paths(G, source, dest, weight='weight'), 3))
            except (nx.NetworkXNoPath, nx.NodeNotFound): continue
            # 経路候補の中で、全体の総遅延が最も小さくなる経路を選択
            for candidate_path in k_shortest_paths:
                temp_paths[i] = candidate_path
                current_total_delay = self._calculate_total_delay_new_formula(data_transfers, temp_paths, G, all_gs, constellation)
                if current_total_delay < min_overall_delay:
                    min_overall_delay, best_path = current_total_delay, candidate_path
            if best_path: final_paths[i] = best_path
        
        # 最終的な総遅延を計算
        final_total_delay_ms = self._calculate_total_delay_new_formula(data_transfers, final_paths, G, all_gs, constellation)
        # 全データサイズで割り、1MBあたりの平均遅延を算出
        total_data_volume_mb = sum(t['size_mb'] for t in data_transfers)
        # この値が最小となるトポロジが最適と判断される
        return final_total_delay_ms / total_data_volume_mb if total_data_volume_mb > 0 else 0

# ==============================================================================
# 5. 実行クラス
# ==============================================================================
class SimulationRunner:
    """シミュレーションの実行フロー全体を管理・制御するクラス。"""
    def __init__(self, config):
        self.config = config
        self.ground_stations = self._load_ground_stations()
        self.constellation = SatelliteConstellation(config)
        self.routing_engine = RoutingEngine(config)

    def _load_ground_stations(self):
        """CSVファイルから地上局の情報を読み込む。"""
        stations = {}
        valid_gs_names = ['Tokyo', 'Sendai', 'Hakodate', 'London', 'Paris', 'Glasgow']
        try:
            with open(self.config.GROUND_STATION_CSV, mode='r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get('name', '').strip()
                    if name in valid_gs_names:
                        stations[name] = GroundStation(name, row['latitude'], row['longitude'], self.config.EARTH_RADIUS_KM)
            return stations
        except FileNotFoundError:
            print(f"エラー: 地上局CSVファイル '{self.config.GROUND_STATION_CSV}' が見つかりません。")
            return {}

    def _generate_validation_data(self):
        """検証フェーズで使用するデータセットを、構築用データセットを元に自動生成する。"""
        validation_data = []
        size_threshold = (self.config.ALPHA_AVG_SIZE_MB + self.config.BETA_AVG_SIZE_MB) / 2.0
        # 構築用データの(送信元, 宛先)ペアはそのままに、データサイズだけを乱数で変動させる
        for original_transfer in self.config.SEARCH_DATA_TRANSFERS:
            source = original_transfer['source']
            dest = original_transfer['dest'].replace('Pairs', 'Paris')
            if original_transfer['size_mb'] < size_threshold:
                new_size = np.random.normal(self.config.ALPHA_AVG_SIZE_MB, self.config.ALPHA_STD_DEV_MB)
                final_size = max(0.01, new_size)
            else:
                new_size = np.random.normal(self.config.BETA_AVG_SIZE_MB, self.config.BETA_STD_DEV_MB)
                final_size = max(1.0, new_size)
            validation_data.append({'source': source, 'dest': dest, 'size_mb': final_size})
        return validation_data
    
    def run_full_optimization_cycle(self, config_name="Default Scenario"):
        """【メイン処理】シミュレーションの全サイクルを実行する。"""
        if not self.ground_stations: return

        # --- 1. 初期設定と情報表示 ---
        print(f"🛰️  衛星ネットワーク トポロジ探索・検証プログラム ({config_name}) 開始 🛰️")
        print("-" * 80)
        print(">>> 構築用データ (SEARCH_DATA_TRANSFERS):")
        for transfer in self.config.SEARCH_DATA_TRANSFERS: print(f"    {transfer}")
        validation_data = self._generate_validation_data()
        print(f"\n### 検証用データセット ({len(validation_data)} 件) ###")
        for transfer in validation_data: print(f"  {transfer}")
        print("-" * 80)

        # --- 2. 全天候パターンの生成とループ実行 ---
        gs_names = list(self.ground_stations.keys())
        all_weather_combinations = list(itertools.product(self.config.WEATHER_CONDITIONS, repeat=len(gs_names)))
        print(f"\n上記データセットを用いて、全 {len(all_weather_combinations)} 通りの天候パターンで探索と検証を開始します。")
        results = {'optimal': {'avg_delay_per_mb': []}, 'nearest': {'avg_delay_per_mb': []}, 'priority': {'avg_delay_per_mb': []}}

        # 各天候パターンについてシミュレーションを逐次実行
        for scenario_idx, weather_combo in enumerate(all_weather_combinations):
            print(f"\n▶▶▶ 天候シナリオ {scenario_idx + 1}/{len(all_weather_combinations)} ◀◀◀")
            weather_details = []
            for gs_name, weather in zip(gs_names, weather_combo):
                self.ground_stations[gs_name].set_weather(weather[0], weather[1])
                rain_rate = weather[0]
                if rain_rate == 0.0: weather_status = "晴天"
                elif rain_rate == 3.0: weather_status = "雨天"
                elif rain_rate == 10.0: weather_status = "強い雨"
                else: weather_status = "不明な天候"
                weather_details.append(f"{gs_name}: {weather_status}")
            print(f"  現在の天候条件: {', '.join(weather_details)}")
            
            # --- 3. フェーズ1: 3つの手法でトポロジを構築 ---
            print("\n  --- [フェーズ1] 各手法によるトポロジ構築 ---")
            topologies = {
                'optimal': self._find_optimal_topology_for_scenario(),
                'nearest': self._construct_nearest_satellite_topology(),
                'priority': self._construct_priority_based_topology()
            }
            
            # --- 4. フェーズ2: 構築したトポロジを統一されたアルゴリズムで検証 ---
            print("\n  --- [フェーズ1 結果] 構築されたトポロジ（地上局-衛星リンク） ---")
            for method_name, topology in topologies.items():
                print(f"    --- {method_name.capitalize()}手法 ---")
                if topology:
                    for gs, sat in sorted(topology.items()): print(f"      {gs:<8} <--> SAT {sat}")
                else: print("      トポロジ構築不能")
            
            print("\n  --- [フェーズ2] 統一された経路制御による性能検証 ---")
            for method_name, topology in topologies.items():
                print(f"\n    --- 検証対象: {method_name.capitalize()}手法のトポロジ ---")
                if not topology:
                    print("      トポロジが構築不能のため、検証をスキップします。")
                    continue
                # 経路を事前計算
                paths_alpha = self._precompute_paths_on_fixed_topology(topology, self.config.ALPHA_AVG_SIZE_MB)
                paths_beta = self._precompute_paths_on_fixed_topology(topology, self.config.BETA_AVG_SIZE_MB)
                precomputed_paths = {'alpha': paths_alpha, 'beta': paths_beta}
                # (途中結果の表示)
                print("      事前計算された最適経路:")
                print("        - α (小サイズデータ)用経路:")
                if paths_alpha:
                    for dest, path in sorted(paths_alpha.items()): print(f"          Tokyo -> {dest:<7}: {path}")
                else: print("          経路なし")
                print("        - β (大サイズデータ)用経路:")
                if paths_beta:
                    for dest, path in sorted(paths_beta.items()): print(f"          Tokyo -> {dest:<7}: {path}")
                else: print("          経路なし")

                # 検証実行と結果の保存
                metrics = self._validate_with_precomputed_paths(precomputed_paths, validation_data, topology)
                if metrics:
                    results[method_name]['avg_delay_per_mb'].append(metrics['avg_delay_per_mb'])
            print("-" * 80)

        # --- 5. 最終結果の集計と表示 ---
        print("\n\n" + "="*58 + f"\n [{config_name}] の最終集計結果\n" + "="*58)
        print("\n### 手法ごとの最終評価 (全天候パターン平均) ###")
        self._print_final_results("提案手法 (最適化)", results['optimal'])
        self._print_final_results("比較手法1 (最近傍)", results['nearest'])
        self._print_final_results("比較手法2 (優先度付き)", results['priority'])
        print("="*58)

    def _print_final_results(self, method_name, result_data):
        """最終的な集計結果を整形して出力する。"""
        if result_data['avg_delay_per_mb']:
            final_avg_metric = sum(result_data['avg_delay_per_mb']) / len(result_data['avg_delay_per_mb'])
            print(f"\n  --- {method_name} ---")
            print(f"    平均遅延時間 (1MBあたり)\t: {final_avg_metric:.6f} [ms/MB]")

    @staticmethod
    def _build_full_network_graph_static(gs_sat_links, all_gs, constellation, config):
        """地上局-衛星リンクを含む完全なネットワークグラフを構築する静的メソッド。"""
        G = nx.Graph()
        gs_names_jp = ['Tokyo', 'Sendai', 'Hakodate']; gs_names_eu = ['London', 'Paris', 'Glasgow']
        G.add_nodes_from(constellation.sats_coords.keys()); G.add_nodes_from(all_gs.keys())
        gs_ecef_map = {name: gs.ecef_coords for name, gs in all_gs.items()}
        for u, v, data in constellation.graph.edges(data=True): G.add_edge(u, v, weight=data['weight'] / config.SPEED_OF_LIGHT_KM_S, type='isl')
        for region in [gs_names_jp, gs_names_eu]:
            for i in range(len(region)):
                for j in range(i + 1, len(region)):
                    u, v = region[i], region[j]; dist = np.linalg.norm(gs_ecef_map[u] - gs_ecef_map[v])
                    G.add_edge(u, v, weight=dist / config.SPEED_OF_LIGHT_KM_S, type='ground')
        for gs_name, sat_id in gs_sat_links.items():
            if gs_name in gs_ecef_map:
                dist = np.linalg.norm(gs_ecef_map[gs_name] - constellation.sats_coords[sat_id])
                G.add_edge(gs_name, sat_id, weight=dist / config.SPEED_OF_LIGHT_KM_S, type='access')
        return G

    def _precompute_paths_on_fixed_topology(self, fixed_topology, data_size_mb):
        """与えられたトポロジ上で、宛先ごとの最適経路を事前計算する。"""
        paths = {}
        source = 'Tokyo'
        search_dests = set(t['dest'].replace('Pairs', 'Paris') for t in self.config.SEARCH_DATA_TRANSFERS)
        G = self._build_full_network_graph_static(fixed_topology, self.ground_stations, self.constellation, self.config)
        for dest in search_dests:
            min_delay, best_path = float('inf'), None
            try: k_paths = list(itertools.islice(nx.shortest_simple_paths(G, source, dest, weight='weight'), self.config.K_SHORTEST_PATHS))
            except (nx.NetworkXNoPath, nx.NodeNotFound): continue
            for path in k_paths:
                single_transfer = [{'source': source, 'dest': dest, 'size_mb': data_size_mb}]
                delay = self.routing_engine._calculate_total_delay_new_formula(single_transfer, {0: path}, G, self.ground_stations, self.constellation)
                if delay < min_delay: min_delay, best_path = delay, path
            if best_path: paths[dest] = best_path
        return paths

    def _validate_with_precomputed_paths(self, precomputed_paths, validation_data, topology):
        """事前計算した経路を用いて、検証用データセットの性能を評価する。"""
        delays_per_mb = []
        size_threshold = (self.config.ALPHA_AVG_SIZE_MB + self.config.BETA_AVG_SIZE_MB) / 2.0
        G_eval = self._build_full_network_graph_static(topology, self.ground_stations, self.constellation, self.config)
        
        print("      --- 個別データ検証 ---")
        for i, transfer in enumerate(validation_data):
            dest, size_mb = transfer['dest'], transfer['size_mb']
            size_type = 'beta' if size_mb >= size_threshold else 'alpha'
            if dest in precomputed_paths[size_type] and size_mb > 0:
                path = precomputed_paths[size_type][dest]
                individual_delay = self.routing_engine._calculate_total_delay_new_formula([transfer], {0: path}, G_eval, self.ground_stations, self.constellation)
                print(f"        - Data[{i+1:02d}]: Size={size_mb: >5.2f} MB, Delay={individual_delay: >8.3f} ms")
                delays_per_mb.append(individual_delay / size_mb)
        if not delays_per_mb:
            print("      評価可能なデータがありませんでした。")
            return None
        avg_delay_per_mb = sum(delays_per_mb) / len(delays_per_mb)
        print("      ------------------------")
        print(f"      検証結果 -> 平均遅延時間 (1MBあたり): {avg_delay_per_mb:.6f} [ms/MB]")
        return {'avg_delay_per_mb': avg_delay_per_mb}
        
    def _find_optimal_topology_for_scenario(self):
        """【提案手法】最適化により、性能指標が最も良くなるトポロジを探索する。"""
        print("    提案手法のトポロジを探索中...")
        gs_candidate_sats = {name: [s_id for _, s_id in sorted([(np.linalg.norm(gs.ecef_coords - p), s_id) for s_id, p in self.constellation.sats_coords.items()])[:3]] for name, gs in self.ground_stations.items()}
        gs_names = list(self.ground_stations.keys())
        candidate_lists = [gs_candidate_sats[name] for name in gs_names]
        all_combos = [c for c in itertools.product(*candidate_lists) if len(set(c)) == len(gs_names)]
        if not all_combos: return None
        min_t_ave, best_topology = float('inf'), None
        for combo in all_combos:
            current_links = {name: sat_id for name, sat_id in zip(gs_names, combo)}
            t_ave = self.routing_engine.find_best_routing_for_search(current_links, self.config.SEARCH_DATA_TRANSFERS, self.ground_stations, self.constellation)
            if t_ave < min_t_ave: min_t_ave, best_topology = t_ave, current_links
        return best_topology

    def _construct_nearest_satellite_topology(self):
        """【比較手法1】各地上局を、単純に最も距離の近い衛星に接続する。"""
        print("    最近傍手法のトポロジを構築中...")
        return {name: min([(np.linalg.norm(gs.ecef_coords - p), s_id) for s_id, p in self.constellation.sats_coords.items()])[1] for name, gs in self.ground_stations.items()}

    def _construct_priority_based_topology(self):
        """【比較手法2】通信品質の悪い地上局から優先的に、重複しないように衛星を選択する。"""
        print("    優先度付き手法のトポロジを構築中...")
        initial_links = []
        for name, gs in self.ground_stations.items():
            _, nearest_sat_id = min([(np.linalg.norm(gs.ecef_coords - p), s_id) for s_id, p in self.constellation.sats_coords.items()])
            capacity = self.routing_engine._get_access_link_capacity_gbps(name, nearest_sat_id, self.ground_stations, self.constellation, True)
            initial_links.append({'gs': name, 'capacity': capacity})
        priority_list = [link['gs'] for link in sorted(initial_links, key=lambda x: x['capacity'])]
        final_topology, assigned_sats = {}, set()
        gs_candidate_sats = {name: [s_id for _, s_id in sorted([(np.linalg.norm(gs.ecef_coords - p), s_id) for s_id, p in self.constellation.sats_coords.items()])[:3]] for name, gs in self.ground_stations.items()}
        for gs_name in priority_list:
            best_sat, max_cap = None, -1
            for sat_id in gs_candidate_sats[gs_name]:
                if sat_id not in assigned_sats:
                    cap = self.routing_engine._get_access_link_capacity_gbps(gs_name, sat_id, self.ground_stations, self.constellation, True)
                    if cap > max_cap: max_cap, best_sat = cap, sat_id
            if best_sat:
                final_topology[gs_name] = best_sat
                assigned_sats.add(best_sat)
            else: 
                unassigned_sat = next((sat for sat in gs_candidate_sats[gs_name] if sat not in assigned_sats), gs_candidate_sats[gs_name][0])
                final_topology[gs_name] = unassigned_sat
                assigned_sats.add(unassigned_sat)
        return final_topology

# ==============================================================================
# 6. メイン実行ブロック
# ==============================================================================
if __name__ == "__main__":
    """
    プログラムのエントリーポイント（ここから実行が開始される）。
    複数の通信シナリオを定義し、ループで順番に実行する。
    """
    # 実行したいシナリオのIDと名前を辞書に定義
    scenarios_to_run = {
        1: "シナリオ3 (大容量データ中心)",
        2: "シナリオ4 (小容量データ中心)"
    }
    # 定義したシナリオをループで一つずつ実行
    for scenario_id, scenario_name in scenarios_to_run.items():
        # =======================================================================
        # <<< 変更箇所 2 >>>
        # ループの開始時に、出力先ファイルと標準出力のリダイレクトを設定。
        # =======================================================================
        
        # シナリオ名からOSで使えるファイル名を生成
        safe_scenario_name = scenario_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_filename = f"{SimulationConfig.OUTPUT_FILENAME_BASE}_{safe_scenario_name}.txt"
        
        # 元の標準出力（コンソール）を保存しておく
        original_stdout = sys.stdout
        
        # `with`構文でファイルを開き、このブロック内の標準出力をファイルに向ける
        with open(output_filename, 'w', encoding='utf-8') as f:
            sys.stdout = f  # printの出力先をファイルに変更

            print(f"### このシミュレーションの出力は {output_filename} に保存されています ###")
        
            print("\n" + "#"*80)
            print(f"# >>> 開始: {scenario_name}")
            print("#"*80)
            start_time = time.time()
            try:
                from RFrev2 import S2U_RF_throughput
                config = SimulationConfig(scenario_id=scenario_id)
                runner = SimulationRunner(config)
                runner.run_full_optimization_cycle(config_name=scenario_name)
            except ImportError:
                print("="*80 + "\nエラー: 外部モジュール 'RFrev.py' が見つかりません。\n" + "="*80)
                exit()
            except Exception as e:
                print(f"❌ 予期せぬエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
            finally:
                end_time = time.time()
                execution_time = end_time - start_time
                print("\n" + "="*58)
                print(f"✅ [{scenario_name}] の処理が完了しました。")
                print(f"⏱️  実行時間: {execution_time:.2f} 秒")
                print("="*58)

        # --- 標準出力先を元のコンソールに戻す ---
        sys.stdout = original_stdout
        # どのファイルに保存されたか、コンソールにメッセージを表示
        print(f"✅ シナリオ「{scenario_name}」が完了しました。結果は '{output_filename}' に保存されています。")
