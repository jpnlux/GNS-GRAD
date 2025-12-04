# ==============================================================
# marl_core.py  ― MultiStationCore (Packet-Level Simulation)
# ==============================================================

import numpy as np
import collections
from topover4 import run_ho_simulation, y_orbit_A, y_orbit_B, z_sat
from traffic import TrafficPattern
from channel.rf_channel import rf_capacity_bps

# 定数
C_LIGHT = 299_792_458.0
DEBUG = False  # デバッグ出力制御

# ==============================================================
# Packet Definition
# ==============================================================
class Packet:
    """個々のパケットを表すクラス"""
    __slots__ = ['id', 'creation_time', 'size_bits']
    
    def __init__(self, pkt_id, creation_time, size_bits):
        self.id = pkt_id
        self.creation_time = creation_time  # 生成時刻 (絶対時刻)
        self.size_bits = size_bits          # パケットサイズ (bit)

# ==============================================================
# Packet Queue Simulator (Infinite Buffer)
# ==============================================================
class PacketQueueSimulator:
    """
    パケットレベルのキューシミュレータ
    - dequeによるFIFO管理
    - 無限バッファ (ドロップなし)
    """
    def __init__(self):
        self.buffer = collections.deque() # パケット格納用 (FIFO)
        self.busy_until = 0.0             # リンクが空く予定時刻 (絶対時刻)

    def add_packets(self, packets):
        """パケットリストをキューに追加"""
        self.buffer.extend(packets)

    def process(self, current_time, dt, capacity_bps):
        """
        現在のタイムステップ [current_time, current_time + dt] の間に
        送信完了できるパケットを処理し、それらの遅延リストを返す。
        """
        served_delays = []
        
        # 容量がほぼ0なら送信不可 (時間は進むがパケットは減らない)
        if capacity_bps <= 1e-9:
            # busy_until は更新せず、ただ何も処理しない
            return served_delays

        # シミュレーション期間の終了時刻
        end_time = current_time + dt
        
        # もしサーバがアイドル状態（現在時刻より前に空いていた）なら、現在時刻から開始
        # これにより、無通信期間のギャップを埋める
        sim_time_cursor = max(self.busy_until, current_time)

        # バッファの先頭から順に処理
        # ※ バッファ内のパケットは破壊的変更(pop)を行う
        while len(self.buffer) > 0:
            pkt = self.buffer[0] # 先頭参照 (まだpopしない)
            
            # このパケットの送信開始時刻
            # (パケット生成時刻 か リンクが空く時刻 の遅い方)
            start_service_time = max(sim_time_cursor, pkt.creation_time)
            
            # 送信所要時間 (Transmission Delay)
            tx_duration = pkt.size_bits / capacity_bps
            
            # 送信完了予定時刻
            finish_time = start_service_time + tx_duration
            
            # このステップ期間内に送信完了するか？
            if finish_time <= end_time:
                # 完了するのでキューから取り出す
                self.buffer.popleft()
                
                # 遅延計算: 完了時刻 - 生成時刻 (Queuing + Transmission)
                delay = finish_time - pkt.creation_time
                served_delays.append(delay)
                
                # リンクの空き時刻を更新
                sim_time_cursor = finish_time
                self.busy_until = finish_time
            else:
                # このステップ内では完了しない (送信中状態で終了)
                # busy_until を更新してループを抜ける
                # 次のステップでもこのパケットは先頭に残り、
                # その時の容量で残りの送信時間が再計算されるような挙動になる
                # (簡易実装として、busy_untilを未来にセットすることで予約状態にする)
                self.busy_until = finish_time
                break
                
        return served_delays
    
    @property
    def packet_count(self):
        return len(self.buffer)


# ==============================================================
# Weather model (Placeholder if not imported)
# ==============================================================
class SimpleWeatherModel:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def get_weather(self, sat_pos_km, gs_pos_km):
        if sat_pos_km is None:
            return 0.0, 0.0
        rain = float(self.rng.uniform(0.0, 10.0))
        cloud = float(self.rng.uniform(0.0, 1.0))
        return rain, cloud


# ==============================================================
# MultiStationCore (Rewritten for Packet Level)
# ==============================================================
class MultiStationCore:

    def __init__(
        self,
        num_links_total=9,
        pkt_bits=8_000_000, # 8Mb
        low_rate=40,
        high_rate=150,
        mode="high",
        seed=0,
        rf_freq_ghz=19.0,
        rf_bw_hz=50e6,
        rf_pt_dbw=10 * np.log10(60.0),
        rf_gt_dbi=40.0,
        rf_gr_dbi=40.0,
        rf_N0_dbw_per_hz=-205.6,
        rf_params=None,
        isl_capacity_bps=5e9,
        isl_hop_dist_m=500e3,
        alpha_local=0.2,
        beta_global=1.0,
        dt_max=0.5, # パケットレベルではスナップショット間隔をこの刻みで処理する
    ):
        # ----------------------------------------------
        # Simulation data (snapshots)
        # ----------------------------------------------
        self.snapshots, self.gs_positions = run_ho_simulation()
        self.num_gs = len(self.gs_positions)
        self.num_steps = len(self.snapshots)

        # ----------------------------------------------
        # Core parameters
        # ----------------------------------------------
        self.num_links_total = num_links_total
        self.pkt_bits = pkt_bits
        self.mode = mode
        self.traffic = TrafficPattern(low_rate, high_rate)

        # RF parameters
        self.rf_freq_ghz = rf_freq_ghz
        self.rf_bw_hz = rf_bw_hz
        self.rf_pt_dbw = rf_pt_dbw
        self.rf_gt_dbi = rf_gt_dbi
        self.rf_gr_dbi = rf_gr_dbi
        self.rf_N0_dbw_per_hz = rf_N0_dbw_per_hz
        self.rf_params = rf_params or {
            "k_rain": 0.08,
            "alpha_rain": 1.0,
            "rain_path_km": 5.0,
            "k_cloud": 0.05,
            "cloud_path_km": 5.0,
            "atm_const_dB": 0.5,
        }

        # FSO / ISL
        self.isl_capacity_bps = isl_capacity_bps
        self.isl_hop_dist_m = isl_hop_dist_m

        # reward parameters
        self.alpha_local = alpha_local
        self.beta_global = beta_global

        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------
        # State variables
        # ----------------------------------------------
        self.links_A = np.zeros(self.num_gs, int)
        self.links_B = np.zeros(self.num_gs, int)
        self.capacity_A = np.zeros(self.num_gs)
        self.capacity_B = np.zeros(self.num_gs)

        # Packet Queues (Replaced MM1Queue with PacketQueueSimulator)
        self.queues_A = [PacketQueueSimulator() for _ in range(self.num_gs)]
        self.queues_B = [PacketQueueSimulator() for _ in range(self.num_gs)]

        # Time Management
        self.current_time = 0.0
        self.global_pkt_id = 0

        # Buffers for observation
        self.last_delays = np.zeros(self.num_gs) # 直近ステップの平均遅延
        self.delay_prop = np.zeros(self.num_gs)
        self.delay_tx = np.zeros(self.num_gs) # 観測用(計算値)
        self.delay_q = np.zeros(self.num_gs)  # 観測用(待ちパケット数などを反映)

        self.current_step = 0
        
        self.weather = SimpleWeatherModel(seed)

        self._init_allocation()
        self._init_link_capacities()

    # ==========================================================
    # Utility functions
    # ==========================================================
    def _gs_pos_km(self, gi):
        return np.array(self.gs_positions[gi], float)

    def _sat_pos_km(self, snapshot, orbit, sat_idx):
        if sat_idx is None:
            return None
        x = snapshot["sat_positions_x"][orbit][sat_idx]
        y = y_orbit_A if orbit == "A" else y_orbit_B
        return np.array([x, y, z_sat], float)

    def _get_neighbor_gs(self, gi):
        return (gi + 1) % self.num_gs

    # ==========================================================
    # Packet Generation
    # ==========================================================
    def _generate_packets(self, lam, dt):
        """
        到着率 lam [pkt/s] で dt 秒間に発生するパケットリストを生成
        到着間隔は指数分布に従う (Poisson Process)
        """
        packets = []
        if lam <= 0:
            return packets

        t_cursor = self.current_time
        end_time = self.current_time + dt
        
        while True:
            # 次のパケットまでの間隔
            interval = self.rng.exponential(1.0 / lam)
            t_cursor += interval
            
            if t_cursor >= end_time:
                break
                
            self.global_pkt_id += 1
            # パケット生成 (ID, 絶対時刻, サイズ)
            pkt = Packet(self.global_pkt_id, t_cursor, self.pkt_bits)
            packets.append(pkt)
            
        return packets

    # ==========================================================
    # Initialization
    # ==========================================================
    def _init_allocation(self):
        for gi in range(self.num_gs):
            a = np.random.randint(1, self.num_links_total - 1)
            self.links_A[gi] = a
            self.links_B[gi] = self.num_links_total - a

    def _compute_cap_rf(self, sat_pos, gs_pos, rain, cloud):
        return float(
            rf_capacity_bps(
                sat_pos,
                gs_pos,
                rain,
                cloud,
                freq_ghz=self.rf_freq_ghz,
                bw_hz=self.rf_bw_hz,
                pt_dbw=self.rf_pt_dbw,
                gt_dbi=self.rf_gt_dbi,
                gr_dbi=self.rf_gr_dbi,
                N0_dbw_per_hz=self.rf_N0_dbw_per_hz,
                params=self.rf_params,
            )
        )

    def _init_link_capacities(self):
        snap = self.snapshots[0]
        for gi in range(self.num_gs):
            gs = self._gs_pos_km(gi)

            idxA = snap["connections"][gi]["A"]
            idxB = snap["connections"][gi]["B"]

            satA = self._sat_pos_km(snap, "A", idxA)
            satB = self._sat_pos_km(snap, "B", idxB)

            rainA, cloudA = self.weather.get_weather(satA, gs)
            rainB, cloudB = self.weather.get_weather(satB, gs)

            self.capacity_A[gi] = self._compute_cap_rf(satA, gs, rainA, cloudA)
            self.capacity_B[gi] = self._compute_cap_rf(satB, gs, rainB, cloudB)

    # ==========================================================
    # ISL delay helper
    # ==========================================================
    def _isl_onehop_delay(self):
        tx = self.pkt_bits / self.isl_capacity_bps
        prop = self.isl_hop_dist_m / C_LIGHT
        return tx + prop

    def _compute_isl_hops(self, snapshot, orbit, old_sat, gi):
        neighbor = self._get_neighbor_gs(gi)
        drop_sat = snapshot["connections"][neighbor][orbit]
        sat_dict = snapshot["sat_positions_x"][orbit]
        n_sat = len(sat_dict)
        diff = abs(drop_sat - old_sat)
        hops = min(diff, n_sat - diff)
        return hops

    # ==========================================================
    # Observation (15D)
    # ==========================================================
    def _build_obs(self, snapshot):
        ho = snapshot["ho_event"]
        ho_gs = ho["gs"] if ho else None

        obs_list = []

        for gi in range(self.num_gs):
            # link allocation
            a_norm = self.links_A[gi] / self.num_links_total
            b_norm = self.links_B[gi] / self.num_links_total

            # capacity
            capA_norm = self.capacity_A[gi] / 1e9
            capB_norm = self.capacity_B[gi] / 1e9

            # traffic estimate for obs
            lam_total = self.traffic.lambda_for_sat(gi, self.mode)
            # 現在の配分比率に基づく予測
            cap_sum = max(self.capacity_A[gi] + self.capacity_B[gi], 1e-9)
            lam_A = lam_total * (self.capacity_A[gi] / cap_sum)
            lam_B = lam_total - lam_A

            lam_total_n = lam_total / 2000.0
            lam_A_n = lam_A / 2000.0
            lam_B_n = lam_B / 2000.0

            # queue length (packet count)
            # パケット数だと大きくなりうるので適当に正規化
            qA_n = self.queues_A[gi].packet_count / 1000.0
            qB_n = self.queues_B[gi].packet_count / 1000.0

            # delays (直近ステップの実測値など)
            prop_n = self.delay_prop[gi] / 0.02
            # パケットシミュレーションでは tx と queue を分離して保持していないため
            # last_delays (total) を分解するか、あるいは観測用には近似値を入れる
            # ここでは last_delays を使う
            total_d_n = self.last_delays[gi] / 0.5 
            
            # visibility (0/1)
            idxA = snapshot["connections"][gi]["A"]
            idxB = snapshot["connections"][gi]["B"]
            visA = 1.0 if idxA is not None else 0.0
            visB = 1.0 if idxB is not None else 0.0

            # HO flag
            ho_flag = 1.0 if gi == ho_gs else 0.0

            obs_list.append(
                np.array([
                    a_norm, b_norm,
                    capA_norm, capB_norm,
                    lam_total_n, lam_A_n, lam_B_n,
                    qA_n, qB_n,
                    prop_n, total_d_n, 0.0, # qd_n は統合されたので0埋め等の調整が必要
                    visA, visB,
                    ho_flag
                ], dtype=float)
            )
        return obs_list

    # ==========================================================
    # Reset
    # ==========================================================
    def reset(self):
        self.current_step = 0
        self.current_time = 0.0
        self.global_pkt_id = 0

        # キューのリセット (新しいdequeを作成)
        self.queues_A = [PacketQueueSimulator() for _ in range(self.num_gs)]
        self.queues_B = [PacketQueueSimulator() for _ in range(self.num_gs)]

        self.delay_prop[:] = 0.0
        self.delay_tx[:] = 0.0
        self.delay_q[:] = 0.0
        self.last_delays[:] = 0.0

        snap = self.snapshots[0]
        return self._build_obs(snap)

    # ==========================================================
    # Step (Packet-Level Event Driven)
    # ==========================================================
    def step(self, actions):

        # --------------------------------------------------
        # 1. Snapshot load & Time Delta
        # --------------------------------------------------
        snap = self.snapshots[self.current_step]
        t_snapshot = snap["time"]

        # 時間進行の決定
        # 初回は微小時間、以降はスナップショット間の差分
        if self.current_step == 0:
            dt = 1e-6
        else:
            prev_t = self.snapshots[self.current_step - 1]["time"]
            dt = max(t_snapshot - prev_t, 1e-9)

        # --------------------------------------------------
        # 2. Update RF Capacity (Weather)
        # --------------------------------------------------
        for gi in range(self.num_gs):
            gs = self._gs_pos_km(gi)
            idxA = snap["connections"][gi]["A"]
            idxB = snap["connections"][gi]["B"]
            satA = self._sat_pos_km(snap, "A", idxA)
            satB = self._sat_pos_km(snap, "B", idxB)
            
            rainA, cloudA = self.weather.get_weather(satA, gs)
            rainB, cloudB = self.weather.get_weather(satB, gs)
            
            self.capacity_A[gi] = self._compute_cap_rf(satA, gs, rainA, cloudA)
            self.capacity_B[gi] = self._compute_cap_rf(satB, gs, rainB, cloudB)

        # --------------------------------------------------
        # 3. Apply Actions (HO logic)
        # --------------------------------------------------
        ho = snap["ho_event"]
        gi_ho = ho["gs"] if ho else None
        orbit_ho = ho["orbit"] if ho else None
        
        acts = np.asarray(actions, int)

        for gi in range(self.num_gs):
            # HO局のみアクション適用 (リンク数の変更)
            if gi != gi_ho:
                continue
            
            a = acts[gi]
            if orbit_ho == "A":
                # fix B, change A
                oldB = self.links_B[gi]
                maxA = self.num_links_total - oldB
                newA = int(np.clip(a, 1, maxA))
                self.links_A[gi] = newA
                self.links_B[gi] = oldB
            else:
                # fix A, change B
                oldA = self.links_A[gi]
                maxB = self.num_links_total - oldA
                newB = int(np.clip(a, 1, maxB))
                self.links_B[gi] = newB
                self.links_A[gi] = oldA

        # --------------------------------------------------
        # 4. HO Handling: Packet Transfer
        # --------------------------------------------------
        # HOが発生した場合、切断される衛星に残っているパケットは
        # ISL経由で隣接局のバッファへ転送される
        ho_isl_delay_mean = 0.0
        
        if ho and self.current_step > 0:
            old_sat = ho["old_sat"]
            # ISLホップ数計算
            hops = self._compute_isl_hops(snap, orbit_ho, old_sat, gi_ho)
            hops = min(hops, 20)
            
            # 1ホップあたりのISL遅延
            d_isl_unit = self._isl_onehop_delay()
            total_isl_delay = d_isl_unit * hops

            # 移動元と移動先のキューを特定
            if orbit_ho == "A":
                q_src = self.queues_A[gi_ho]
                q_dest = self.queues_A[self._get_neighbor_gs(gi_ho)]
            else:
                q_src = self.queues_B[gi_ho]
                q_dest = self.queues_B[self._get_neighbor_gs(gi_ho)]
            
            # パケット移動 (Buffer Transfer)
            # srcにある全パケットを取り出し、destの末尾に追加
            # 各パケットの creation_time はいじらない（累積遅延に自然に加算されるため）
            # ただし、ISL転送にかかる物理時間分だけ creation_time をマイナスするか、
            # あるいは busy_until を後ろ倒しにするかだが、
            # ここでは「パケットが瞬時に移動するが、ISL遅延分だけ余計に時間がかかった」とみなすため、
            # 追加的な処理はせず、パケットが宛先キューで処理されるのを待つ。
            # ISL遅延を明示的にRewardに反映させたい場合、パケットの送信完了時刻にゲタを履かせる等の処理が必要。
            # 今回はシンプルに、「バッファが移動して、そこで処理待ちになる」ことで遅延が増えるモデルとする。
            
            # ※ ISL遅延を明示的に加算する場合、Packetに 'extra_delay' 属性を持たせる手もあるが、
            # ここではHOペナルティとして固定値をReward計算時に考慮する方式をとるか、
            # あるいは単純にキューが混むことによる遅延増加に任せる。
            # ユーザ要望の「累積遅延」観点では、転送自体に時間がかかるはず。
            # 簡易実装: 移動するパケット全ての creation_time を total_isl_delay 分だけ過去にずらす(=遅延が増える)
            # これにより、送信完了時の (finish - creation) が増大する。
            
            moving_packets = list(q_src.buffer)
            q_src.buffer.clear()
            q_src.busy_until = self.current_time # リセット
            
            for pkt in moving_packets:
                pkt.creation_time -= total_isl_delay # 遅延かさ増し
            
            q_dest.add_packets(moving_packets)
            
            ho_isl_delay_mean = total_isl_delay # ログ用

            if DEBUG:
                print(f"[HO] GS{gi_ho} Orbit{orbit_ho}: Moved {len(moving_packets)} pkts to neighbor. ISL delay added: {total_isl_delay:.5f}")

        # --------------------------------------------------
        # 5. Packet Processing Loop
        # --------------------------------------------------
        # 各GSでパケット生成 -> 振り分け -> 処理
        
        step_served_delays = [] # このステップで送信完了した全パケットの遅延
        prop_delay_const = 800e3 / C_LIGHT # 固定伝搬遅延

        # パケット処理開始時刻
        t_start = self.current_time

        for gi in range(self.num_gs):
            
            # (A) パケット生成
            lam_total = self.traffic.lambda_for_sat(gi, self.mode)
            new_packets = self._generate_packets(lam_total, dt)
            
            # (B) A/Bへの振り分け
            # 容量比率に応じて振り分ける (Probabilistic Routing)
            capA = self.capacity_A[gi] * self.links_A[gi]
            capB = self.capacity_B[gi] * self.links_B[gi]
            total_cap = capA + capB
            
            pkts_A = []
            pkts_B = []
            
            if total_cap > 1e-9:
                prob_A = capA / total_cap
                # 乱数で振り分け
                rands = self.rng.random(len(new_packets))
                for i, pkt in enumerate(new_packets):
                    if rands[i] < prob_A:
                        pkts_A.append(pkt)
                    else:
                        pkts_B.append(pkt)
            else:
                # リンク容量ゼロなら、とりあえずAに入れて詰まらせる (あるいはランダム)
                pkts_A = new_packets

            # (C) キューへ追加
            self.queues_A[gi].add_packets(pkts_A)
            self.queues_B[gi].add_packets(pkts_B)
            
            # (D) パケット処理 (送信シミュレーション)
            # 返り値は [delay1, delay2, ...] (sec)
            delays_A = self.queues_A[gi].process(t_start, dt, capA)
            delays_B = self.queues_B[gi].process(t_start, dt, capB)
            
            # (E) 遅延集計
            # PacketQueueSimulatorが返す遅延は (Queue + Tx) なので、Prop を足す
            gs_served_delays = []
            for d in delays_A + delays_B:
                total_d = d + prop_delay_const
                gs_served_delays.append(total_d)
                step_served_delays.append(total_d)

            # 観測用: このGSでの平均遅延を記録 (完了パケットがなければ直近値を維持 or 0)
            if gs_served_delays:
                self.last_delays[gi] = np.mean(gs_served_delays)
            # 伝搬遅延記録
            self.delay_prop[gi] = prop_delay_const

        # --------------------------------------------------
        # 6. Reward Calculation
        # --------------------------------------------------
        # 全体平均遅延
        if len(step_served_delays) > 0:
            global_delay = np.mean(step_served_delays)
        else:
            # パケットが一つも完了しなかった場合
            # (負荷が低い、あるいは容量不足で詰まっている)
            # ペナルティとして直近の観測値や固定値を与えるか、0にするか
            # ここでは0にしておく(遅延なし=良いこと、ではないが完了なしなので評価不能)
            global_delay = 0.0 

        r_global = -global_delay
        
        rewards = np.ones(self.num_gs) * (self.beta_global * r_global)
        
        # ローカル報酬 (HOしたGSだけ、自分の遅延悪化をペナルティとするなど)
        # ここではシンプルに全員にグローバル遅延ベースを与える設定を維持
        # 必要なら self.last_delays[gi] を使って個別報酬を追加可能
        r_local = np.zeros(self.num_gs)
        if ho and self.current_step > 0:
             r_local[gi_ho] = -self.last_delays[gi_ho]
        
        rewards += self.alpha_local * r_local

        # --------------------------------------------------
        # 7. Advance Steps & Build Next Obs
        # --------------------------------------------------
        self.current_step += 1
        # 時間を進める
        self.current_time += dt
        
        done = self.current_step >= self.num_steps
        next_snap = snap if done else self.snapshots[self.current_step]

        obs = self._build_obs(next_snap)

        # --------------------------------------------------
        # 8. Info
        # --------------------------------------------------
        ho_flags = np.zeros(self.num_gs)
        if ho and self.current_step > 0:
            ho_flags[gi_ho] = 1.0

        info = dict(
            delay_mean=float(global_delay),
            r_global=r_global,
            prop_delay=float(prop_delay_const),
            isl_delay_added=float(ho_isl_delay_mean),
            ho_flags=ho_flags,
            total_packets_in_queues=sum(q.packet_count for q in self.queues_A + self.queues_B)
        )

        return obs, rewards, done, info

if __name__ == "__main__":
    # Test Run
    core = MultiStationCore(num_links_total=9, mode="high", dt_max=0.5)
    obs = core.reset()
    actions = np.random.randint(0, core.num_links_total + 1, size=core.num_gs)

    print("=== Packet-Level Simulation Test ===")
    for i in range(10):
        obs, reward, done, info = core.step(actions)
        print(f"Step {i}: Reward={reward[0]:.4f}, Delay={info['delay_mean']:.4f}s, QueuedPkts={info['total_packets_in_queues']}")
        if done: break