# ==============================================================
# marl_core.py  ― MultiStationCore (Full Rewrite)
# ==============================================================

import numpy as np
from topover4 import run_ho_simulation, y_orbit_A, y_orbit_B, z_sat
from traffic import TrafficPattern
from channel.rf_channel import rf_capacity_bps

C_LIGHT = 299_792_458.0
DEBUG = True


# ==============================================================
# M/M/1 Queue simulator
# ==============================================================
class MM1QueueSimulator:
    def __init__(self, mu_init=1.0, seed=0):
        self.queue = 0.0
        self.mu = mu_init
        self.rng = np.random.default_rng(seed)

    def update_service_rate(self, mu):
        self.mu = max(mu, 1e-9)

    def step(self, lam, dt):
        lam = max(lam, 0.0)
        mu = max(self.mu, 1e-9)

        arrivals = self.rng.poisson(lam * dt)
        services = self.rng.poisson(mu * dt)

        self.queue = max(0.0, self.queue + arrivals - services)

        if lam <= 0:
            return 0.0
        return self.queue / lam  # Little's Law W ≒ N/λ


# ==============================================================
# Weather model
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
# MultiStationCore (rewritten)
# ==============================================================
class MultiStationCore:

    def __init__(
        self,
        num_links_total=9,
        pkt_bits=8_000_000,
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
        dt_max=0.5,
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

        # queue update microstep
        self.dt_max = dt_max
        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------
        # State variables
        # ----------------------------------------------
        self.links_A = np.zeros(self.num_gs, int)
        self.links_B = np.zeros(self.num_gs, int)
        self.capacity_A = np.zeros(self.num_gs)
        self.capacity_B = np.zeros(self.num_gs)

        # queue per GS (A/B)
        self.queues_A = [
            MM1QueueSimulator(seed=seed + 1000 + gi) for gi in range(self.num_gs)
        ]
        self.queues_B = [
            MM1QueueSimulator(seed=seed + 2000 + gi) for gi in range(self.num_gs)
        ]

        # buffers
        self.last_delays = np.zeros(self.num_gs)
        self.delay_prop = np.zeros(self.num_gs)
        self.delay_tx = np.zeros(self.num_gs)
        self.delay_q = np.zeros(self.num_gs)

        self.current_step = 0
        self.last_time = 0.0

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
    # ISL delay
    # ==========================================================
    def _isl_onehop_delay(self):
        tx = self.pkt_bits / self.isl_capacity_bps
        prop = self.isl_hop_dist_m / C_LIGHT
        return tx + prop

    def _compute_isl_hops(self, snapshot, orbit, old_sat, gi):
        """
        HOしたGS gi の old_sat から、
        隣接GS neighbor が保持する衛星 drop_sat までの
        ISL ホップ数を計算する。
        """

        neighbor = self._get_neighbor_gs(gi)
        drop_sat = snapshot["connections"][neighbor][orbit]

        sat_dict = snapshot["sat_positions_x"][orbit]
        n_sat = len(sat_dict)

        # --- 最短距離 hop ---
        diff = abs(drop_sat - old_sat)
        hops = min(diff, n_sat - diff)

        return hops


    # ==========================================================
    # Observation (new 15D)
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

            # traffic
            lam_total = self.traffic.lambda_for_sat(gi, self.mode)
            lam_A = lam_total * (self.capacity_A[gi] /
                                 max(self.capacity_A[gi] + self.capacity_B[gi], 1e-9))
            lam_B = lam_total - lam_A

            lam_total_n = lam_total / 2000.0
            lam_A_n = lam_A / 2000.0
            lam_B_n = lam_B / 2000.0

            # queue
            qA_n = self.queues_A[gi].queue / 10000.0
            qB_n = self.queues_B[gi].queue / 10000.0

            # delays
            prop_n = self.delay_prop[gi] / 0.02
            tx_n = self.delay_tx[gi] / 0.02
            qd_n = self.delay_q[gi] / 0.1

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
                    prop_n, tx_n, qd_n,
                    visA, visB,
                    ho_flag
                ], dtype=float)
            )
        return obs_list

    # ==========================================================
    # reset
    # ==========================================================
    def reset(self):
        self.current_step = 0

        for q in self.queues_A:
            q.queue = 0.0
        for q in self.queues_B:
            q.queue = 0.0

        self.delay_prop[:] = 0.0
        self.delay_tx[:] = 0.0
        self.delay_q[:] = 0.0
        self.last_delays[:] = 0.0

        snap = self.snapshots[0]
        return self._build_obs(snap)

    # ==========================================================
    # step (HO-aware)
    # ==========================================================
    def step(self, actions):

        # --------------------------------------------------
        # 1. Snapshot load
        # --------------------------------------------------
        snap = self.snapshots[self.current_step]
        t_now = snap["time"]

        # ΔT
        if self.current_step == 0:
            dt_snapshot = 1e-6
        else:
            dt_snapshot = max(t_now - self.snapshots[self.current_step - 1]["time"], 1e-9)

        # microstep
        if self.dt_max:
            n_sub = int(np.ceil(dt_snapshot / self.dt_max))
            dt_sub = dt_snapshot / n_sub
        else:
            n_sub = 1
            dt_sub = dt_snapshot

        # HO event
        ho = snap["ho_event"]
        gi_ho = ho["gs"] if ho else None
        orbit_ho = ho["orbit"] if ho else None

        # --------------------------------------------------
        # 2. update RF capacity
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
        # 3. Apply actions only to HO GS
        # --------------------------------------------------
        acts = np.asarray(actions, int)

        for gi in range(self.num_gs):
            if gi != gi_ho:
                continue
            a = acts[gi]

            if DEBUG:
                print(f"[HO] GS{gi} orbit={orbit_ho}, action={acts[gi]}")
                print(f"[LINK BEFORE] GS{gi}: A={self.links_A[gi]}, B={self.links_B[gi]}")
                    
            if orbit_ho == "A":
                # fix B
                oldB = self.links_B[gi]
                maxA = self.num_links_total - oldB
                newA = int(np.clip(a, 1, maxA))
                self.links_A[gi] = newA
                self.links_B[gi] = oldB
            else:
                oldA = self.links_A[gi]
                maxB = self.num_links_total - oldA
                newB = int(np.clip(a, 1, maxB))
                self.links_B[gi] = newB
                self.links_A[gi] = oldA

        # --------------------------------------------------
        # 4. HO extra delay (ISL)
        # --------------------------------------------------
        ho_extra = np.zeros(self.num_gs)
        if ho :#and self.current_step > 0:
            old_sat = ho["old_sat"]
            hops = self._compute_isl_hops(snap, orbit_ho, old_sat, gi_ho)
            hops = min(hops, 20) 
            d1 = self._isl_onehop_delay()

            lam_ho = self.traffic.lambda_for_sat(gi_ho, self.mode)
            q_ho = self.queues_A[gi_ho] if orbit_ho == "A" else self.queues_B[gi_ho]

            factor = q_ho.queue / max(lam_ho * dt_snapshot, 1e-9) if lam_ho > 0 else 0
            factor = 1 #min(factor, 10.0)
            ho_extra[gi_ho] = d1 * hops * factor

            if DEBUG:
                print(f"[ISL] GS{gi_ho} ho_extra={ho_extra[gi_ho]:.8f}")

            # move queue to neighbor
            neighbor = self._get_neighbor_gs(gi_ho)
            q_dest = self.queues_A[neighbor] if orbit_ho == "A" else self.queues_B[neighbor]
            moved = q_ho.queue
            q_dest.queue += moved
            q_ho.queue = 0.0


        # --------------------------------------------------
        # 5. Fixed propagation delay
        # --------------------------------------------------
        prop_delay = 800e3 / C_LIGHT

        # --------------------------------------------------
        # 6. Delay calculation per GS
        # --------------------------------------------------
        for gi in range(self.num_gs):

            capA = max(self.capacity_A[gi] * self.links_A[gi], 0.0)
            capB = max(self.capacity_B[gi] * self.links_B[gi], 0.0)

            lam_total = self.traffic.lambda_for_sat(gi, self.mode)

            if lam_total <= 0:
                self.delay_prop[gi] = prop_delay
                self.delay_tx[gi] = 0.0
                self.delay_q[gi] = 0.0
                self.last_delays[gi] = prop_delay + ho_extra[gi]
                continue

            cap_sum = max(capA + capB, 1e-9)
            lam_A = lam_total * capA / cap_sum
            lam_B = lam_total - lam_A

            mu_A = capA / self.pkt_bits if capA > 0 else 1e-9
            mu_B = capB / self.pkt_bits if capB > 0 else 1e-9

            W_A = 0.0
            W_B = 0.0

            for _ in range(n_sub):
                if capA > 0:
                    self.queues_A[gi].update_service_rate(mu_A)
                    W_A = self.queues_A[gi].step(lam_A, dt_sub)
                if capB > 0:
                    self.queues_B[gi].update_service_rate(mu_B)
                    W_B = self.queues_B[gi].step(lam_B, dt_sub)

            W_sys = (lam_A * W_A + lam_B * W_B) / lam_total

            txA = self.pkt_bits / capA if capA > 0 else 0
            txB = self.pkt_bits / capB if capB > 0 else 0
            tx_delay = (lam_A * txA + lam_B * txB) / lam_total

            total = W_sys + tx_delay + prop_delay + ho_extra[gi]

            self.delay_q[gi] = W_sys
            self.delay_tx[gi] = tx_delay
            self.delay_prop[gi] = prop_delay
            self.last_delays[gi] = total

        # --------------------------------------------------
        # 7. Reward
        # --------------------------------------------------
        D = self.last_delays
        global_delay = float(np.mean(D))
        r_global = -global_delay

        rewards = np.ones(self.num_gs) * (self.beta_global * r_global)
        r_local = np.zeros(self.num_gs)

        if ho and self.current_step > 0:
            r_local[gi_ho] = -D[gi_ho]

        rewards += self.alpha_local * r_local

        # --------------------------------------------------
        # 8. Next observation
        # --------------------------------------------------
        self.current_step += 1
        done = self.current_step >= self.num_steps
        next_snap = snap if done else self.snapshots[self.current_step]

        obs = self._build_obs(next_snap)

        # --------------------------------------------------
        # 9. Info
        # --------------------------------------------------
        ho_flags = np.zeros(self.num_gs)
        if ho and self.current_step > 0:
            ho_flags[gi_ho] = 1.0

        isl_delay_mean = float(np.mean(ho_extra))
        isl_delay_max = float(np.max(ho_extra))

        info = dict(
            delay_mean=global_delay,
            r_global=r_global,
            r_local=r_local,
            prop_delay=float(prop_delay),
            isl_delay_mean=isl_delay_mean,
            isl_delay_max=isl_delay_max,
            #isl_delay=float(np.mean(ho_extra)),
            ho_flags=ho_flags,
        )

        return obs, rewards, done, info




def _apply_ho_link_change(self, gi, orbit):
    """
    HO が発生した地上局 gi のリンク本数を更新する。
    orbit: "A" or "B"
    """

    total = self.num_links_total

    if orbit == "A":
        # A軌道のリンクが HO → A側を再配置
        other = self.links_B[gi]
        remain = total - other

        if remain <= 0:
            newA = 0
        else:
            newA = np.random.randint(1, remain + 1)

        self.links_A[gi] = newA
        self.links_B[gi] = total - newA

    else:  # orbit == "B"
        other = self.links_A[gi]
        remain = total - other

        if remain <= 0:
            newB = 0
        else:
            newB = np.random.randint(1, remain + 1)

        self.links_B[gi] = newB
        self.links_A[gi] = total - newB

    if DEBUG:
        print(f"[HO-LINK-CHANGE] GS{gi} orbit={orbit}: "
              f"remain={remain}, A={self.links_A[gi]}, B={self.links_B[gi]}")



# ==============================================================
# M/M/1 sanity check helper
# ==============================================================

def mm1_sanity_check(
    lam: float = 50.0,
    mu: float = 100.0,
    dt: float = 0.001,
    steps: int = 1_000_000,
    seed: int = 123,
):
    """
    M/M/1 キューの sanity check 用関数。
    ・理論値 E[W] = 1 / (μ - λ) と
    ・シミュレーション値を比較する。
    """
    q = MM1QueueSimulator(mu_init=mu, seed=seed)

    delays = []
    for _ in range(steps):
        w = q.step(lam, dt)
        delays.append(w)

    sim_EW = float(np.mean(delays))
    theo_EW = 1.0 / (mu - lam)

    print("=== M/M/1 sanity check ===")
    print(f"lambda = {lam:.1f} [pkt/s], mu = {mu:.1f} [pkt/s]")
    print(f"theoretical E[W] = {theo_EW:.4f} [s]")
    print(f"simulated  E[W] = {sim_EW:.4f} [s]")


if __name__ == "__main__":
    # 直接実行したときだけ sanity check と簡単な動作確認を行う
    mm1_sanity_check()

    core = MultiStationCore(num_links_total=9, mode="high", dt_max=0.5)

    obs = core.reset()

    actions = np.random.randint(0, core.num_links_total + 1, size=core.num_gs)


    for i in range(core.num_steps):
        print(f"\n================ STEP {i} ================")
        obs, reward, done, info = core.step(actions)
        print(
            f"[RESULT] mean_delay = {info['delay_mean']:.6f} s, "
            f"prop_delay = {info['prop_delay']:.6f} s, "
            f"isl_delay_max = {info['isl_delay_max']:.6f} s, "
            f"reward_mean = {reward.mean():.6f}"
        )
        if done:
            print("[DEBUG] DONE: reached final snapshot")
            break
