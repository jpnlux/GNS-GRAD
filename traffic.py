# traffic_model.py
import math

C_LIGHT = 299_792_458.0  # 伝搬速度（光速）

# ---------------------------
# ★ M/M/1 キュー（定常平均遅延）
# ---------------------------
class MM1Queue:
    def __init__(self, lam, mu):
        if lam >= mu:
            raise ValueError(f"λ={lam} μ={mu} -> 発散してしまう (λ < μ 必須)")
        self.lam = lam
        self.mu = mu

    def mean_delay(self):
        """定常平均遅延 E[T] = 1/(μ − λ)"""
        return 1.0 / (self.mu - self.lam)

# ---------------------------
# ★ M/M/1 キュー（逐次シミュレーション版：各時間ステップの遅延を計算）
# ---------------------------
class MM1QueueSimulator:
    """
    M/M/1 キューの簡易シミュレータ（パケット単位）
    queue : パケット数 E[N] を連続値で持つ
    μ     : サービス率 [packet/sec]
    """
    def __init__(self, mu_init=1.0, seed=0):
        self.queue = 0.0
        self.mu = mu_init
        self.rng = np.random.default_rng(seed)

    def update_service_rate(self, mu):
        """HO や天候で容量 cap が変わったときに呼ぶ"""
        self.mu = max(mu, 1e-9)

    def step(self, lam, dt=1.0):
        """
        lam : 到着率 λ [packet/sec]
        dt  : ステップ幅 [sec]

        ・Poisson 到着 / サービスで queue（パケット数）を更新
        ・1 ステップあたりの平均遅延 W を返す
        """
        lam = max(lam, 0.0)
        mu = max(self.mu, 1e-9)

        # ← せきさんする版（Poisson）
        arrivals = self.rng.poisson(lam * dt)
        services = self.rng.poisson(mu * dt)

        self.queue = max(0.0, self.queue + arrivals - services)

        if mu <= 0.0:
            return 0.0

        # W ≒ N / μ で平均遅延[s] を近似
        return self.queue / mu

# ---------------------------
# ★ サービス率 μ（リンク容量とパケットサイズで決める）
# ---------------------------
def service_rate(capacity_bps, pkt_bits):
    """
    capacity_bps : リンク容量 [bps]
    pkt_bits     : 1パケットのサイズ [bit]
    """
    return capacity_bps / pkt_bits

# ---------------------------
# ★ 伝搬遅延（距離から計算）
# ---------------------------
def propagation_delay(distance_m):
    return distance_m / C_LIGHT

# ---------------------------
# ★ 到着率 λ の設定（低負荷 / 高負荷）
# ---------------------------
class TrafficPattern:
    def __init__(self, low_rate, high_rate):
        self.low = low_rate
        self.high = high_rate

    def lambda_for_sat(self, sat_index, mode="low"):
        if mode == "low":
            return self.low
        elif mode == "high":
            return self.high
        else:
            raise ValueError("mode は 'low' または 'high'")

#---------
class Packet:
    def __init__(self, pkt_id, creation_time, size_bits):
        self.id = pkt_id
        self.creation_time = creation_time  # 生成時刻 (絶対時刻)
        self.size_bits = size_bits          # サイズ (bit)