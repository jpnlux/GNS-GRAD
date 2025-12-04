# ============================================================
# burst_compare_test.py
# 期間最初だけ隣接局からパケットが来る場合と来ない場合の
# キュー遅延を比較するテストコード
# ============================================================

from copy import deepcopy

# ---------------------------
# M/M/1 逐次シミュレーション
# ---------------------------
class MM1QueueSimulator:
    def __init__(self, mu_init=1.0, seed=0):
        self.queue = 0.0
        self.mu = mu_init
        self.rng = np.random.default_rng(seed)

    def update_service_rate(self, mu):
        self.mu = max(mu, 1e-9)

    def step(self, lam, dt=1.0):
        lam = max(lam, 0.0)
        mu = max(self.mu, 1e-9)

        arrivals = self.rng.poisson(lam * dt)
        services = self.rng.poisson(mu * dt)

        self.queue = max(0.0, self.queue + arrivals - services)

        # ✔ 遅延 = N / λ (Little's law)
        return self.queue / max(lam, 1e-9)



# ---------------------------
# シミュレーション処理
# ---------------------------
def run_snapshot(mu, lam, burst_packets, duration=10):
    """
    snapshot期間（duration秒）における遅延平均を返す
    burst_packets: 期間最初に到来する追加パケット数
    """

    q = MM1QueueSimulator(mu)
    delays = []

    for t in range(duration):
        if t == 0 and burst_packets > 0:
            q.queue += burst_packets   # ★ 期間開始時だけ追加

        d = q.step(lam)
        delays.append(d)

    return sum(delays) / len(delays)


# ---------------------------
# メインテスト
# ---------------------------
if __name__ == "__main__":

    # --- 例：リンク 100 Mbps、パケット 1Mbit ---
    mu = service_rate(100e6, 1e6)
    lam = 80

    burst_amount = 50  # 隣接衛星から一度に送られてくるパケット量

    print("=== スナップショット比較開始 ===")

    # --- ケース 1: バーストなし ---
    avg_no_burst = run_snapshot(mu, lam, burst_packets=0)

    # --- ケース 2: 期間最初だけ追加パケットあり ---
    avg_with_burst = run_snapshot(mu, lam, burst_packets=burst_amount)

    print("\n=== 結果 ===")
    print(f"期間最初のバースト無し平均遅延 : {avg_no_burst:.4f} sec")
    print(f"期間最初のバースト有り平均遅延 : {avg_with_burst:.4f} sec")
    print(f"バースト差分（増加量）           : {avg_with_burst - avg_no_burst:.4f} sec")
