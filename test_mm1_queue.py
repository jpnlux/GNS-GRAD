# test_mm1_queue.py

import numpy as np
from marl_core import MM1QueueSimulator

def main():
    # λ, μ を適当に安定系に設定（μ > λ）
    lam = 50.0   # packet/sec
    mu  = 100.0  # packet/sec
    dt  = 0.01   # sec
    steps = 200_000

    q = MM1QueueSimulator(mu_init=mu, seed=0)

    delays = []

    for t in range(steps):
        # μ は固定なので update_service_rate は本当は不要だが、
        # コア環境と同じ使い方にしておく
        q.update_service_rate(mu)
        W = q.step(lam, dt=dt)  # 平均遅延の推定量
        delays.append(W)

    sim_mean_delay = float(np.mean(delays[int(steps*0.2):]))  # 前半を捨てて定常部だけ
    theo_mean_delay = 1.0 / (mu - lam)  # M/M/1 の理論値 E[W] = 1/(μ - λ)

    print("=== M/M/1 sanity check ===")
    print(f"lambda = {lam} [pkt/s], mu = {mu} [pkt/s]")
    print(f"theoretical E[W] = {theo_mean_delay:.4f} [s]")
    print(f"simulated  E[W] = {sim_mean_delay:.4f} [s]")

if __name__ == "__main__":
    main()
