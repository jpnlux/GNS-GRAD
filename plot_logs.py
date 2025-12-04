import pandas as pd
import matplotlib.pyplot as plt

# ===== 1. 学習曲線 =====
df = pd.read_csv("train_log.csv")

plt.figure(figsize=(10,4))
plt.plot(df["episode"], df["delay_mean"], label="Delay")
plt.xlabel("Episode")
plt.ylabel("Mean Delay")
plt.title("Training Progress (Delay)")
plt.grid()
plt.legend()
plt.savefig("training_delay_curve.png")
plt.close()

# ===== 2. ランダム vs 学習 =====
r = pd.read_csv("eval_random.csv")
t = pd.read_csv("eval_learned.csv")

labels = ["Random", "Trained"]
delays = [r["delay_mean"][0], t["delay_mean"][0]]

plt.figure(figsize=(6,4))
plt.bar(labels, delays)
plt.ylabel("Mean Delay")
plt.title("Policy Comparison")
plt.savefig("policy_comparison.png")
plt.close()

print("Graphs saved: training_delay_curve.png, policy_comparison.png")
