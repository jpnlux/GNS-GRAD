import csv
import matplotlib.pyplot as plt

episodes = []
rewards = []
delays = []

with open("train_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        episodes.append(int(row["episode"]))
        rewards.append(float(row["reward_mean"]))
        delays.append(float(row["delay_mean"]))

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(episodes, rewards)
plt.title("Episode Reward")
plt.xlabel("Episode")

plt.subplot(1,2,2)
plt.plot(episodes, delays)
plt.title("Mean Delay")
plt.xlabel("Episode")

plt.tight_layout()
plt.savefig("train_curve.png")
plt.show()
