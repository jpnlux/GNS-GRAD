# train_mappo.py
# test for git
import time
import numpy as np
import torch
import csv

from mappo_pz_wropper import MultiStationMAPPOEnv
from mappo.buffer import RolloutBufferMAPPO
from mappo.mappo_agent_a import MAPPOAgent


import csv

def train_mappo(
    total_episodes: int = 200,
    max_steps_per_episode: int | None = None,
    device: str = "cpu",
    log_csv: str = "train_log.csv",   # ← 追加
):
    env = MultiStationMAPPOEnv()

    num_agents = env.num_agents
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    global_state_dim = num_agents * obs_dim

    if max_steps_per_episode is None:
        max_steps_per_episode = getattr(env.pz_env.core, "num_snapshots", 120)

    # ===== ログファイル初期化 =====
    with open(log_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "reward_mean", "delay_mean"])

    agent = MAPPOAgent(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_state_dim=global_state_dim,
        device=device,
    )

    print("=== MAPPO Training Start ===")

    for episode in range(1, total_episodes + 1):
        obs = env.reset()
        buffer = RolloutBufferMAPPO(max_steps_per_episode, num_agents, obs_dim, global_state_dim)

        episode_reward = np.zeros(num_agents, dtype=np.float32)
        delay_mean = None

        for t in range(max_steps_per_episode):
            actions, logprobs, value, global_state = agent.act(obs)
            next_obs, rewards, done, info = env.step(actions)

            buffer.add(
                obs_t=obs,
                global_state_t=global_state,
                actions_t=actions,
                rewards_t=rewards,
                done_t=done,
                value_t=value,
                logprobs_t=logprobs,
                ho_flags_t=info["ho_flags"],   # ← これを忘れると MAPPO が HO を無視してしまう
            )


            episode_reward += rewards
            delay_mean = info.get("delay_mean", None)
            obs = next_obs

            if done:
                break

        last_value = agent.get_value(obs)
        rollout_data = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        log_dict = agent.update(rollout_data, num_epochs=5)

        mean_ep_reward = float(episode_reward.mean())

        print(f"[Episode {episode}] reward={mean_ep_reward:.4f}, delay={delay_mean}")

        # ===== ログ CSV に追記 =====
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, mean_ep_reward, delay_mean])

    print("=== Training Finished ===")
    return agent, env, max_steps_per_episode



def evaluate_policy(
    env,
    agent,
    max_steps_per_episode: int,
    n_eval_episodes: int = 10,
    mode: str = "trained",   # "random" or "trained"
    log_csv: str | None = None,
    summary_csv=None
):
    """
    mode:
      - "random"  : 環境の action space に対して一様ランダム
      - "trained" : MAPPO エージェントの方策（stochastic）をそのまま使う
                    （必要なら後で greedy に拡張）
    log_csv:
      - None でなければ、(action, delay) ログを CSV に書き出す
    """
    num_agents = env.num_agents
    action_dim = env.action_dim

    episode_rewards = []
    episode_delay_means = []

    logs = []  # action-delay ログをここに溜める

    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = np.zeros(num_agents, dtype=np.float32)
        ep_delay_sum = 0.0
        ep_steps = 0

        while not done and ep_steps < max_steps_per_episode:
            if mode == "random":
                # 各エージェント一様ランダム
                actions = np.random.randint(0, action_dim, size=num_agents)
            else:
                # 学習済み方策（stochastic）
                actions, _, _, _ = agent.act(obs)
                actions = actions.astype(int)

            next_obs, rewards, done, info = env.step(actions)

            # info から per-GS 遅延を取得（なければ None）
            delays = info.get("delays", None)

            # ログに記録
            if delays is not None:
                for gi in range(num_agents):
                    logs.append({
                        "mode": mode,
                        "episode": ep,
                        "step": ep_steps,
                        "agent": gi,
                        "action": int(actions[gi]),
                        "delay": float(delays[gi]),
                    })

            ep_reward += rewards
            ep_delay_sum += info.get("delay_mean", 0.0)
            ep_steps += 1
            obs = next_obs

        episode_rewards.append(ep_reward.mean())
        if ep_steps > 0:
            episode_delay_means.append(ep_delay_sum / ep_steps)
        else:
            episode_delay_means.append(np.nan)

    print(
        f"[EVAL-{mode}] episodes={n_eval_episodes}, "
        f"reward_mean={np.mean(episode_rewards):.4f}, "
        f"delay_mean={np.nanmean(episode_delay_means):.6f}"
    )

    # ログを CSV に保存
    if log_csv is not None and len(logs) > 0:
        fieldnames = ["mode", "episode", "step", "agent", "action", "delay"]
        with open(log_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in logs:
                writer.writerow(row)
        print(f"[EVAL-{mode}] action-delay log saved to {log_csv}")

   
    if summary_csv is not None:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["mode", "reward_mean", "delay_mean"])
            writer.writerow([mode, np.mean(episode_rewards), np.nanmean(episode_delay_means)])


    return np.mean(episode_rewards), np.nanmean(episode_delay_means)


def save_eval_csv(filename, history):
    """
    history: list of delay_mean (per episode)
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "delay_mean"])
        for ep, d in enumerate(history, start=1):
            writer.writerow([ep, d])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 学習
    agent, env, max_steps = train_mappo(
        total_episodes=100,
        max_steps_per_episode=None,
        device=device,
    )

    # 2) ランダム方策の評価
    eval_random_history = evaluate_policy(
        env=env,
        agent=agent,
        max_steps_per_episode=max_steps,
        n_eval_episodes=20,
        mode="random",
        log_csv="action_delay_random.csv",
    )

    # 3) 学習済み方策の評価
    eval_learned_history = evaluate_policy(
        env=env,
        agent=agent,
        max_steps_per_episode=max_steps,
        n_eval_episodes=20,
        mode="trained",
        log_csv="action_delay_trained.csv",
    )

    # 4) それぞれの評価結果を CSV に保存（plot 用）
    save_eval_csv("eval_random.csv", [eval_random_history])
    save_eval_csv("eval_learned.csv", [eval_learned_history])

    print("=== Training + Evaluation Finished ===")


"""
if __name__ == "__main__":
    # GPU があれば "cuda" にしても OK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_mappo(
        total_episodes=2,
        max_steps_per_episode=None,
        device=device,
    )
"""