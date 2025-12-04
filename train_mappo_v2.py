# train_mappo.py
import time
import numpy as np
import torch
import csv

from mappo_pz_wropper import MultiStationMAPPOEnv
# バッファとエージェントの実装がどこにあるかに依存しますが、
# 仮に mappo_agent.py や buffer.py がない場合、それらも必要です。
# 今回は "mappo.py" に全て入っていると仮定して修正します。
# もし mappo.py に Agent クラスがない場合は、別途 mappo_agent.py が必要です。

# アップロードされた mappo.py には Actor/Critic しかなかったため、
# おそらく MAPPOAgent や RolloutBufferMAPPO は別のファイルにあるか、
# 提供漏れの可能性があります。
# ここでは "mappo_agent.py", "mappo_buffer.py" がある前提、
# あるいは既存のインポートが正しい前提で進めます。
from mappo.buffer import RolloutBufferMAPPO
from mappo.mappo_agent_a import MAPPOAgent

def train_mappo(
    total_episodes: int = 200,
    max_steps_per_episode: int | None = None,
    device: str = "cpu",
    log_csv: str = "train_log.csv",
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
        # 遅延平均をエピソード全体で集計するためのリスト
        episode_delays = []

        for t in range(max_steps_per_episode):
            actions, logprobs, value, global_state = agent.act(obs)
            next_obs, rewards, done, info = env.step(actions)

            # infoからho_flagsを安全に取得
            ho_flags = info.get("ho_flags", np.zeros(num_agents))

            buffer.add(
                obs_t=obs,
                global_state_t=global_state,
                actions_t=actions,
                rewards_t=rewards,
                done_t=done,
                value_t=value,
                logprobs_t=logprobs,
                ho_flags_t=ho_flags, 
            )

            episode_reward += rewards
            
            d = info.get("delay_mean", None)
            if d is not None:
                episode_delays.append(d)
            
            obs = next_obs

            if done:
                break

        last_value = agent.get_value(obs)
        rollout_data = buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=agent.gamma,
            gae_lambda=agent.gae_lambda,
        )

        agent.update(rollout_data, num_epochs=5)

        mean_ep_reward = float(episode_reward.mean())
        # エピソード全体の平均遅延を計算
        mean_ep_delay = np.mean(episode_delays) if episode_delays else 0.0

        print(f"[Episode {episode}] reward={mean_ep_reward:.4f}, delay={mean_ep_delay:.4f}")

        # ===== ログ CSV に追記 =====
        with open(log_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, mean_ep_reward, mean_ep_delay])

    print("=== Training Finished ===")
    return agent, env, max_steps_per_episode


def evaluate_policy(
    env,
    agent,
    max_steps_per_episode: int,
    n_eval_episodes: int = 10,
    mode: str = "trained",
    log_csv: str | None = None,
    summary_csv=None
):
    """
    戻り値: (reward_mean_all, delay_mean_all, episode_delay_history)
    episode_delay_history: 各エピソードの平均遅延のリスト [ep1_delay, ep2_delay, ...]
    """
    num_agents = env.num_agents
    action_dim = env.action_dim

    episode_rewards = []
    episode_delay_means = [] # 各エピソードごとの平均遅延を格納

    logs = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        done = False
        ep_reward = np.zeros(num_agents, dtype=np.float32)
        
        # 1エピソード内のステップ毎の遅延
        step_delays = []
        ep_steps = 0

        while not done and ep_steps < max_steps_per_episode:
            if mode == "random":
                actions = np.random.randint(0, action_dim, size=num_agents)
            else:
                actions, _, _, _ = agent.act(obs)
                actions = actions.astype(int)

            next_obs, rewards, done, info = env.step(actions)

            # 詳細ログ用 (per-GS delay)
            # marl_core の info に "delays" (list of float) が入っている前提
            # もし入っていなければ info['last_delays'] か info.get('delay_mean') 等で代用
            current_gs_delays = info.get("delays", None)
            
            if log_csv is not None and current_gs_delays is not None:
                for gi in range(num_agents):
                    logs.append({
                        "mode": mode,
                        "episode": ep,
                        "step": ep_steps,
                        "agent": gi,
                        "action": int(actions[gi]),
                        "delay": float(current_gs_delays[gi]),
                    })

            ep_reward += rewards
            
            d_mean = info.get("delay_mean", 0.0)
            step_delays.append(d_mean)
            
            ep_steps += 1
            obs = next_obs

        # エピソード終了処理
        episode_rewards.append(ep_reward.mean())
        
        if len(step_delays) > 0:
            episode_delay_means.append(np.mean(step_delays))
        else:
            episode_delay_means.append(0.0)

    # 全エピソードの総平均
    avg_reward = np.mean(episode_rewards)
    avg_delay = np.nanmean(episode_delay_means)

    print(
        f"[EVAL-{mode}] episodes={n_eval_episodes}, "
        f"reward_mean={avg_reward:.4f}, "
        f"delay_mean={avg_delay:.6f}"
    )

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
            writer.writerow([mode, avg_reward, avg_delay])

    # 修正: 3つの値を返す
    return avg_reward, avg_delay, episode_delay_means


def save_eval_csv(filename, delay_history):
    """
    delay_history: list of float (各エピソードの平均遅延)
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "delay_mean"])
        for ep, d in enumerate(delay_history, start=1):
            writer.writerow([ep, d])
    print(f"Saved evaluation history to {filename}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 学習
    # エピソード数はテスト用に少なめに設定しても良い
    agent, env, max_steps = train_mappo(
        total_episodes=10, 
        max_steps_per_episode=None,
        device=device,
    )

    # 2) ランダム方策の評価
    # 戻り値のアンパックを修正
    _, _, eval_random_history = evaluate_policy(
        env=env,
        agent=agent,
        max_steps_per_episode=max_steps,
        n_eval_episodes=10,
        mode="random",
        log_csv="action_delay_random.csv",
    )

    # 3) 学習済み方策の評価
    _, _, eval_learned_history = evaluate_policy(
        env=env,
        agent=agent,
        max_steps_per_episode=max_steps,
        n_eval_episodes=10,
        mode="trained",
        log_csv="action_delay_trained.csv",
    )

    # 4) それぞれの評価結果を CSV に保存
    # リストのリストではなく、フラットなリストを渡す
    save_eval_csv("eval_random.csv", eval_random_history)
    save_eval_csv("eval_learned.csv", eval_learned_history)

    print("=== Training + Evaluation Finished ===")