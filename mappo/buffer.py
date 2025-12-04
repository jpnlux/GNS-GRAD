from __future__ import annotations
import numpy as np


class RolloutBufferMAPPO:
    """
    MAPPO 用のロールアウトバッファ（中央価値・パラメータ共有型）。
    各 step, 各 agent の以下を保存する：
      - obs
      - global_state
      - actions
      - rewards
      - done
      - values
      - logprobs
      - ho_flags : HO が起きた agent のみ 1, それ以外 0
    """

    def __init__(
        self,
        num_steps: int,
        num_agents: int,
        obs_dim: int,
        global_state_dim: int,
    ):
        self.num_steps = num_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.global_state_dim = global_state_dim

        self.step_ptr = 0
        self.full = False

        self.obs = np.zeros((num_steps, num_agents, obs_dim), dtype=np.float32)
        self.global_states = np.zeros(
            (num_steps, global_state_dim), dtype=np.float32
        )
        self.actions = np.zeros((num_steps, num_agents), dtype=np.int64)
        self.rewards = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.dones = np.zeros((num_steps,), dtype=np.float32)
        self.values = np.zeros((num_steps, num_agents), dtype=np.float32)
        self.logprobs = np.zeros((num_steps, num_agents), dtype=np.float32)

        # HO フラグ（0 or 1）: shape = (T, N)
        self.ho_flags = np.zeros((num_steps, num_agents), dtype=np.float32)

    def add(
        self,
        obs_t: np.ndarray,              # shape (N, obs_dim)
        global_state_t: np.ndarray,     # shape (global_state_dim,)
        actions_t: np.ndarray,          # shape (N,)
        rewards_t: np.ndarray,          # shape (N,)
        done_t: bool,
        value_t: np.ndarray,            # shape (N,)
        logprobs_t: np.ndarray,         # shape (N,)
        ho_flags_t: np.ndarray | None = None,  # shape (N,), 0/1
    ):
        i = self.step_ptr
        if i >= self.num_steps:
            return

        self.obs[i] = obs_t
        self.global_states[i] = global_state_t
        self.actions[i] = actions_t
        self.rewards[i] = rewards_t
        self.dones[i] = float(done_t)
        self.values[i] = value_t
        self.logprobs[i] = logprobs_t

        if ho_flags_t is None:
            self.ho_flags[i] = 0.0
        else:
            self.ho_flags[i] = ho_flags_t.astype(np.float32)

        self.step_ptr += 1
        if self.step_ptr >= self.num_steps:
            self.full = True

    def compute_returns_and_advantages(
        self,
        last_value: np.ndarray,  # shape (N,)
        gamma: float,
        gae_lambda: float,
    ):
        """
        GAE を用いて returns と advantages を計算。
        last_value はエピソード終端時の V(s_T)
        """
        T = self.step_ptr
        N = self.num_agents

        values = np.vstack([self.values[:T], last_value.reshape(1, N)])  # (T+1, N)
        rewards = self.rewards[:T]        # (T, N)
        dones = self.dones[:T]            # (T,)
        ho_flags = self.ho_flags[:T]      # (T, N)

        advantages = np.zeros((T, N), dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):
            delta = (
                rewards[t]
                + gamma * values[t + 1] * (1.0 - dones[t])
                - values[t]
            )
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + self.values[:T]

        # advantage を正規化
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        rollout_data = {
            "obs": self.obs[:T],                     # (T, N, obs_dim)
            "global_states": self.global_states[:T], # (T, G)
            "actions": self.actions[:T],             # (T, N)
            "rewards": rewards,                      # (T, N)
            "dones": dones,                          # (T,)
            "values": self.values[:T],               # (T, N)
            "logprobs": self.logprobs[:T],           # (T, N)
            "returns": returns,                      # (T, N)
            "advantages": advantages,                # (T, N)
            "ho_flags": ho_flags,                    # (T, N)
        }
        return rollout_data
