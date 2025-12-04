# mappo/mappo_agent_a.py

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, obs_dim)
        logits = self.net(x)
        return logits


class CriticNet(nn.Module):
    def __init__(self, global_state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, global_state_dim)
        v = self.net(x)
        return v  # (B, 1)


class MAPPOAgent:
    """
    パラメータ共有 Actor + 中央 Critic の MAPPO 実装。

    - 各ステップ、各エージェントは act() で行動を出す
    - ただし、HO が起きていない局の advantage は 0 にマスクし、
      その局の行動は学習に使わない（勾配が流れない）
    """

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        global_state_dim: int,
        device: str = "cpu",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm

        self.actor = ActorNet(obs_dim, action_dim).to(self.device)
        self.critic = CriticNet(global_state_dim).to(self.device)

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.dist = torch.distributions.Categorical

    # ---------------------------------------------------------
    # Action selection
    # ---------------------------------------------------------
    def act(self, obs: np.ndarray):
        """
        obs : shape (N, obs_dim) の numpy
        戻り値:
          actions: shape (N,)
          logprobs: shape (N,)
          values: shape (N,)
          global_state: shape (num_agents * obs_dim,)  (単純 concat)
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)  # (N, obs_dim)

        # グローバル状態を単純に concat で作る
        global_state = obs.reshape(-1).astype(np.float32)  # (N * obs_dim,)
        global_state_tensor = (
            torch.from_numpy(global_state).float().to(self.device).unsqueeze(0)
        )  # (1, N*obs_dim)

        # Actor
        logits = self.actor(obs_tensor)             # (N, action_dim)
        dist = self.dist(logits=logits)
        actions_t = dist.sample()                   # (N,)
        logprobs_t = dist.log_prob(actions_t)       # (N,)

        # Critic（中央）
        values_t = self.critic(global_state_tensor).squeeze(0)  # (N,) ではなく (1,) にしてもOK
        # 今回は「全体の価値」を各エージェント共通として扱うなら：
        # values_t = self.critic(global_state_tensor).view(-1)   # (1,)
        # v_per_agent = values_t.repeat(self.num_agents)         # (N,)
        # とするパターンもあるが、ここでは 1値を N にブロードキャスト
        if values_t.numel() == 1:
            v_per_agent = values_t.view(1).repeat(self.num_agents)
        else:
            v_per_agent = values_t  # 既に (N,) ならそのまま

        actions = actions_t.detach().cpu().numpy()
        logprobs = logprobs_t.detach().cpu().numpy()
        values = v_per_agent.detach().cpu().numpy()

        return actions, logprobs, values, global_state

    def get_value(self, obs: np.ndarray):
        """
        終端状態の価値を計算する。
        ここでは簡単のため、obs から global_state を作って Critic に通す。
        戻り値: shape (N,)
        """
        global_state = obs.reshape(-1).astype(np.float32)
        global_state_tensor = (
            torch.from_numpy(global_state).float().to(self.device).unsqueeze(0)
        )
        v = self.critic(global_state_tensor).view(-1)  # (1,) or (N,)
        if v.numel() == 1:
            v = v.repeat(obs.shape[0])
        return v.detach().cpu().numpy()

    # ---------------------------------------------------------
    # Update (PPO / MAPPO)
    # ---------------------------------------------------------
    def update(self, rollout_data: dict, num_epochs: int = 5, batch_size: int | None = None):
        """
        rollout_data: buffer.compute_returns_and_advantages() の出力。
        advantages に ho_flags を掛けて、HO の起きなかった局の勾配は 0 にする。
        """

        obs = rollout_data["obs"]                # (T, N, obs_dim)
        global_states = rollout_data["global_states"]  # (T, global_state_dim)
        actions = rollout_data["actions"]        # (T, N)
        logprobs_old = rollout_data["logprobs"]  # (T, N)
        returns = rollout_data["returns"]        # (T, N)
        values_old = rollout_data["values"]      # (T, N)
        advantages = rollout_data["advantages"]  # (T, N)
        ho_flags = rollout_data["ho_flags"]      # (T, N) 0 or 1

        T, N, obs_dim = obs.shape
        device = self.device

        # flatten T, N
        obs_flat = torch.from_numpy(obs.reshape(T * N, obs_dim)).float().to(device)
        global_states_flat = torch.from_numpy(global_states).float().to(device)  # (T, G)
        actions_flat = torch.from_numpy(actions.reshape(T * N)).long().to(device)
        logprobs_old_flat = torch.from_numpy(logprobs_old.reshape(T * N)).float().to(device)
        returns_flat = torch.from_numpy(returns.reshape(T * N)).float().to(device)
        values_old_flat = torch.from_numpy(values_old.reshape(T * N)).float().to(device)
        advantages_flat = torch.from_numpy(advantages.reshape(T * N)).float().to(device)
        ho_flags_flat = torch.from_numpy(ho_flags.reshape(T * N)).float().to(device)

        # HOが起きていないサンプルは、advantage を 0 にして勾配を止める
        advantages_masked = advantages_flat * ho_flags_flat

        if batch_size is None:
            batch_size = T * N  # 全件一括

        num_samples = T * N
        idxs = np.arange(num_samples)

        actor_loss_epoch = 0.0
        value_loss_epoch = 0.0

        for _ in range(num_epochs):
            np.random.shuffle(idxs)

            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]
                if len(mb_idx) == 0:
                    continue

                mb_obs = obs_flat[mb_idx]                   # (B, obs_dim)
                mb_actions = actions_flat[mb_idx]           # (B,)
                mb_logprobs_old = logprobs_old_flat[mb_idx] # (B,)
                mb_advantages = advantages_masked[mb_idx]   # (B,)
                mb_returns = returns_flat[mb_idx]           # (B,)
                mb_values_old = values_old_flat[mb_idx]     # (B,)
                mb_ho_flags = ho_flags_flat[mb_idx]         # (B,)

                # ===== Actor 更新 =====
                logits = self.actor(mb_obs)
                dist = self.dist(logits=logits)
                logprobs = dist.log_prob(mb_actions)        # (B,)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logprobs - mb_logprobs_old)  # (B,)

                # クリッピング付き PPO Loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
                )
                # HOの起きていないサンプルは mb_advantages=0 のため自然と寄与しない
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ===== Critic 更新：ここでは ho_flags を使わず全体で更新 =====
                # （もし HO局のみで学習したければ mb_returns, mb_values に mb_ho_flags を掛けてもよい）
                # 中央値なので global_states_flat を使う。
                # 簡単には、T 個の global_state それぞれから 1 つの V を出して、
                # それを各エージェントにコピーするといった設計もありうるので、
                # ここはあなたの設計に合わせて調整して良い。
                # ここでは global_states_flat を T×G として扱い、バッチに対応する t を復元する簡易版とする。
                # （厳密には indices から t を計算して対応させる必要がある）

                # ここでは簡単のため、value を「行列の形で再計算」してしまう：
                #   V_all: (T, 1) -> flatten -> (T*N,) に repeat しておき、
                #   そこからミニバッチを抜き出す実装。
                with torch.no_grad():
                    v_all = self.critic(global_states_flat).view(T)  # (T,)
                    v_all_rep = v_all.unsqueeze(1).repeat(1, N).reshape(T * N)  # (T*N,)
                v_all_rep = v_all_rep.to(device)
                mb_values = v_all_rep[mb_idx]

                value_loss_unclipped = (mb_values - mb_returns) ** 2
                v_clipped = mb_values + (mb_values - mb_values_old[mb_idx]).clamp(
                    -self.clip_eps, self.clip_eps
                )
                value_loss_clipped = (v_clipped - mb_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                loss = actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer_actor.step()
                self.optimizer_critic.step()

                actor_loss_epoch += actor_loss.item()
                value_loss_epoch += value_loss.item()

        # 平均を返す
        log_dict = {
            "actor_loss": actor_loss_epoch / num_epochs,
            "value_loss": value_loss_epoch / num_epochs,
        }
        return log_dict
