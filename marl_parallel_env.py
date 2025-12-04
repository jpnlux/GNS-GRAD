# marl_parallel_env.py

import numpy as np
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces

from marl_core import MultiStationCore  # コア環境（遅延・報酬計算など）

class MultiStationParallelEnv(ParallelEnv):
    """
    MultiStationCore を PettingZoo ParallelEnv 形式にしたラッパ。
    - エージェント: agent_0 ... agent_{N-1}
    - action: 各エージェントの「軌道Aに割り当てるリンク本数」(0..num_links_total)
    - obs: core.reset() / core.step() が返す観測（長さ obs_dim のベクトル）
           [links_A_norm, links_B_norm, capA_Gbps, capB_Gbps, last_delay, is_ho]
           などを想定（実際の中身は marl_core 側に依存）
    """

    metadata = {
        "render_modes": [],
        "name": "LEO_multi_station_marl",
    }

    def __init__(self, core: MultiStationCore | None = None):
        super().__init__()

        # Core 環境
        self.core = core if core is not None else MultiStationCore()

        # PettingZoo の仕様上、num_agents というプロパティ名が予約されているので
        # 内部用に _num_agents を定義
        self._num_agents = self.core.num_gs
        self.agents = [f"agent_{i}" for i in range(self._num_agents)]

        # ----- observation / action space -----
        obs_dim = 6  # [linksA, linksB, capA, capB, last_delay, is_ho] を想定

        # 適当な上限（容量は数十Gbpsまでは想定）
        high = np.array([1.0, 1.0, 50.0, 50.0, 1e4, 1.0], dtype=np.float32)
        low = np.zeros_like(high)

        self._observation_spaces = {
            agent: spaces.Box(low=low, high=high, dtype=np.float32)
            for agent in self.agents
        }

        # 行動は 0..num_links_total の離散値
        n_actions = self.core.num_links_total + 1
        self._action_spaces = {
            agent: spaces.Discrete(n_actions)
            for agent in self.agents
        }

    # -------------- 必須プロパティ --------------

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def observation_spaces(self):
        return self._observation_spaces

    @property
    def action_spaces(self):
        return self._action_spaces

    # -------------- reset --------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            # 必要なら core 側に seed を渡す処理をここに書く
            np.random.seed(seed)

        # core.reset() は list[np.ndarray] (len = num_gs) を返す想定
        obs_list = self.core.reset()

        obs = {
            agent: np.asarray(obs_list[i], dtype=np.float32)
            for i, agent in enumerate(self.agents)
        }

        infos = {agent: {} for agent in self.agents}
        return obs, infos

    # -------------- step --------------

    def step(self, actions):
        """
        actions: {agent_i: int, ...}
        PettingZoo ParallelEnv 仕様では、done 状態のエージェントも dict key に含める。
        今回は「全員同時に done」なので単純に全員から行動を取る前提。
        """

        # MultiStationCore.step が「list[int]」を受け取る前提
        act_list = []
        for i, agent in enumerate(self.agents):
            a = actions.get(agent, 0)
            act_list.append(int(a))

        next_obs_list, rew_array, done, core_info = self.core.step(act_list)

        # dict 化
        obs = {
            agent: np.asarray(next_obs_list[i], dtype=np.float32)
            for i, agent in enumerate(self.agents)
        }
        rewards = {
            agent: float(rew_array[i])
            for i, agent in enumerate(self.agents)
        }
        terminations = {
            agent: bool(done)
            for agent in self.agents
        }
        truncations = {
            agent: False
            for agent in self.agents
        }

        infos = {
            agent: core_info.copy()
            for agent in self.agents
        }

        # done 時も self.agents は維持（SB3 互換 wrapper などが扱いやすい）
        return obs, rewards, terminations, truncations, infos

    def render(self):
        # 必要なら delay_mean の推移などを print/plot しても良い
        pass

    def close(self):
        pass


# 互換用エイリアス
MARLParallelEnv = MultiStationParallelEnv
