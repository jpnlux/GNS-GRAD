# env_pz_wrapper.py

import numpy as np
from marl_parallel_env import MultiStationParallelEnv


class MultiStationMAPPOEnv:
    """
    MultiStationParallelEnv（PettingZoo ParallelEnv）を、
    MAPPO の学習コードから使いやすいように薄くラップしたクラス。

    - reset()  -> obs: np.ndarray shape=(N, obs_dim)
    - step(a)  -> obs_next: (N, obs_dim), rewards: (N,), done: bool, info: dict
    """

    def __init__(self):
        self.pz_env = MultiStationParallelEnv()
        self.agents = self.pz_env.agents
        self.num_agents = len(self.agents)

        # 1 つだけ取り出して obs_dim / action_dim を決定
        sample_core_obs, _ = self.pz_env.reset()
        first_agent = self.agents[0]
        obs_sample = sample_core_obs[first_agent]
        self.obs_dim = obs_sample.shape[-1]

        self.pz_env.core.reset()  # reset を戻しておく

        # 離散行動数
        self.action_dim = self.pz_env.action_spaces[first_agent].n

    def reset(self, seed=None):
        obs_dict, infos = self.pz_env.reset(seed=seed)
        obs = self._dict_obs_to_array(obs_dict)
        return obs  # shape = (N, obs_dim)

    def step(self, actions):
        """
        actions: np.ndarray shape=(N,) または list[int]
        """
        if isinstance(actions, np.ndarray):
            actions = actions.astype(int).tolist()

        action_dict = {
            agent: int(actions[i])
            for i, agent in enumerate(self.agents)
        }

        obs_dict, rew_dict, term_dict, trunc_dict, infos = self.pz_env.step(action_dict)

        obs_next = self._dict_obs_to_array(obs_dict)
        rewards = np.array([rew_dict[agent] for agent in self.agents], dtype=np.float32)

        # 今回は「全エージェント同時に終了」想定なので、どれか一つ True なら episode 終了
        done = any(term_dict.values()) or any(trunc_dict.values())

        # info は基本的に core_info（delay_mean など）が入っている
        # 共通部分だけ 1 つ返しておく
        core_info = infos[self.agents[0]] if len(self.agents) > 0 else {}

        return obs_next, rewards, done, core_info

    def _dict_obs_to_array(self, obs_dict):
        obs_list = [obs_dict[agent] for agent in self.agents]
        obs = np.stack(obs_list, axis=0).astype(np.float32)
        return obs
