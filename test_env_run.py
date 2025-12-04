import numpy as np
from marl_parallel_env import MARLParallelEnv


def main():
    env = MARLParallelEnv()

    obs, infos = env.reset()
    print("Initial agents:", env.agents)
    print("Initial obs shape (agent_0):", obs[env.agents[0]].shape)

    max_steps = 40

    for step in range(max_steps):
        # ランダム行動
        actions = {
            agent: env.action_spaces[agent].sample()
            for agent in env.agents
        }

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\n--- step {step} ---")
        print("actions:", actions)
        print("rewards:", rewards)
        print("terminated:", terminations)

        # 代表として agent_0 の info を表示（全部見たければ infos そのまま print）
        any_agent = env.agents[0]
        print("info:", infos[any_agent])

        # どれか一つでも True ならエピソード終了とみなす
        if any(terminations.values()) or any(truncations.values()):
            break


if __name__ == "__main__":
    main()
