import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sumo_rl


class QLearningAgent:
    def __init__(
        self,
        state_space,
        action_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    ):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(action_space.n))
        self.acc_reward = 0

    def act(self, state):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        return int(np.argmax(self.q_table[state]))

    def learn(self, state, action, reward, next_state, done):
        current_q = self.q_table[state][action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error
        self.acc_reward += reward

        return td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def run_ql_experiment():
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    runs = 1
    episodes = 5

    env = sumo_rl.parallel_env(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=80000,
        out_csv_name="outputs/4x4grid_ql",
    )

    for run in range(1, runs + 1):
        print(f"\n=== Run {run}/{runs} ===")

        agents = {
            ts: QLearningAgent(
                state_space=env.observation_space(ts),
                action_space=env.action_space(ts),
                alpha=alpha,
                gamma=gamma,
                epsilon=epsilon,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
            )
            for ts in env.possible_agents
        }

        for ep in tqdm(range(1, episodes + 1), desc="Episodes", unit="ep"):
            observations, infos = env.reset()

            done = {ts: False for ts in env.possible_agents}
            episode_rewards = {ts: 0 for ts in env.possible_agents}

            step_pbar = tqdm(desc="Steps", unit="step", leave=False)

            while env.agents:
                actions = {
                    ts: agents[ts].act(str(observations[ts]))
                    for ts in env.agents
                    if ts in observations
                }

                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                for ts in next_obs.keys():
                    agents[ts].learn(
                        str(observations[ts]),
                        actions[ts],
                        rewards[ts],
                        str(next_obs[ts]),
                        terminations[ts] or truncations[ts],
                    )
                    episode_rewards[ts] += rewards[ts]
                    done[ts] = terminations[ts] or truncations[ts]

                observations = next_obs
                step_pbar.update(1)

            step_pbar.close()

            for agent in agents.values():
                agent.decay_epsilon()

            total_reward = sum(episode_rewards.values())
            eps = list(agents.values())[0].epsilon if agents else 0
            tqdm.write(f"Episode {ep}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {eps:.4f}")

    env.close()


if __name__ == "__main__":
    run_ql_experiment()