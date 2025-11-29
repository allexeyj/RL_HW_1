import os
import sys
from tqdm.notebook import tqdm

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from collections import defaultdict
import sumo_rl


class QLearningAgent:
    def __init__(
            self,
            state_space,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.05,
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

    def get_q_table_size(self):
        return len(self.q_table)


def discretize_observation(observation, bins=3):
    """
    Более грубая дискретизация - разбиваем на bins категорий
    low/medium/high вместо точных значений
    """
    discretized = []
    for x in observation:
        if x < 0.33:
            discretized.append(0)
        elif x < 0.66:
            discretized.append(1)
        else:
            discretized.append(2)
    return tuple(discretized)


def run_ql_experiment():
    alpha = 0.1
    gamma = 0.99

    runs = 1
    episodes = 200
    sim_seconds = 3000

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    env = sumo_rl.parallel_env(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=sim_seconds,
        out_csv_name="outputs/4x4grid_ql",
    )

    reward_history = []

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

        for ep in tqdm(range(1, episodes + 1), desc="Episodes"):
            observations, infos = env.reset()
            episode_rewards = {ts: 0 for ts in env.possible_agents}

            state_keys = {
                ts: discretize_observation(observations[ts])
                for ts in env.possible_agents
            }

            step_count = 0
            while env.agents:
                actions = {
                    ts: agents[ts].act(state_keys[ts])
                    for ts in env.agents
                    if ts in observations
                }

                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                for ts in next_obs.keys():
                    next_state_key = discretize_observation(next_obs[ts])
                    done = terminations[ts] or truncations[ts]

                    agents[ts].learn(
                        state_keys[ts],
                        actions[ts],
                        rewards[ts],
                        next_state_key,
                        done
                    )

                    episode_rewards[ts] += rewards[ts]
                    state_keys[ts] = next_state_key

                observations = next_obs
                step_count += 1

            for agent in agents.values():
                agent.decay_epsilon()

            total_reward = sum(episode_rewards.values())
            reward_history.append(total_reward)

            if ep % 10 == 0:
                avg_reward = np.mean(reward_history[-10:])
                q_table_sizes = [a.get_q_table_size() for a in agents.values()]
                eps = list(agents.values())[0].epsilon

                print(f"  Ep {ep}: Avg Reward (last 10): {avg_reward:.2f}, "
                      f"Epsilon: {eps:.4f}, "
                      f"Q-table sizes: {sum(q_table_sizes)} total states")

    env.close()

    return reward_history, agents


if __name__ == "__main__":
    rewards, agents = run_ql_experiment()