import os
import sys
import pickle
from datetime import datetime

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import numpy as np
from collections import defaultdict
import sumo_rl
from tqdm import tqdm


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


def discretize_observation(observation, bins=4):
    discretized = []
    for x in observation:
        x_clipped = np.clip(x, 0.0, 1.0)
        bin_idx = int(x_clipped * bins)
        bin_idx = min(bin_idx, bins - 1)
        discretized.append(bin_idx)
    return tuple(discretized)


def discretize_observation_adaptive(observation, bins=3):
    discretized = []
    for x in observation:
        if x <= 0.25:
            discretized.append(0)
        elif x <= 0.75:
            discretized.append(1)
        else:
            discretized.append(2)
    return tuple(discretized)


def run_ql_experiment():
    alpha = 0.1
    gamma = 0.95

    runs = 1
    episodes = 100
    sim_seconds = 1500

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.98

    discretization_bins = 3

    env = sumo_rl.parallel_env(
        net_file="nets/4x4-Lucas/4x4.net.xml",
        route_file="nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=sim_seconds,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        out_csv_name="outputs/4x4grid_ql",
    )

    print(f"Traffic lights: {env.possible_agents}")
    print(f"Observation space sample: {env.observation_space(env.possible_agents[0])}")
    print(f"Action space: {env.action_space(env.possible_agents[0])}")

    reward_history = []
    best_avg_reward = -float('inf')

    for run in range(1, runs + 1):
        print(f"\n{'=' * 50}")
        print(f"=== Run {run}/{runs} ===")
        print(f"{'=' * 50}")

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
                ts: discretize_observation(observations[ts], bins=discretization_bins)
                for ts in env.possible_agents
            }

            step_count = 0

            while env.agents:
                actions = {}
                for ts in env.agents:
                    if ts in state_keys:
                        actions[ts] = agents[ts].act(state_keys[ts])

                if not actions:
                    break

                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                for ts in next_obs.keys():
                    if ts not in actions:
                        continue

                    next_state_key = discretize_observation(
                        next_obs[ts], bins=discretization_bins
                    )
                    done = terminations.get(ts, False) or truncations.get(ts, False)

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

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    improvement = "NEW BEST!"
                else:
                    improvement = ""

                print(f"\n  Ep {ep:3d}: Reward={total_reward:8.1f}, "
                      f"Avg(10)={avg_reward:8.1f}, "
                      f"e={eps:.3f}, "
                      f"States={sum(q_table_sizes):5d} {improvement}")

    env.close()

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final epsilon: {list(agents.values())[0].epsilon:.4f}")
    print(f"Best avg reward: {best_avg_reward:.2f}")
    print(f"Final avg reward (last 10): {np.mean(reward_history[-10:]):.2f}")

    for ts, agent in agents.items():
        print(f"  {ts}: {agent.get_q_table_size()} states learned")

    return reward_history, agents


def plot_results(reward_history, window=10):
    import matplotlib.pyplot as plt

    smoothed = np.convolve(
        reward_history,
        np.ones(window) / window,
        mode='valid'
    )

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history, alpha=0.3, label='Raw')
    plt.plot(range(window - 1, len(reward_history)), smoothed, label=f'MA({window})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(reward_history, bins=30, edgecolor='black')
    plt.xlabel('Total Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/ql_training_results.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    rewards, agents = run_ql_experiment()