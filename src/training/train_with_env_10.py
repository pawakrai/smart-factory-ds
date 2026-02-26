import sys
import os

# Add the workspace directory to Python path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, workspace_dir)

import numpy as np
import torch
import random
from src.environment.aluminum_melting_env_10 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent
from src.visualization.plot_training import (
    plot_training_results,
    replay_episode,
)
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()

# Debug / training behavior switches
DEBUG_EVAL_Q_VALUES = True
DEBUG_EVAL_Q_STEPS = 5
EPSILON_DECAY_PER_STEP = True  # set True to keep legacy behavior
USE_DENSE_REWARD = True  # reward shaping (recommended for target_temp=950)
TARGET_TEMP_C = 950.0


def train_agent(
    env, agent, episodes=5000, save_model_every=500, save_path="./checkpoints_env_10"
):
    os.makedirs(save_path, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    episode_temperatures = []
    episode_powers = []
    episode_energies = []
    episode_step_rewards = []
    episode_scrap_info = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        losses = []

        action_counts = {i: 0 for i in range(agent.action_dim)}

        temperatures = []
        powers = []
        energies = []
        step_rewards = []

        while not done:
            steps += 1
            action = agent.select_action(state, explore=True)
            action_counts[action] += 1

            next_state, reward, done = env.step(action)

            # Env10 state: [temp, weight, time, power, status, energy_kwh, scrap_added, wall_temp]
            temperatures.append(float(next_state[0]))
            powers.append(float(next_state[3]))
            energies.append(float(next_state[5]))
            step_rewards.append(float(reward))

            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None and float(loss) > 0:
                losses.append(float(loss))

            state = next_state
            total_reward += float(reward)

        # Per-episode epsilon decay (optional)
        agent.end_episode()

        avg_temp = float(np.mean(temperatures)) if temperatures else 0.0
        max_temp = float(np.max(temperatures)) if temperatures else 0.0
        avg_power = float(np.mean(powers)) if powers else 0.0
        final_energy = float(energies[-1]) if energies else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        scrap_info = env.get_scrap_info()

        total_actions = max(1, int(sum(action_counts.values())))
        action_percentages = {
            env.action_space[i]: (action_counts[i] / total_actions) * 100
            for i in action_counts
        }

        logger.info(f"\nEpisode {episode+1} Summary:")
        logger.info(f"Total steps: {steps}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Average reward per step: {total_reward/max(1, steps):.4f}")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Agent epsilon: {agent.epsilon:.4f}")
        logger.info("Action Distribution:")
        for action_name, percentage in action_percentages.items():
            logger.info(f"  {action_name}: {percentage:.1f}%")
        logger.info("Process Statistics:")
        logger.info(f"  Average Temperature: {avg_temp:.1f}°C")
        logger.info(f"  Maximum Temperature: {max_temp:.1f}°C")
        if temperatures:
            logger.info(f"  Final Temperature: {temperatures[-1]:.1f}°C")
        logger.info(f"  Average Power: {avg_power:.1f} kW")
        logger.info(f"  Total Energy Consumption: {final_energy:.1f} kWh")
        logger.info(f"  Initial Weight: {env.initial_mass:.1f} kg")
        logger.info(f"  Final Weight: {scrap_info['current_mass']:.1f} kg")
        logger.info(f"  Total Scrap Added: {scrap_info['total_scrap_added']:.1f} kg")
        logger.info(f"  Capacity Utilization: {scrap_info['capacity_utilization']:.1%}")
        logger.info(
            f"  Number of Scrap Additions: {len(scrap_info['scrap_additions'])}"
        )

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_losses.append(avg_loss)
        episode_temperatures.append(temperatures)
        episode_energies.append(final_energy)
        episode_powers.append(powers)
        episode_step_rewards.append(step_rewards)
        episode_scrap_info.append(scrap_info)

        if (episode + 1) % int(save_model_every) == 0:
            checkpoint_path = os.path.join(save_path, f"dqn_episode_{episode+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_recent_reward = (
                float(np.mean(recent_rewards)) if recent_rewards else 0.0
            )
            logger.info(f"\n--- Episode {episode+1} Milestone ---")
            logger.info(f"Average reward (last 100 episodes): {avg_recent_reward:.2f}")
            logger.info(f"Best episode reward: {max(episode_rewards):.2f}")
            logger.info(f"Current epsilon: {agent.epsilon:.4f}")

    plot_training_results(
        episode_rewards, episode_lengths, episode_energies, episode_losses
    )

    np.save(os.path.join(save_path, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_path, "episode_lengths.npy"), np.array(episode_lengths))
    np.save(os.path.join(save_path, "episode_losses.npy"), np.array(episode_losses))
    np.save(os.path.join(save_path, "episode_energies.npy"), np.array(episode_energies))

    return (
        episode_rewards,
        episode_lengths,
        episode_energies,
        episode_losses,
        episode_scrap_info,
    )


def evaluate_agent(env, agent, episodes=50):
    """ประเมิน performance ของ Agent โดยตั้ง epsilon=0 (ไม่สุ่ม action)"""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    agent.model.eval()

    eval_rewards = []
    eval_energies = []
    eval_temps = []
    eval_scraps = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step_i = 0

        while not done:
            if DEBUG_EVAL_Q_VALUES and ep == 0 and step_i < int(DEBUG_EVAL_Q_STEPS):
                try:
                    q = agent.get_q_values(state)
                    top = int(np.argmax(q)) if len(q) else -1
                    logger.info(
                        f"Eval step={step_i} Q={np.round(q, 3)} | argmax={top} ({env.action_space.get(top)})"
                    )
                except Exception:
                    pass

            action = agent.select_action(state, explore=False)
            next_state, reward, done = env.step(action)
            total_reward += float(reward)
            state = next_state
            step_i += 1

        eval_rewards.append(float(total_reward))
        eval_energies.append(float(next_state[5]))
        eval_temps.append(float(next_state[0]))
        eval_scraps.append(env.get_scrap_info())

        logger.info(
            f"Evaluation Episode {ep+1}: Reward={total_reward:.2f}, "
            f"Temp={next_state[0]:.1f}°C, Energy={next_state[5]:.1f}kWh"
        )

    logger.info(f"\n=== EVALUATION RESULTS ===")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Average Reward: {float(np.mean(eval_rewards)):.2f}")
    logger.info(f"Average Energy Consumption: {float(np.mean(eval_energies)):.1f} kWh")
    logger.info(f"Average Final Temperature: {float(np.mean(eval_temps)):.1f}°C")
    logger.info(f"Best Reward: {float(max(eval_rewards)):.2f}")
    logger.info(f"Worst Reward: {float(min(eval_rewards)):.2f}")

    agent.epsilon = original_epsilon
    return eval_rewards, eval_energies, eval_temps, eval_scraps


if __name__ == "__main__":
    # Reproducibility (training is still stochastic due to env randomness unless seeded)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = AluminumMeltingEnvironment(
        seed=seed,
        target_temp_c=float(TARGET_TEMP_C),
        dense_reward=bool(USE_DENSE_REWARD),
    )
    init_state = env.reset()
    state_dim = int(np.asarray(init_state).shape[0])
    action_dim = int(len(env.action_space))

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        epsilon_decay_per_step=EPSILON_DECAY_PER_STEP,
    )

    logger.info("=== TRAINING CONFIGURATION ===")
    logger.info("Environment: AluminumMeltingEnvironment (version 10)")
    logger.info("Agent: DQNAgent (agent2.py)")
    logger.info(f"State Dimension: {state_dim}")
    logger.info(f"Action Dimension: {action_dim}")
    logger.info(f"Initial Epsilon: {agent.epsilon}")
    logger.info(f"Epsilon Decay: {agent.epsilon_decay}")
    logger.info(f"Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
    logger.info(f"Batch Size: {agent.batch_size}")
    logger.info(f"Target Update Frequency: {agent.target_update_freq}")
    logger.info("=" * 35)

    episodes = 1500
    logger.info(f"Starting training for {episodes} episodes...")

    (
        episode_rewards,
        episode_lengths,
        episode_energies,
        episode_losses,
        episode_scrap_info,
    ) = train_agent(env, agent, episodes=episodes, save_path="./checkpoints_env_10")

    os.makedirs("models", exist_ok=True)
    final_model_path = "models/dqn_final_model_env10.pth"
    torch.save(agent.model.state_dict(), final_model_path)
    logger.info(f"Saved final model as {final_model_path}")

    logger.info("\n=== STARTING FINAL EVALUATION ===")
    eval_rewards, eval_energies, eval_temps, eval_scraps = evaluate_agent(
        env, agent, episodes=50
    )

    logger.info(f"\n=== TRAINING SUMMARY ===")
    logger.info(f"Total Episodes: {episodes}")
    logger.info(f"Best Training Reward: {max(episode_rewards):.2f}")
    logger.info(f"Final Training Reward: {episode_rewards[-1]:.2f}")
    logger.info(f"Average Evaluation Reward: {float(np.mean(eval_rewards)):.2f}")
    logger.info(f"Model saved to: {final_model_path}")

    try:
        replay_episode(env, agent)
    except Exception as e:
        logger.warning(f"Could not replay episode: {e}")

    logger.info("Training completed successfully!")
