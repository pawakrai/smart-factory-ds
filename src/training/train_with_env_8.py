import sys
import os
# Add the workspace directory to Python path
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, workspace_dir)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
from src.environment.aluminum_melting_env_8 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent
from src.visualization.plot_training import (
    plot_episode_details,
    plot_training_results,
    replay_episode,
)
import matplotlib.pyplot as plt
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def train_agent(
    env, agent, episodes=5000, save_model_every=500, save_path="./checkpoints_env_11"
):
    # สร้างโฟลเดอร์สำหรับบันทึก model หากยังไม่มี
    os.makedirs(save_path, exist_ok=True)

    episode_rewards = []
    episode_lengths = []
    episode_losses = []  # เก็บค่า loss เฉลี่ยของแต่ละ episode
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
        losses = []  # เก็บ loss ของแต่ละ batch ใน episode

        # ใช้ dictionary สำหรับนับ action count สำหรับ 5 actions
        action_counts = {i: 0 for i in range(5)}

        temperatures = []
        powers = []
        energies = []  # ใช้เก็บ energy_consumption
        step_rewards = []

        while not done:
            steps += 1
            # เลือก action ด้วย epsilon-greedy
            action = agent.select_action(state)
            action_counts[action] += 1

            next_state, reward, done = env.step(action)

            temperatures.append(next_state[0])  # Temperature
            powers.append(next_state[3])  # Power
            energies.append(next_state[5])  # Energy consumption (kWh)
            step_rewards.append(reward)

            # อัปเดต Agent (agent.update ควรรองรับการคำนวณ loss ตาม architecture ของ agent ที่ใช้งาน)
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None and loss > 0:
                losses.append(loss)

            state = next_state
            total_reward += reward

        # คำนวณสถิติของ episode
        avg_temp = np.mean(temperatures)
        max_temp = np.max(temperatures)
        avg_power = np.mean(powers)
        final_energy = energies[-1] if energies else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # ข้อมูลเกี่ยวกับ scrap
        scrap_info = env.get_scrap_info()
        
        total_actions = sum(action_counts.values())
        action_percentages = {
            env.action_space[i]: (action_counts[i] / total_actions) * 100
            for i in action_counts
        }

        # แสดงสรุปผลของ episode
        logger.info(f"\nEpisode {episode+1} Summary:")
        logger.info(f"Total steps: {steps}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Average reward per step: {total_reward/steps:.4f}")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info(f"Agent epsilon: {agent.epsilon:.4f}")
        logger.info("Action Distribution:")
        for action_name, percentage in action_percentages.items():
            logger.info(f"  {action_name}: {percentage:.1f}%")
        logger.info("Process Statistics:")
        logger.info(f"  Average Temperature: {avg_temp:.1f}°C")
        logger.info(f"  Maximum Temperature: {max_temp:.1f}°C")
        logger.info(f"  Final Temperature: {temperatures[-1]:.1f}°C")
        logger.info(f"  Average Power: {avg_power:.1f} kW")
        logger.info(f"  Total Energy Consumption: {final_energy:.1f} kWh")
        logger.info(f"  Initial Weight: {env.initial_mass:.1f} kg")
        logger.info(f"  Final Weight: {scrap_info['current_mass']:.1f} kg")
        logger.info(f"  Total Scrap Added: {scrap_info['total_scrap_added']:.1f} kg")
        logger.info(f"  Capacity Utilization: {scrap_info['capacity_utilization']:.1%}")
        logger.info(f"  Number of Scrap Additions: {len(scrap_info['scrap_additions'])}")

        # บันทึกข้อมูลของแต่ละ episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_losses.append(avg_loss)
        episode_temperatures.append(temperatures)
        episode_energies.append(final_energy)
        episode_powers.append(powers)
        episode_step_rewards.append(step_rewards)
        episode_scrap_info.append(scrap_info)

        # บันทึก model checkpoint ทุก ๆ save_model_every episode
        if (episode + 1) % save_model_every == 0:
            checkpoint_path = os.path.join(save_path, f"dqn_episode_{episode+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        # แสดงสถิติสำคัญทุก 100 episodes
        if (episode + 1) % 100 == 0:
            recent_rewards = episode_rewards[-100:]
            avg_recent_reward = np.mean(recent_rewards)
            logger.info(f"\n--- Episode {episode+1} Milestone ---")
            logger.info(f"Average reward (last 100 episodes): {avg_recent_reward:.2f}")
            logger.info(f"Best episode reward: {max(episode_rewards):.2f}")
            logger.info(f"Current epsilon: {agent.epsilon:.4f}")

    # Plot ผลการฝึกทั้งหมด
    plot_training_results(
        episode_rewards, episode_lengths, episode_energies, episode_losses
    )

    # บันทึก log หรือผลลัพธ์การฝึกลงไฟล์ (เช่น numpy array)
    np.save(os.path.join(save_path, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_path, "episode_lengths.npy"), np.array(episode_lengths))
    np.save(os.path.join(save_path, "episode_losses.npy"), np.array(episode_losses))
    np.save(os.path.join(save_path, "episode_energies.npy"), np.array(episode_energies))

    return episode_rewards, episode_lengths, episode_energies, episode_losses, episode_scrap_info


def evaluate_agent(env, agent, episodes=50):
    """
    ประเมิน performance ของ Agent โดยตั้ง epsilon=0 (ไม่สุ่ม action)
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # ปิด epsilon-greedy เพื่อทดสอบนโยบายที่ดีที่สุด

    eval_rewards = []
    eval_energies = []
    eval_temps = []
    eval_scraps = []
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
            
        # เก็บข้อมูลการประเมิน
        eval_rewards.append(total_reward)
        eval_energies.append(next_state[5])  # energy consumption
        eval_temps.append(next_state[0])  # final temperature
        eval_scraps.append(env.get_scrap_info())
        
        logger.info(f"Evaluation Episode {ep+1}: Reward={total_reward:.2f}, "
                   f"Temp={next_state[0]:.1f}°C, Energy={next_state[5]:.1f}kWh")

    avg_reward = np.mean(eval_rewards)
    avg_energy = np.mean(eval_energies)
    avg_temp = np.mean(eval_temps)
    
    logger.info(f"\n=== EVALUATION RESULTS ===")
    logger.info(f"Episodes: {episodes}")
    logger.info(f"Average Reward: {avg_reward:.2f}")
    logger.info(f"Average Energy Consumption: {avg_energy:.1f} kWh")
    logger.info(f"Average Final Temperature: {avg_temp:.1f}°C")
    logger.info(f"Best Reward: {max(eval_rewards):.2f}")
    logger.info(f"Worst Reward: {min(eval_rewards):.2f}")
    
    agent.epsilon = original_epsilon  # คืนค่า epsilon เดิม
    return eval_rewards, eval_energies, eval_temps, eval_scraps


if __name__ == "__main__":
    # สร้าง Environment และ Agent ใหม่ให้เข้ากับ state_dim และ action_dim ของ Env ใหม่
    env = AluminumMeltingEnvironment()
    
    # state_dim = 7 (temperature, weight, time, power, status, energy_consumption, scrap_added)
    # action_dim = 5 (increase_power_strong, increase_power_mild, maintain, decrease_power_mild, decrease_power_strong)
    agent = DQNAgent(state_dim=7, action_dim=5)

    logger.info("=== TRAINING CONFIGURATION ===")
    logger.info(f"Environment: AluminumMeltingEnvironment (version 8)")
    logger.info(f"Agent: DQNAgent (agent2.py)")
    logger.info(f"State Dimension: 7")
    logger.info(f"Action Dimension: 5")
    logger.info(f"Initial Epsilon: {agent.epsilon}")
    logger.info(f"Epsilon Decay: {agent.epsilon_decay}")
    logger.info(f"Learning Rate: {agent.optimizer.param_groups[0]['lr']}")
    logger.info(f"Batch Size: {agent.batch_size}")
    logger.info(f"Target Update Frequency: {agent.target_update_freq}")
    logger.info("="*35)

    episodes = 1500
    logger.info(f"Starting training for {episodes} episodes...")
    
    episode_rewards, episode_lengths, episode_energies, episode_losses, episode_scrap_info = train_agent(
        env, agent, episodes=episodes, save_path="./checkpoints_env_11"
    )

    # บันทึกโมเดลสุดท้าย
    final_model_path = "models/dqn_final_model_11.pth"
    torch.save(agent.model.state_dict(), final_model_path)
    logger.info(f"✅ Saved final model as {final_model_path}")

    # ประเมิน Agent หลังการฝึก
    logger.info("\n=== STARTING FINAL EVALUATION ===")
    eval_rewards, eval_energies, eval_temps, eval_scraps = evaluate_agent(env, agent, episodes=50)

    # สรุปผลการฝึก
    logger.info(f"\n=== TRAINING SUMMARY ===")
    logger.info(f"Total Episodes: {episodes}")
    logger.info(f"Best Training Reward: {max(episode_rewards):.2f}")
    logger.info(f"Final Training Reward: {episode_rewards[-1]:.2f}")
    logger.info(f"Average Evaluation Reward: {np.mean(eval_rewards):.2f}")
    logger.info(f"Model saved to: {final_model_path}")

    # ตัวอย่างการ replay episode พร้อม plot รายละเอียดของ episode
    try:
        replay_episode(env, agent)
    except Exception as e:
        logger.warning(f"Could not replay episode: {e}")

    logger.info("🎉 Training completed successfully!")