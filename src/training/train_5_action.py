import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from collections import deque
from src.environment.aluminum_melting_env_7 import AluminumMeltingEnvironment
from src.agents.agent2 import DQNAgent
from src.visualization.plot_training import (
    plot_episode_details,
    plot_training_results,
    replay_episode,
)
import matplotlib.pyplot as plt
import os
import logging

# ตั้งค่า logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger()


def train_agent(
    env, agent, episodes=5000, save_model_every=500, save_path="./checkpoints"
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

            # Update Agent
            loss = agent.update(state, action, reward, next_state, done)
            if loss is not None:
                losses.append(loss)

            state = next_state
            total_reward += reward

        # คำนวณสถิติของ episode
        avg_temp = np.mean(temperatures)
        max_temp = np.max(temperatures)
        avg_power = np.mean(powers)
        final_energy = energies[-1] if energies else 0
        avg_loss = np.mean(losses) if losses else 0

        total_actions = sum(action_counts.values())
        action_percentages = {
            env.action_space[i]: (action_counts[i] / total_actions) * 100
            for i in action_counts
        }

        # แสดงสรุปผลของ episode
        logger.info(f"\nEpisode {episode+1} Summary:")
        logger.info(f"Total steps: {steps}")
        logger.info(f"Total reward: {total_reward:.2f}")
        logger.info(f"Average reward per step: {total_reward/steps:.2f}")
        logger.info(f"Average loss: {avg_loss:.4f}")
        logger.info("Action Distribution:")
        for action_name, percentage in action_percentages.items():
            logger.info(f"  {action_name}: {percentage:.1f}%")
        logger.info("Process Statistics:")
        logger.info(f"  Average Temperature: {avg_temp:.1f}°C")
        logger.info(f"  Maximum Temperature: {max_temp:.1f}°C")
        logger.info(f"  Average Power: {avg_power:.1f} kW")
        logger.info(f"  Total Energy Consumption: {final_energy:.1f} kWh")

        # บันทึกข้อมูลของแต่ละ episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_losses.append(avg_loss)
        episode_temperatures.append(temperatures)
        episode_energies.append(final_energy)
        episode_powers.append(powers)
        episode_step_rewards.append(step_rewards)

        # บันทึก model checkpoint ทุก ๆ save_model_every episode
        if (episode + 1) % save_model_every == 0:
            checkpoint_path = os.path.join(save_path, f"dqn_episode_{episode+1}.pth")
            torch.save(agent.model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

    # Plot ผลการฝึกทั้งหมด
    plot_training_results(
        episode_rewards, episode_lengths, episode_energies, episode_losses
    )

    # บันทึก log หรือผลลัพธ์การฝึกลงไฟล์
    np.save(os.path.join(save_path, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(save_path, "episode_lengths.npy"), np.array(episode_lengths))
    np.save(os.path.join(save_path, "episode_losses.npy"), np.array(episode_losses))

    return episode_rewards, episode_lengths, episode_energies, episode_losses


def evaluate_agent(env, agent, episodes=100):
    """
    ประเมิน performance ของ Agent โดยตั้ง epsilon=0 (ไม่สุ่ม action)
    """
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # ปิด epsilon-greedy เพื่อทดสอบความแม่นยำ

    eval_rewards = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        eval_rewards.append(total_reward)
        logger.info(f"Evaluation Episode {ep+1}: Total Reward = {total_reward:.2f}")

    avg_reward = np.mean(eval_rewards)
    logger.info(
        f"\nEvaluation over {episodes} episodes, Average Reward: {avg_reward:.2f}"
    )
    agent.epsilon = original_epsilon  # คืนค่า epsilon เดิม
    return eval_rewards


if __name__ == "__main__":
    # สร้าง Environment และ Agent
    env = AluminumMeltingEnvironment()
    # state_dim ใหม่ = 6 (temperature, weight, time, power, status, energy_consumption)
    # action_dim ใหม่ = 5 (increase_power_strong, increase_power_mild, maintain, decrease_power_mild, decrease_power_strong)
    # agent = DQNAgent(state_dim=6, action_dim=5)
    checkpoint_path = "models/dqn_final_model_10.pth"
    agent = DQNAgent.load_checkpoint(checkpoint_path, state_dim=6, action_dim=5)

    episodes = 2000
    episode_rewards, episode_lengths, episode_energies, episode_losses = train_agent(
        env, agent, episodes=episodes, save_path="./checkpoints_env_10"
    )

    # บันทึกโมเดลสุดท้าย
    torch.save(agent.model.state_dict(), "models/dqn_final_model_10.pth")
    logger.info("Saved final model as models/dqn_final_model_10.pth")

    # ประเมิน Agent หลังการฝึก
    eval_rewards = evaluate_agent(env, agent, episodes=100)

    # ตัวอย่างการ replay episode
    # replay_episode(env, agent)

    # # ตัวอย่างการโหลด Checkpoint
    # checkpoint_path = "models/dqn_final_model_10.pth"
    # env = AluminumMeltingEnvironment()
    # agent = DQNAgent.load_checkpoint(checkpoint_path, state_dim=6, action_dim=5)

    # # ทดสอบ Agent ที่โหลดมา
    # agent.epsilon = 0.0  # ตั้งค่า epsilon เป็นค่าต่ำสุดก่อนเรียก replay_episode
    # replay_episode(env, agent)

    # # ประเมิน Agent
    # eval_rewards = evaluate_agent(env, agent, episodes=100)
