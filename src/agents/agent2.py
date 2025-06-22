import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque  # Add this import
import random  # Also needed for random.sample
import copy  # Needed for deep copying the target network


class DQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        target_update_freq=100,
        hidden_size=512,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)  # Experience replay buffer

        # Configurable network architecture
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_dim),
        )

        # Target network for stable learning
        self.target_model = copy.deepcopy(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.steps = 0

    def select_action(self, state):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def store_experience(self, state, action, reward, next_state, done):
        """Alias for remember method to match sensitivity analysis code"""
        self.remember(state, action, reward, next_state, done)

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # Next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self, state, action, reward, next_state, done):  # เพิ่ม done
        self.remember(state, action, reward, next_state, done)  # บันทึก transition

        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Current Q values
        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        # Next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = (
                rewards + (1 - dones) * self.gamma * max_next_q_values
            )  # ใช้ done

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 1.0
        )  # Gradient clipping
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def load_checkpoint(checkpoint_path, state_dim=6, action_dim=5):
        """
        โหลดโมเดลที่บันทึกไว้จากไฟล์ .pth และคืนค่า agent ที่ถูกฝึกมาแล้ว
        """
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.model.load_state_dict(torch.load(checkpoint_path))
        agent.model.eval()  # ตั้งค่าเป็น evaluation mode
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
        return agent
