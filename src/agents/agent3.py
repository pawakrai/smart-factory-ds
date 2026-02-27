import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy


class DQNAgent:
    def __init__(
        self,
        state_dim=8,  # ปรับค่า Default เป็น 8 ตาม State ของ Env ใหม่
        action_dim=10,  # ปรับค่า Default เป็น 10 ตาม Action 0-450 kW
        learning_rate=0.0005,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        epsilon_decay_per_step=True,
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
        self.epsilon_decay_per_step = bool(epsilon_decay_per_step)
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.memory = deque(maxlen=memory_size)
        self.steps = 0

        # ตรวจสอบและเลือกใช้ GPU อัตโนมัติ (ถ้ามี)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Configurable network architecture
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, action_dim),
        ).to(
            self.device
        )  # ย้าย Model ไปยัง Device

        # Target network for stable learning
        self.target_model = copy.deepcopy(self.model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def end_episode(self):
        """Optional epsilon decay hook for per-episode schedules."""
        if not self.epsilon_decay_per_step:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_q_values(self, state):
        """Return Q-values as a 1D numpy array (no side effects)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.model(s).cpu().numpy().reshape(-1)
        return q

    def select_action(self, state, explore: bool = True):
        """
        Select an action.
        - explore=True: epsilon-greedy with decay
        - explore=False: purely greedy
        """
        if explore:
            if self.epsilon_decay_per_step:
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if np.random.random() < self.epsilon:
                return np.random.randint(0, self.action_dim)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.model(s)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def store_experience(self, state, action, reward, next_state, done):
        """Alias for remember method"""
        self.remember(state, action, reward, next_state, done)

    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # แปลงเป็น Tensor และย้ายไป Device เดียวกับ Model
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Update target network with current model weights"""
        self.target_model.load_state_dict(self.model.state_dict())

    def update(self, state, action, reward, next_state, done):
        """รวมคำสั่งจำประสบการณ์และการสอนเข้าด้วยกัน (เรียกใช้หลักตอนเทรน)"""
        self.remember(state, action, reward, next_state, done)

        if len(self.memory) < self.batch_size:
            return 0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q_values = self.model(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    @classmethod
    def load_checkpoint(cls, checkpoint_path, state_dim=8, action_dim=10):
        """
        โหลดโมเดลที่บันทึกไว้ (ถูกแก้ให้เป็น Class Method เพื่อให้สร้าง Object ถูกต้อง)
        """
        agent = cls(state_dim=state_dim, action_dim=action_dim)

        # กรณีรันเครื่องที่ไม่มี GPU แต่โมเดลเทรนจาก GPU มา ต้องใส่ map_location
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent.model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        agent.model.eval()
        print(f"✅ Loaded checkpoint from {checkpoint_path} onto {device}")
        return agent
