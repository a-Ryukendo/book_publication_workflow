import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from loguru import logger
from config.settings import settings

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state: List[float], action: str, reward: float, next_state: Optional[List[float]], done: bool):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def __len__(self):
        return len(self.buffer)

class ScrapingRewardSystem:
    def __init__(self):
        self.state_size = 10
        self.action_size = 5
        self.hidden_size = 128
        self.learning_rate = settings.rl_learning_rate
        self.discount_factor = settings.rl_discount_factor
        self.epsilon = settings.rl_epsilon
        self.memory_size = settings.rl_memory_size
        self.q_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.batch_size = 32
        self.update_target_every = 100
        self.training_step = 0
        self.load_model()
    def calculate_quality_score(self, content: str, title: str, content_length: int) -> float:
        if not content or not title:
            return 0.0
        word_count = len(content.split())
        title_length = len(title)
        content_length_score = min(content_length / 1000, 1.0)
        has_paragraphs = content.count('\n\n') > 0
        has_sentences = content.count('.') > 0
        has_words = word_count > 50
        quality_factors = [
            content_length_score * 0.3,
            (word_count / 1000) * 0.2,
            min(title_length / 100, 1.0) * 0.1,
            float(has_paragraphs) * 0.15,
            float(has_sentences) * 0.15,
            float(has_words) * 0.1
        ]
        quality_score = sum(quality_factors)
        return min(quality_score, 1.0)
    def update_state(self, state: List[float], action: str, reward: float, next_state: Optional[List[float]] = None):
        done = next_state is None
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.training_step += 1
    def _train_network(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = [exp[3] for exp in batch]
        dones = torch.BoolTensor([exp[4] for exp in batch])
        action_to_idx = {"aggressive": 0, "conservative": 1, "balanced": 2, "selective": 3, "adaptive": 4}
        action_indices = torch.LongTensor([action_to_idx[action] for action in actions])
        current_q_values = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        next_q_values = torch.zeros(self.batch_size)
        non_final_mask = ~dones
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor([next_states[i] for i in range(len(next_states)) if non_final_mask[i]])
            next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.discount_factor * next_q_values)
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = f"models/scraping_rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }
        torch.save(model_data, filepath)
        logger.info(f"Scraping model saved to {filepath}")
    def load_model(self, filepath: str = None):
        if filepath is None:
            model_dir = "models"
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.startswith('scraping_rl_model') and f.endswith('.pth')]
                if model_files:
                    filepath = os.path.join(model_dir, sorted(model_files)[-1])
        if filepath and os.path.exists(filepath):
            try:
                model_data = torch.load(filepath)
                self.q_network.load_state_dict(model_data['q_network_state_dict'])
                self.target_network.load_state_dict(model_data['target_network_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.training_step = model_data.get('training_step', 0)
                logger.info(f"Scraping model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Failed to load scraping model: {e}")

class ContentProcessingRewardSystem:
    def __init__(self):
        self.state_size = 10
        self.action_size = 5
        self.hidden_size = 128
        self.learning_rate = settings.rl_learning_rate
        self.discount_factor = settings.rl_discount_factor
        self.epsilon = settings.rl_epsilon
        self.memory_size = settings.rl_memory_size
        self.q_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.memory_size)
        self.batch_size = 32
        self.update_target_every = 100
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.quality_scores = []
        self.load_model()
    def calculate_processing_reward(self, original_content: str, processed_content: str, quality_score: float, iteration_count: int) -> float:
        return quality_score
    def get_processing_action(self, state: List[float]) -> str:
        actions = ["aggressive", "conservative", "balanced", "selective", "adaptive"]
        if random.random() < self.epsilon:
            return random.choice(actions)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        action_idx = q_values.argmax().item()
        return actions[action_idx]
    def update_processing_state(self, state: List[float], action: str, reward: float, next_state: Optional[List[float]] = None):
        done = next_state is None
        self.replay_buffer.push(state, action, reward, next_state, done)
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.training_step += 1
    def _train_network(self):
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = [exp[3] for exp in batch]
        dones = torch.BoolTensor([exp[4] for exp in batch])
        action_to_idx = {"aggressive": 0, "conservative": 1, "balanced": 2, "selective": 3, "adaptive": 4}
        action_indices = torch.LongTensor([action_to_idx[action] for action in actions])
        current_q_values = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        next_q_values = torch.zeros(self.batch_size)
        non_final_mask = ~dones
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor([next_states[i] for i in range(len(next_states)) if non_final_mask[i]])
            next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        expected_q_values = rewards + (self.discount_factor * next_q_values)
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save_model(self, filepath: str = None):
        if filepath is None:
            filepath = f"models/processing_rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'quality_scores': self.quality_scores
        }
        torch.save(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    def load_model(self, filepath: str = None):
        if filepath is None:
            model_dir = "models"
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
                if model_files:
                    filepath = os.path.join(model_dir, sorted(model_files)[-1])
        if filepath and os.path.exists(filepath):
            try:
                model_data = torch.load(filepath)
                self.q_network.load_state_dict(model_data['q_network_state_dict'])
                self.target_network.load_state_dict(model_data['target_network_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.training_step = model_data.get('training_step', 0)
                self.episode_rewards = model_data.get('episode_rewards', [])
                self.episode_lengths = model_data.get('episode_lengths', [])
                self.quality_scores = model_data.get('quality_scores', [])
                logger.info(f"Model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    def get_performance_stats(self) -> Dict[str, Any]:
        return {} 