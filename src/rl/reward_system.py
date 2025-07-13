"""
Reinforcement Learning Reward System for content processing optimization
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from config.settings import settings


class DQNNetwork(nn.Module):
    """Deep Q-Network for RL optimization"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: List[float], action: str, reward: float, 
             next_state: Optional[List[float]], done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class ScrapingRewardSystem:
    """RL-based reward system for scraping optimization"""
    
    def __init__(self):
        self.state_size = 10
        self.action_size = 5  # Different scraping strategies
        self.hidden_size = 128
        self.learning_rate = settings.rl_learning_rate
        self.discount_factor = settings.rl_discount_factor
        self.epsilon = settings.rl_epsilon
        self.memory_size = settings.rl_memory_size
        
        # Initialize networks
        self.q_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.replay_buffer = ReplayBuffer(self.memory_size)
        
        # Training parameters
        self.batch_size = 32
        self.update_target_every = 100
        self.training_step = 0
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.quality_scores = []
        
        # Load pre-trained model if exists
        self.load_model()
    
    def calculate_quality_score(self, content: str, title: str, content_length: int) -> float:
        """Calculate quality score for scraped content"""
        if not content or not title:
            return 0.0
        
        # Content quality factors
        word_count = len(content.split())
        title_length = len(title)
        content_length_score = min(content_length / 1000, 1.0)  # Normalize to 0-1
        
        # Text quality indicators
        has_paragraphs = content.count('\n\n') > 0
        has_sentences = content.count('.') > 0
        has_words = word_count > 50
        
        # Calculate composite score
        quality_factors = [
            content_length_score * 0.3,
            (word_count / 1000) * 0.2,  # Word density
            min(title_length / 100, 1.0) * 0.1,  # Title quality
            float(has_paragraphs) * 0.15,
            float(has_sentences) * 0.15,
            float(has_words) * 0.1
        ]
        
        quality_score = sum(quality_factors)
        return min(quality_score, 1.0)
    
    def get_action(self, state: List[float]) -> str:
        """Get action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random action
            actions = ["aggressive", "conservative", "balanced", "selective", "adaptive"]
            return random.choice(actions)
        else:
            # Greedy action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            actions = ["aggressive", "conservative", "balanced", "selective", "adaptive"]
            return actions[action_idx]
    
    def calculate_reward(self, content: str, title: str, content_length: int, 
                        scraping_time: float, success: bool) -> float:
        """Calculate reward based on scraping results"""
        if not success:
            return -1.0
        
        # Base quality score
        quality_score = self.calculate_quality_score(content, title, content_length)
        
        # Time penalty (faster is better)
        time_penalty = min(scraping_time / 30.0, 0.5)  # Max 0.5 penalty
        
        # Content length bonus (more content is better, up to a point)
        length_bonus = min(content_length / 5000, 0.3)  # Max 0.3 bonus
        
        # Title quality bonus
        title_bonus = min(len(title) / 100, 0.2)  # Max 0.2 bonus
        
        # Calculate final reward
        reward = quality_score - time_penalty + length_bonus + title_bonus
        
        return max(reward, -1.0)  # Ensure reward is not too negative
    
    def update_state(self, state: List[float], action: str, reward: float, 
                    next_state: Optional[List[float]] = None):
        """Update RL state and train network"""
        # Add experience to replay buffer
        done = next_state is None
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train network if enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
        
        # Update target network periodically
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
    
    def _train_network(self):
        """Train the DQN network"""
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = [exp[3] for exp in batch]
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        # Convert actions to indices
        action_to_idx = {"aggressive": 0, "conservative": 1, "balanced": 2, 
                        "selective": 3, "adaptive": 4}
        action_indices = torch.LongTensor([action_to_idx[action] for action in actions])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        
        # Next Q values
        next_q_values = torch.zeros(self.batch_size)
        non_final_mask = ~dones
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor([next_states[i] for i in range(len(next_states)) if non_final_mask[i]])
            next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        
        # Expected Q values
        expected_q_values = rewards + (self.discount_factor * next_q_values)
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, filepath: str = None):
        """Save trained model"""
        if filepath is None:
            filepath = f"models/scraping_rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
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
        """Load trained model"""
        if filepath is None:
            # Try to find the latest model
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
        """Get performance statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'epsilon': self.epsilon,
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'avg_quality_score': np.mean(self.quality_scores[-100:]) if self.quality_scores else 0,
            'total_episodes': len(self.episode_rewards)
        }
    
    def update_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """Update epsilon for exploration"""
        self.epsilon = max(self.epsilon * decay_rate, min_epsilon)


class ContentProcessingRewardSystem:
    """RL-based reward system for content processing optimization"""
    
    def __init__(self):
        self.state_size = 15  # Larger state for content processing
        self.action_size = 8  # Different processing strategies
        self.hidden_size = 256
        
        # Initialize networks
        self.q_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network = DQNNetwork(self.state_size, self.action_size, self.hidden_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer and replay buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=settings.rl_learning_rate)
        self.replay_buffer = ReplayBuffer(settings.rl_memory_size)
        
        # Training parameters
        self.batch_size = 32
        self.update_target_every = 100
        self.training_step = 0
        
        # Performance tracking
        self.processing_scores = []
        self.iteration_counts = []
        
        self.load_model()
    
    def calculate_processing_reward(self, original_content: str, processed_content: str,
                                  quality_score: float, iteration_count: int) -> float:
        """Calculate reward for content processing"""
        # Check for None or empty content
        if not original_content or not processed_content:
            return 0.0
            
        # Content improvement score
        original_length = len(original_content)
        processed_length = len(processed_content)
        
        # Length change penalty (shouldn't change too much)
        length_change = abs(processed_length - original_length) / max(original_length, 1)
        length_penalty = min(length_change * 0.5, 0.3)
        
        # Quality improvement bonus
        quality_bonus = quality_score * 0.6
        
        # Iteration efficiency (fewer iterations is better)
        iteration_penalty = min(iteration_count * 0.1, 0.2)
        
        # Content preservation bonus
        preservation_score = self._calculate_preservation_score(original_content, processed_content)
        preservation_bonus = preservation_score * 0.2
        
        # Calculate final reward
        reward = quality_bonus + preservation_bonus - length_penalty - iteration_penalty
        
        return max(reward, -1.0)
    
    def _calculate_preservation_score(self, original: str, processed: str) -> float:
        """Calculate how well the original content is preserved"""
        # Simple word overlap calculation
        original_words = set(original.lower().split())
        processed_words = set(processed.lower().split())
        
        if not original_words:
            return 0.0
        
        overlap = len(original_words.intersection(processed_words))
        return overlap / len(original_words)
    
    def get_processing_action(self, state: List[float]) -> str:
        """Get processing action using epsilon-greedy policy"""
        if random.random() < settings.rl_epsilon:
            actions = ["conservative", "moderate", "aggressive", "creative", 
                      "formal", "casual", "detailed", "concise"]
            return random.choice(actions)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
            actions = ["conservative", "moderate", "aggressive", "creative", 
                      "formal", "casual", "detailed", "concise"]
            return actions[action_idx]
    
    def update_processing_state(self, state: List[float], action: str, reward: float,
                              next_state: Optional[List[float]] = None):
        """Update processing RL state"""
        done = next_state is None
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        if len(self.replay_buffer) >= self.batch_size:
            self._train_network()
        
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
    
    def _train_network(self):
        """Train the processing DQN network"""
        # Similar to scraping training but with different action mapping
        batch = self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = [exp[1] for exp in batch]
        rewards = torch.FloatTensor([exp[2] for exp in batch])
        next_states = [exp[3] for exp in batch]
        dones = torch.BoolTensor([exp[4] for exp in batch])
        
        action_to_idx = {
            "conservative": 0, "moderate": 1, "aggressive": 2, "creative": 3,
            "formal": 4, "casual": 5, "detailed": 6, "concise": 7
        }
        action_indices = torch.LongTensor([action_to_idx[action] for action in actions])
        
        current_q_values = self.q_network(states).gather(1, action_indices.unsqueeze(1))
        
        next_q_values = torch.zeros(self.batch_size)
        non_final_mask = ~dones
        if any(non_final_mask):
            non_final_next_states = torch.FloatTensor([next_states[i] for i in range(len(next_states)) if non_final_mask[i]])
            next_q_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0].detach()
        
        expected_q_values = rewards + (settings.rl_discount_factor * next_q_values)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, filepath: str = None):
        """Save processing model"""
        if filepath is None:
            filepath = f"models/processing_rl_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'processing_scores': self.processing_scores,
            'iteration_counts': self.iteration_counts
        }
        
        torch.save(model_data, filepath)
        logger.info(f"Processing model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load processing model"""
        if filepath is None:
            model_dir = "models"
            if os.path.exists(model_dir):
                model_files = [f for f in os.listdir(model_dir) if f.startswith('processing_rl_model') and f.endswith('.pth')]
                if model_files:
                    filepath = os.path.join(model_dir, sorted(model_files)[-1])
        
        if filepath and os.path.exists(filepath):
            try:
                model_data = torch.load(filepath)
                self.q_network.load_state_dict(model_data['q_network_state_dict'])
                self.target_network.load_state_dict(model_data['target_network_state_dict'])
                self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
                self.training_step = model_data.get('training_step', 0)
                self.processing_scores = model_data.get('processing_scores', [])
                self.iteration_counts = model_data.get('iteration_counts', [])
                logger.info(f"Processing model loaded from {filepath}")
            except Exception as e:
                logger.error(f"Failed to load processing model: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_processing_score': np.mean(self.processing_scores[-100:]) if self.processing_scores else 0,
            'avg_iteration_count': np.mean(self.iteration_counts[-100:]) if self.iteration_counts else 0,
            'total_processing_episodes': len(self.processing_scores)
        } 