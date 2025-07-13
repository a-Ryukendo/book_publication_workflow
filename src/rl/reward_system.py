import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from ..config.settings import settings

class DQN(nn.Module):
    def __init__(self, ins, outs, h=128):
        super().__init__()
        self.fc1 = nn.Linear(ins, h)
        self.fc2 = nn.Linear(h, h)
        self.fc3 = nn.Linear(h, outs)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class Replay:
    def __init__(self, maxlen):
        self.buf = deque(maxlen=maxlen)
    def push(self, s, a, r, next_s, done):
        self.buf.append((s, a, r, next_s, done))
    def sample(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))
    def __len__(self):
        return len(self.buf)

class ScrapeRL:
    def __init__(self):
        self.in_size = 10
        self.out_size = 5
        self.h_size = 128
        self.lr = settings.rl_learning_rate
        self.gamma = settings.rl_discount_factor
        self.eps = settings.rl_epsilon
        self.mem = settings.rl_memory_size
        self.q = DQN(self.in_size, self.out_size, self.h_size)
        self.target = DQN(self.in_size, self.out_size, self.h_size)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=self.lr)
        self.replay = Replay(self.mem)
        self.batch = 32
        self.update_every = 100
        self.step = 0
        self.rewards = []
        self.lengths = []
        self.scores = []
        self.load_model()
    def quality(self, text, title, length):
        if not text or not title:
            return 0.0
        wc = len(text.split())
        tc = len(title)
        len_score = min(length / 1000, 1.0)
        has_para = text.count('\n\n') > 0
        has_sent = text.count('.') > 0
        has_words = wc > 50
