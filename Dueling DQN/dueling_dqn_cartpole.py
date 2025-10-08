# Dueling DQN for CartPole
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1️⃣ Imports
import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time  # for slowing down evaluation steps

# Gymnasium or Gym
try:
    import gymnasium as gym
    NEW_GYM = True
except:
    import gym
    NEW_GYM = False

# 2️⃣ Environment setup for training (no render)
env = gym.make("CartPole-v1")

if NEW_GYM:
    state, _ = env.reset(seed=42)
else:
    state = env.reset()

print("Environment created. State shape:", env.observation_space.shape)
print("Action space:", env.action_space.n)
print("Initial state:", state)

# 3️⃣ Dueling DQN Network
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, action_dim)
        )

    def forward(self, x):
        x = self.feature(x)
        V = self.value(x)
        A = self.adv(x)
        A = A - A.mean(dim=1, keepdim=True)
        Q = V + A
        return Q

# 4️⃣ Replay Buffer & Agent
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_dim, action_dim, hidden=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.online = DuelingDQN(state_dim, action_dim, hidden).to("cpu")
        self.target = DuelingDQN(state_dim, action_dim, hidden).to("cpu")
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = optim.Adam(self.online.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(20000)
        self.total_steps = 0

    def epsilon(self):
        EPS_START = 1.0
        EPS_END = 0.02
        EPS_DECAY = 30000
        t = min(self.total_steps, EPS_DECAY)
        return EPS_END + (EPS_START - EPS_END) * (1 - t / EPS_DECAY)

    def act(self, state, explore=True):
        if explore and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_tensor)
        return int(q_values.argmax(dim=1).item())

    def push(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

# 5️⃣ Training & Update
def update(agent, batch_size=64, gamma=0.99, double_dqn=False):  # double_dqn disabled
    if len(agent.buffer) < batch_size:
        return None

    batch = agent.buffer.sample(batch_size)
    states = torch.tensor(np.vstack(batch.state), dtype=torch.float32)
    actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(np.vstack(batch.next_state), dtype=torch.float32)
    dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)

    q_values = agent.online(states).gather(1, actions)

    with torch.no_grad():
        # Standard DQN update (Double DQN disabled)
        next_q = agent.target(next_states).max(dim=1, keepdim=True).values
        target_q = rewards + gamma * next_q * (1 - dones)

    loss = nn.functional.smooth_l1_loss(q_values, target_q)
    agent.opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(agent.online.parameters(), 10)
    agent.opt.step()
    return loss.item()

def train_cartpole(num_episodes=50, max_steps=200, target_update_every=5):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = Agent(state_dim, action_dim)

    all_rewards = []
    losses = []

    for ep in range(1, num_episodes + 1):
        if NEW_GYM:
            state, _ = env.reset(seed=42)
        else:
            state = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            agent.total_steps += 1
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)[:4]
            agent.push(state, action, reward, next_state, float(done))
            loss = update(agent)  # Double DQN disabled
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_reward += reward
            if done:
                break

        all_rewards.append(ep_reward)

        if ep % target_update_every == 0:
            agent.target.load_state_dict(agent.online.state_dict())

        if ep % 10 == 0:
            avg_reward = np.mean(all_rewards[-50:])
            print(f"Episode {ep} | Reward {ep_reward:.1f} | Avg50 {avg_reward:.2f} | Epsilon {agent.epsilon():.3f} | Loss {np.mean(losses[-100:]) if losses else 0:.4f}")

    plt.plot(all_rewards, label="Episode Reward")
    plt.plot(np.convolve(all_rewards, np.ones(10)/10, mode='valid'), label="Rolling Avg (10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()

    torch.save(agent.online.state_dict(), "dueling_dqn_cartpole.pt")
    print("Model saved as dueling_dqn_cartpole.pt")

# 6️⃣ Run training
train_cartpole(num_episodes=50, max_steps=200)

# 7️⃣ Evaluate / Watch the agent (with smooth visualization and window stay)
eval_env = gym.make("CartPole-v1", render_mode="human")  # render window
state, _ = eval_env.reset()
state_dim = eval_env.observation_space.shape[0]
action_dim = eval_env.action_space.n
agent = Agent(state_dim, action_dim)
agent.online.load_state_dict(torch.load("dueling_dqn_cartpole.pt"))

done = False
total_reward = 0
while not done:
    action = agent.act(state, explore=False)
    next_state, reward, done, _, _ = eval_env.step(action)
    state = next_state
    total_reward += reward
    time.sleep(0.02)  # slows down steps for smooth visualization

# Keep the GUI window open after the game ends
input("Episode finished! Press Enter to close the window...")
eval_env.close()

print("Evaluation Reward:", total_reward)
