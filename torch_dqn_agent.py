import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Define the neural network layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, output_size)

    def forward(self, x):
        # Define the forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.fc8(x)
        return x

# Define Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, lr=0.001):
        # Initialize the DQN agent
        self.state_size = state_size  # Dimensionality of the state space
        self.action_size = action_size  # Dimensionality of the action space
        self.memory = []  # Replay memory to store experiences
        self.batch_size = 64  # Batch size for training
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.lr = lr  # Learning rate for the optimizer
        # Define the DQN model
        self.model = DQN(state_size, action_size)
        # Define the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def remember(self, state, action, reward, next_state, done):
        # Store the experience tuple (state, action, reward, next_state, done) in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose an action based on epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            # Exploration: Choose a random action
            return random.choice(range(self.action_size))
        else:
            # Exploitation: Choose the action with the highest Q-value
            # Convert the state to a tensor and pass it through the model
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.model(state)
            # Choose the action with the highest Q-value
            return torch.argmax(q_values).item()

    def replay(self):
        # Experience replay: Train the model using experiences stored in memory
        if len(self.memory) < self.batch_size:
            return
        # Sample a mini-batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        # states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Compute the target Q-value using the target model
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                if (reward < 0):
                    target = self.gamma * torch.max(self.model(next_state)).item() - reward
                else:
                    target = self.gamma * torch.max(self.model(next_state)).item() + reward
            # Update the Q-value of the chosen action
            target_f = self.model(torch.tensor(state, dtype=torch.float).unsqueeze(0)).squeeze(0).tolist()
            target_f[action] = target
            # states.append(state)
            # targets.append(target_f)
            # Compute and minimize the Mean Squared Error loss
            self.optimizer.zero_grad()
            state = torch.tensor(np.array(state), dtype=torch.float)
            target_f = torch.tensor(np.array(target_f), dtype=torch.float)
            loss = F.mse_loss(self.model(state), target_f)
            loss.backward()
            self.optimizer.step()
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
