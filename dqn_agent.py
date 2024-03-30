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
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    

# Define Deep Q-Learning Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, lr=0.001):
        # Initialize agent parameters
        self.state_size = state_size  # Dimensionality of the state space
        self.action_size = action_size  # Number of possible actions
        self.memory = []  # Replay memory to store experiences
        self.batch_size = 64  # Batch size for training
        self.gamma = gamma  # Discount factor for future rewards
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Decay rate for exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.lr = lr  # Learning rate for the optimizer
        
        # Define the main model and target model
        self.model = DQN(state_size, action_size)  # Main Q-network
        self.target_model = DQN(state_size, action_size)  # Target Q-network
        self.target_model.load_state_dict(self.model.state_dict())  # Initialize target model with the same weights
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)  # Optimizer for training the main model

    def remember(self, state, action, reward, next_state, done):
        # Store experience tuple (state, action, reward, next_state, done) in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose an action based on epsilon-greedy policy
        if np.random.random() <= self.epsilon:
            # Exploration: Choose a random action
            # this is based on environment action_sapve
            return np.random.uniform(0,1,size=self.action_size)
        else:
            # Exploitation: Choose the action with the highest Q-value
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            q_values = self.model(state)
            return q_values.squeeze(0).detach().numpy()

    def replay(self):
        # Experience replay: Train the model using experiences stored in memory
        if len(self.memory) < self.batch_size:
            return
        # Sample a mini-batch of experiences from memory
        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                # Compute the target Q-value using the target model
                next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            # Update the Q-value of the chosen action
            target_f = self.model(torch.tensor(state, dtype=torch.float).unsqueeze(0)).squeeze(0).tolist()
            action_index = np.argmax(action)
            target_f[action_index] = target
            states.append(state)
            targets.append(target_f)
        # Compute and minimize the Mean Squared Error loss
        self.optimizer.zero_grad()
        states = torch.tensor(np.array(states), dtype=torch.float)
        targets = torch.tensor(np.array(targets), dtype=torch.float)
        loss = F.mse_loss(self.model(states), targets)
        loss.backward()
        self.optimizer.step()
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network weights every few steps (e.g., every 50 steps)
        if len(self.memory) % 10 == 0:
            self.update_target_model()

    def update_target_model(self):
        # Update the weights of the target network with the weights of the main network
        self.target_model.load_state_dict(self.model.state_dict())
