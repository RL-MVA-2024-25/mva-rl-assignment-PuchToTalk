from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from copy import deepcopy
import random
import numpy as np
import os

# DQN Model (more complex)
class DQN(nn.Module):
    def __init__(self, state_dim, nb_neurons, n_action):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons), 
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons),
            nn.ReLU(),
            nn.Linear(nb_neurons, nb_neurons), 
            nn.ReLU(),
            nn.Linear(nb_neurons, n_action) 
        )

    def forward(self, x):
        return self.layers(x)

# Env config
env_df = TimeLimit(HIVPatient(domain_randomization=False), max_episode_steps=200)
env_rd = TimeLimit(HIVPatient(domain_randomization=True), max_episode_steps=200)
state_dim = env_df.observation_space.shape[0]


class ProjectAgent:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gamma = config['gamma']
        self.batch_size = config['batch_size']
        self.nb_actions = config['nb_actions']
        self.memory = ReplayBuffer(config['buffer_size'], device)
        self.epsilon_max = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_stop = config['epsilon_decay_period']
        self.epsilon_delay = config['epsilon_delay_decay']
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = DQN(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        self.target_model = deepcopy(self.model).eval()
        self.criterion = config['criterion']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.nb_gradient_steps = config['gradient_steps']
        self.update_target = config['update_target']
        self.best_model_path = "./models/model_4.pt" 

        os.makedirs(os.path.dirname(self.best_model_path), exist_ok=True)

    def act(self, observation, use_random=False):
        device = next(self.model.parameters()).device
        with torch.no_grad():
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def save(self, path=None):
        path = path or self.best_model_path
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory) 
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self):
        device = torch.device('cpu')
        self.model = DQN(state_dim, config['nb_neurons'], self.nb_actions).to(device)
        path = "./models/model_4.pt"
        try:

            self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            print(f"Model loaded successfully from {path}")
        except TypeError:

            print("Loading without weights_only due to older PyTorch version...")
            self.model.load_state_dict(torch.load(path, map_location=device))
        self.model.eval()

    def greedy_action(self, network, state):
        device = next(network.parameters()).device
        with torch.no_grad():
            Q = network(torch.Tensor(state).unsqueeze(0).to(device))
            return torch.argmax(Q).item()

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            target = R + (1 - D) * self.gamma * QYmax
            QXA = self.model(X).gather(1, A.long().unsqueeze(1))
            loss = self.criterion(QXA, target.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def display_learning_curve(self, rewards_per_episode):
        """
        Plot the learning curve based on total rewards per episode.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(rewards_per_episode, label="Total reward")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Learning curve")
        plt.legend()
        plt.grid()
        plt.show()


    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        epsilon = self.epsilon_max
        episode = 0
        step = 0
        env = env_df
        state, _ = env.reset()
        episode_reward = 0  
        rewards_per_episode = []  

        while episode < config['max_episode']:

            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

       
            action = env.action_space.sample() if np.random.rand() < epsilon else self.greedy_action(self.model, state)


            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)


            episode_reward += reward

            # Train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target network
            if step % self.update_target == 0:
                self.target_model.load_state_dict(self.model.state_dict())

            state = next_state
            step += 1

            if done or trunc:
                episode += 1
                rewards_per_episode.append(episode_reward) 
                print(f"Episode: {episode}, Epsilon: {epsilon:.2f}, Total Reward: {episode_reward:.2f}")
                episode_reward = 0 
                state, _ = env.reset()


        print("Training completed. Saving model...")
        self.save("./models/model_4.pt")
        print(f"Model saved to ./models/model_4.pt")


        #self.display_learning_curve(rewards_per_episode) # display the curve to observe the peak 



class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


# Play with the parameters
config = {
    'nb_actions': env_df.action_space.n,
    'max_episode': 5000,
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'buffer_size': 1000000,
    'epsilon_min': 0.10,
    'epsilon_max': 1.0,
    'epsilon_decay_period': 20000,
    'epsilon_delay_decay': 700,
    'batch_size': 256,
    'gradient_steps': 5,
    'criterion': nn.HuberLoss(),
    'update_target': 700,
    'nb_neurons': 256
}


"""
if __name__ == "__main__":
    agent = ProjectAgent()
    print("Starting training...")
    agent.train()
    print("Training completed.")
    print("Loading the saved model for verification...")
    agent.load()
"""