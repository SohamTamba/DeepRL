import argparse
import gym
import numpy as np

from statistics import mean
import torch
import torch.optim as optim
from torch.distributions import Categorical

from mlp import make_mlp
from plot import plot_loss, plot_reward






class ReinforceTrainer():

    def __init__(
            self,
            num_hidden_layers=1,
            hidden_layer_size=128,
            discount_factor=0.99,
            learning_rate=1e-2,
            seed=123
    ):

        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_size = hidden_layer_size
        self.discount_factor = discount_factor
        self.lr = learning_rate
        self.seed = seed

        self.env = gym.make('CartPole-v1')
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)

	action_size = 2
        self.policy_model = make_mlp(self.num_hidden_layers, N_hidden=self.hidden_layer_size, N_out=action_size, use_softmax=True)
        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)

        self.eps = np.finfo(np.float32).eps.item()

        self.log_probs = []
        self.losses = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.running_reward = 0
        self.num_episodes = 0

    def reset_trainer(self):
        self.log_probs = []
        self.losses = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.running_reward = 0
        self.num_episodes = 0


    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy_model(state.unsqueeze(0))
        policy_distr = Categorical(probs)
        action = policy_distr.sample()
        # Store for loss computation - Refactor this later
        self.log_probs.append(policy_distr.log_prob(action))
        return action.detach().cpu().item()

    def train_loop(self):
        state = self.env.reset()
        episode_reward = 0
        step_limit = 5000
        r_s = []
        # Execute a single episode in the environment
        for step in range(1, step_limit):
            action = self.get_action(state)
            state, r, done, _ = self.env.step(action)
            self.env.render()
            r_s.append(r)
            episode_reward += r
            if done:
                break
        # Keep track of running reward to check if you won
        self.running_reward = 0.05 * episode_reward + 0.95 * self.running_reward
        self.episode_rewards.append(episode_reward)
        self.episode_rewards_mean.append(mean(self.episode_rewards[-100:]))
        
        # Calulate normalized discounted total rewards instead of per-step reward
        rewards = []
        R = 0
        for r in reversed(r_s):
            R = r + self.discount_factor * R
            rewards.append(R)
        rewards = torch.tensor(rewards[::-1])
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps).detach()

	# Policy Loss
        policy_loss = 0
        for log_prob, R in zip(self.log_probs, rewards):
            policy_loss = policy_loss - log_prob * R
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        del self.log_probs[:]

        self.losses.append(policy_loss.detach().cpu().item()) # Logged and to be plotted after training
        self.num_episodes += 1


def main():

    trainer = ReinforceTrainer()
    max_episodes = 500 #3000
    for i_episode in range(max_episodes):
        trainer.train_loop()
        episode_reward = trainer.episode_rewards[-1]

        if i_episode % 10 == 0:
            print('-'*20)
            print(f'Episode {i_episode}')
            print(f'Last reward: {episode_reward}\tAverage reward: {trainer.running_reward}/{trainer.env.spec.reward_threshold}')
            print(f'Policy Loss: {trainer.losses[-1]}')

        if i_episode % 100 == 0:
            plot_loss(f"Reinforce_loss", trainer.losses, None)
            plot_reward(f"Reinforce_reward", trainer.episode_rewards, trainer.episode_rewards_mean)

        if trainer.running_reward >= trainer.env.spec.reward_threshold:
            print("YOU WIN!")
            break

    trainer.env.close()



if __name__ == '__main__':
    main()
