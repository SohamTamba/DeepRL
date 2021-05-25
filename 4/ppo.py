import argparse
import gym
import numpy as np

from statistics import mean
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

from mlp import make_mlp
from plot import plot_loss, plot_reward



class PPOTrainer():

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
        self.weight_clip = 0.2

        self.env = gym.make('CartPole-v1')
        self.env.seed(self.seed)
        torch.manual_seed(self.seed)

        self.policy_model = make_mlp(self.num_hidden_layers, N_hidden=self.hidden_layer_size, N_out=2, use_softmax=True)
        self.old_policy_model = make_mlp(self.num_hidden_layers, N_hidden=self.hidden_layer_size, N_out=2, use_softmax=True)
        self.old_policy_model.load_state_dict(self.policy_model.state_dict())
        self.critic_model = make_mlp(self.num_hidden_layers, N_hidden=self.hidden_layer_size, N_out=1, use_softmax=False)

        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic_model.parameters(), lr=learning_rate)


        self.reps_per_episode = 3
        self.eps = np.finfo(np.float32).eps.item()

        self.log_probs = []
        self.distr = []

        self.losses = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.running_reward = 10
        self.num_episodes = 0

    def reset_trainer(self):
        self.log_probs = []
        self.distr = []
        self.losses = []
        self.episode_rewards = []
        self.episode_rewards_mean = []
        self.running_reward = 10
        self.num_episodes = 0


    def get_action(self, state):
        state = torch.from_numpy(state).float()
        probs = self.policy_model(state.unsqueeze(0)).detach()
        policy_distr = Categorical(probs)
        action = policy_distr.sample()
        # Store for loss computation - Refactor this later
        self.log_probs.append(policy_distr.log_prob(action))
        self.distr.append(policy_distr)
        return action.detach().cpu().item()

    def train_loop(self):
        state = self.env.reset()
        episode_reward = 0
        step_limit = 5000
        r_s = []
        states = []
        actions = []
        # Execute a single episode in the environment
        for step in range(1, step_limit):
            states.append(state)
            action = self.get_action(state)
            state, r, done, _ = self.env.step(action)
            self.env.render()
            r_s.append(r)
            actions.append(action)
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

        with torch.no_grad():
            rewards = torch.tensor(rewards[::-1])
            rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps).detach()

            states = torch.tensor(states, dtype=torch.float32)
            pred_rewards = self.critic_model(states).squeeze(1)
            advantages = rewards - pred_rewards
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps).detach()

            actions = torch.tensor(actions)

        # Calculate losses
        total_loss = 0
        for episode_rep in range(self.reps_per_episode):
            policy_loss = 0
            critic_loss = 0
            for old_log_prob, old_distr, state, action, R, advantage in\
                    zip(self.log_probs, self.distr, states, actions, rewards, advantages):

                pred_R = self.critic_model(state.unsqueeze(0)).squeeze()
                critic_loss = critic_loss + F.mse_loss(pred_R, R)

                probs = self.policy_model(state.unsqueeze(0))
                distr = Categorical(probs)
                log_prob = distr.log_prob(action)
                importance_sampling_weight = torch.exp(log_prob-old_log_prob)

                policy_loss_w_clip = advantage * importance_sampling_weight
                policy_loss_wo_clip = advantage * torch.clamp(importance_sampling_weight,\
                                                  min=1.0 - self.weight_clip, max=1.0 + self.weight_clip)
                best_policy_loss = -torch.min(policy_loss_w_clip, policy_loss_wo_clip)
                #breakpoint()
                policy_loss = policy_loss + best_policy_loss
                policy_loss = policy_loss - 0.001 * kl_divergence(old_distr, distr)

                total_loss += critic_loss.detach().cpu().item() + critic_loss.detach().cpu().item()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        del self.log_probs[:]
        del self.distr[:]            
        
        # Logged and to be plotted after training
        total_loss /= self.reps_per_episode
        self.losses.append(total_loss)

        self.num_episodes += 1


def main():

    trainer = PPOTrainer()

    max_episodes = 500
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
