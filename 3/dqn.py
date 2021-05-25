import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np
import argparse
import os

from models.model import build_model
from models.agent import DQNAgent
from env_wrapper import CarRacingEnv
from statistics import mean
from train_dataset import dqn_train_dataloader
from replay_buffer import ReplayBuffer

from plot import plot_loss, plot_reward

import faulthandler
import pickle

# Normalise Inputs

class DQNTrainer():
    def __init__(
        self,
        encoder,
        num_decoder_layers,
        learning_rate = 0.0003,
        batch_size=16,
        discount_factor = 0.99,
        imgs_per_state = 2,
        encoding_spat_size=3,
        clip_grad=1,
        eps_decay = 0.99,
        target_update_freq = 5,
        encoder_mode = "frozen",
        buffer_mode = "priority",
        action_space_mode = "decoupled",
        num_skip=2
        **kwargs
    ):
        super().__init__()

        self.lr = learning_rate
        self.bs = batch_size
        self.discount_factor = discount_factor
        self.buffer_mode = buffer_mode
        self.encoder_mode = encoder_mode
        self.action_space_mode = action_space_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target_update_freq = target_update_freq
        self.clip_grad = clip_grad

        self.model_name = f'eps_decay={eps_decay}-buffer_mode={buffer_mode}-imgs_per_state={imgs_per_state}-'
        self.model_name += f'action_space_mode={action_space_mode}-target_update_freq={target_update_freq}-clip_grad={clip_grad}-'
        self.model_name += f'encoder={encoder}-encoder_mode={encoder_mode}-num_decoder_layers={num_decoder_layers}-'
        self.model_name += f'lr={learning_rate}-encoding_spat_size={encoding_spat_size}-imgs_per_state={imgs_per_state}'

        eps_start = 1.0
        eps_end = 0.1

        self.frames_per_epoch = 64 # Approx
        self.num_epochs = 600
        self.batches_per_epoch = self.frames_per_epoch//self.bs
        self.buffer_capacity = 5000
        self.time_diff_stack = 3
        self.num_skip = num_skip
        self.color = "bw" if encoder == "simple" else "rgb"

        self.imgs_per_state = imgs_per_state

        action_size = 12
        model_type = "simple"#"encoder-decoder" if encoder != "simple" else "simple"
        self.q_model = build_model(model_type, encoder, num_decoder_layers, encoder_mode, imgs_per_state,\
                             encoding_spat_size, action_size).to(self.device)
        self.target_model = build_model(model_type, encoder, num_decoder_layers, encoder_mode, imgs_per_state,\
                                  encoding_spat_size, action_size).to(self.device)

        
        self.agent = DQNAgent(self.q_model, action_size, eps_start, eps_end, eps_decay, self.device)

        self.train_rewards = []
        self.mean_train_rewards = []
        self.test_rewards = []
        self.mean_test_rewards = []
        self.train_losses = []
        self.val_losses = []
        self.val_actions_taken = None
        self.actions_trained_on = None
        self.steps_trained = 0

        self.train_loader = self.get_train_dataloader()
        self.optimizer = self.get_optimizer()



    def get_losses(self, states, actions, rewards, dones, next_states, loss_weights):

        states = states.float().to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        next_states = next_states.float().to(self.device)
        loss_weights = loss_weights.to(self.device)

        self.q_model.train()
        self.target_model.train()

        pred_q_values = self.q_model(states)
        pred_rewards = pred_q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            pred_next_rewards = self.target_model(next_states).max(1).values
            pred_next_rewards[dones] = 0.0
            target_pred_rewards = rewards + pred_next_rewards*self.discount_factor
        target_pred_rewards = target_pred_rewards.detach()
        losses = (pred_rewards - target_pred_rewards) ** 2
        weighted_loss = (loss_weights * losses).mean()

        return weighted_loss, losses

    def train_epoch(self):

        n_batch = 0
        total_loss = 0
        self.actions_trained_on = np.zeros(self.train_loader.dataset.env.action_space_size())


        for batch in self.train_loader:
            states, actions, rewards, dones, next_states, indices, loss_weights = batch

            indices = indices.cpu().numpy()

            loss, losses = self.get_losses(states, actions, rewards, dones, next_states, loss_weights)
            losses = losses.detach().cpu().numpy()

            self.optimizer.zero_grad()
            loss.backward()
            if not self.clip_grad is None:
                clip_grad_norm_(self.q_model.parameters(), self.clip_grad, float('inf'))

            self.optimizer.step()

            self.buffer.update_priorities(losses, indices)

            self.steps_trained += 1

            total_loss += loss.detach().cpu().item()
            n_batch += 1
            for a in actions:
                self.actions_trained_on[a] += 1

        self.agent.update_epsilon()
        total_loss /= n_batch
        self.train_losses.append(total_loss)

        self.train_rewards.append(self.train_loader.dataset.episode_reward)
        avg_reward = mean(self.train_rewards[-100:]) # Might be less than 100 rewards
        self.mean_train_rewards.append(avg_reward)

    @torch.no_grad()
    def val_loop(self):
        env = self.test_env
        state = env.reset()
        episode_reward = 0
        done = False

        ep_len = 0
        loss = 0

        self.q_model.eval()
        self.target_model.eval()
        self.val_actions_taken = np.zeros(self.test_env.action_space_size())
        while not done:
            action = self.agent.get_best_action(state)
            next_state, r, done, info = env.step(action)
            self.val_actions_taken[action] += 1
            episode_reward += r

            pred_q = self.q_model(torch.tensor(state, device=self.device, dtype=torch.float).unsqueeze(0))[0, action].cpu().item()
            pred_next_q = r+self.discount_factor *self.target_model(torch.tensor(next_state, device=self.device,\
                                                                                 dtype=torch.float).unsqueeze(0)).max().cpu().item()
            state = next_state
            loss += (pred_q - pred_next_q)**2
            ep_len += 1

        self.test_rewards.append(episode_reward)
        avg_reward = mean(self.test_rewards[-100:]) # Might be less than 100 rewards
        self.mean_test_rewards.append(avg_reward)
        self.val_losses.append(loss/ep_len)


    def get_train_dataloader(self, init_buffer_size=50):
        env = CarRacingEnv(f"./video/tmp/{self.model_name}/train", self.action_space_mode,\
                           self.imgs_per_state, self.time_diff_stack, self.num_skip, self.color)
        obs_shape = env.obs_shape
        max_frames = self.num_epochs*self.frames_per_epoch
        self.buffer = ReplayBuffer(self.buffer_capacity, obs_shape, max_frames, mode=self.buffer_mode)
        # Populate Buffer
        state = env.reset()

        for _ in range(init_buffer_size):
            action = self.agent.get_random_action()
            next_state, reward, done, info = env.step(action)
            self.buffer.insert(state, action, reward, done, next_state)
            if done:
                state = env.reset()
        env.close()

        save_path = f"./video/{self.model_name}/train"
        return dqn_train_dataloader(self.bs, save_path, self.action_space_mode,\
                                    self.imgs_per_state, self.time_diff_stack, self.agent, self.buffer,\
                                    self.num_skip, self.color)


    def update_target(self):
        print("Updating Target Network")
        self.target_model.load_state_dict(self.q_model.state_dict())

    def get_optimizer(self):
        trainable = self.q_model.decoder if self.encoder_mode == "frozen" else self.q_model
        return optim.Adam(trainable.parameters(), lr=self.lr, amsgrad=True)

    @staticmethod
    def add_args(arg_parser):
        arg_parser.add_argument(
            "--encoder_mode",
            choices=['frozen', 'fine_tuned'],
            default="fine_tuned"
        )
        arg_parser.add_argument(
            "--buffer_mode",
            choices=['priority', 'vanilla'],
            default="vanilla"
        )
        arg_parser.add_argument(
            "--action_space_mode",
            choices=['mutually_exclusive', 'decoupled'],
            default="decoupled"
        )
        arg_parser.add_argument(
            "--encoder",
            choices=["pretrained_resnet50_imagenet", "pretrained_resnet50_swav", "simple"],
            default="simple"
        )
        arg_parser.add_argument(
            "--target_update_freq",
            default=5,
            type=int
        )
        arg_parser.add_argument(
            "--num_skip",
            default=2,
            type=int
        )
        arg_parser.add_argument(
            "--num_decoder_layers",
            default=2,
            type=int
        )
        arg_parser.add_argument(
            "--imgs_per_state",
            default=3,
            type=int
        )
        arg_parser.add_argument(
            "--encoding_spat_size",
            default=3,
            type=int
        )
        arg_parser.add_argument(
            "--learning_rate",
            default=0.001,
            type=float
        )
        arg_parser.add_argument(
            "--batch_size",
            default=32,
            type=int
        )
        arg_parser.add_argument(
            "--discount_factor",
            default=0.95,
            type=float
        )
        arg_parser.add_argument(
            "--eps_decay",
            default=0.99,
            type=float
        )
        return arg_parser

    def restart_envs(self, epoch):
        self.train_loader.dataset.env.close()
        self.train_loader.dataset.env = CarRacingEnv(f"./video/{self.model_name}/train_{epoch}", self.action_space_mode,\
                                     self.imgs_per_state, self.time_diff_stack, self.num_skip, self.color)
        self.train_loader.dataset.is_done = True

if __name__ == '__main__':
    log_dir = "./log/"


    parser = argparse.ArgumentParser(add_help=False)
    parser = DQNTrainer.add_args(parser)
    args = parser.parse_args()

    trainer = DQNTrainer(**args.__dict__)
    output_path = os.path.join(log_dir, trainer.model_name)
    if not os.path.exists(output_path): os.mkdir(output_path)


    plot_freq = 50

    faulthandler.enable()
    print(output_path)
    for epoch in range(1, trainer.num_epochs+1):
        print("-"*30)
        print(f"Epoch = {epoch}", flush=True)
        trainer.train_epoch()

        if epoch%trainer.target_update_freq:
            trainer.update_target()

        print(f"Reward = {trainer.train_loader.dataset.episode_reward}")
        print(f"Loss = {trainer.train_losses[-1]}")

        print(f"\ntrainer.agent.epsilon = {trainer.agent.epsilon}")
        print(f"Buffer size = {trainer.buffer.len}")
        print(f"Best Actions Taken = {trainer.agent.num_best}")
        print(f"Random Actions Taken = {trainer.agent.num_random}")

        print(f"\nDataLoader Actions Taken = {trainer.agent.action_freq}")
        print(f"Actions Trained On = {trainer.actions_trained_on}")

        trainer.restart_envs(epoch)
        if epoch%plot_freq == 0 or epoch == trainer.num_epochs or epoch == 1:
            plot_loss(os.path.join(output_path, "loss.jpg"), trainer.train_losses, None)
            plot_reward(os.path.join(output_path, "rewards.jpg"), trainer.train_rewards, trainer.mean_train_rewards)

    print(output_path)
