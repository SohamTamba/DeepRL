from torch.utils.data import IterableDataset, DataLoader
from env_wrapper import CarRacingEnv


def dqn_train_dataloader(batch_size, save_path, env_action_mode, \
                         imgs_per_state, time_diff_stack, agent, buffer, num_skip, color):
    dataset = DQNTrainDataset(save_path, env_action_mode, \
                              imgs_per_state, time_diff_stack, agent, buffer, num_skip, color)
    return DataLoader(dataset, batch_size)


class DQNTrainDataset(IterableDataset):

    def __init__(self, save_path, env_action_mode, imgs_per_state, \
                time_diff_stack, agent, buffer, num_skip, color):
        self.env = CarRacingEnv(save_path, env_action_mode, \
                                imgs_per_state, time_diff_stack, num_skip, color)
        self.agent = agent
        self.buffer = buffer
        self.is_done = True
        self.color = color

        self.min_len = 16
        self.episode_reward = None
        self.tolerance = 20

    def __iter__(self):
        self.state = self.env.reset()
        self.no_gain_count = 0
        self.episode_reward = 0
        self.num_returned = 0

        while True:
            action = self.agent.get_action(self.state)
            next_state, r, done, _ = self.env.step(action)
            self.is_done = done
            self.episode_reward += r
            if r < 0:
                self.no_gain_count += 1
            else:
                self.no_gain_count = 0

            self.buffer.insert(self.state, action, r, done, next_state)
            yield self.buffer.sample()
            self.num_returned += 1
            self.state = next_state

            if self.is_done or self.episode_reward < -1 or self.no_gain_count >= self.tolerance:
                while self.num_returned < self.min_len or self.num_returned%self.min_len != 0:
                    yield self.buffer.sample()
                    self.num_returned += 1

                self.is_done = True
                break
