import numpy as np

class ReplayBuffer(object):
    def __init__(self, capacity, obs_shape, max_frames, alpha=0.6, beta_start=0.4, mode="priority"):
        self.beta_start = beta_start
        self.beta = self.beta_start
        self.max_frames = max_frames
        self.alpha = alpha

        self.max_len = capacity
        self.len = 0

        assert mode in ["priority", "vanilla"]
        self.mode = mode

        obs_shape = tuple(obs_shape)
        self.states = np.zeros((capacity,) + tuple(obs_shape))
        self.actions = np.zeros((capacity,), dtype=np.int)
        self.rewards = np.zeros((capacity,))
        self.dones = np.ones((capacity,), dtype=np.bool)
        self.next_states = np.zeros((capacity,) + tuple(obs_shape))
        self.priorities = np.ones((capacity,)) if self.mode == "priority" else None

        self.insertion_ind = 0

    def update_beta(self, step):
        beta = self.beta_start + step * (1.0 - self.beta_start) / self.max_frames
        self.beta = min(1.0, beta)

    def insert(self, state, action, reward, done, next_state):
        self.states[self.insertion_ind] = state
        self.actions[self.insertion_ind] = action
        self.rewards[self.insertion_ind] = reward
        self.dones[self.insertion_ind] = done
        self.next_states[self.insertion_ind] = next_state
        if self.mode == "priority": self.priorities[self.insertion_ind] = self.priorities.max()

        self.len = min(self.max_len, 1+self.len)
        self.insertion_ind = (self.insertion_ind + 1) % self.len

    def sample(self):
        if self.mode == "vanilla":
            index = np.random.randint(0, self.len)
            importance_sampling_weight = np.array(1)
        else:
            priorities = self.priorities[:self.len] if self.len < self.max_len else self.priorities
            probs = priorities ** self.alpha
            probs /= probs.sum()
            index = np.random.choice(np.arange(self.len), p=probs)

            weights = (self.len * probs) ** (-self.beta)
            importance_sampling_weight = weights[index]/weights.max()

        state = self.states[index]
        action = self.actions[index]
        reward = self.rewards[index]
        done = self.dones[index]
        next_state = self.next_states[index]

        return state, action, reward, done, next_state, index, importance_sampling_weight

    def update_priorities(self, priorities, indices):
        if self.mode == "priority":
            self.priorities[indices] = np.maximum(1e-5, priorities)