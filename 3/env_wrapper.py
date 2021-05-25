import gym
import numpy as np
from gym.wrappers import Monitor

# A states consists of the last images_per_state
class CarRacingEnv(object):
    def __init__(self, vid_path, action_mode, images_per_state=4, time_diff_stack=1, num_skip=2, color="bw"):
        self.n = images_per_state
        self.t = time_diff_stack
        self.num_skip = num_skip
        env = gym.make('CarRacing-v0')
        self.env = Monitor(env, vid_path, force=True)

        self.color = color
        env_obs_shape = self.env.observation_space.shape

        self.obs_shape = (self.n, 3 if color=="rgb" else 1, env_obs_shape[0], env_obs_shape[1])


        imgs_to_store = 1+self.t*(self.n-1)
        self.last_nt_images = np.zeros((imgs_to_store, self.obs_shape[1], self.obs_shape[2], self.obs_shape[3]))
        

        self.index_to_action = CarRacingEnv.make_action_space(action_mode)

    def new_epsiode(self):
        self.last_nt_images[:, :, :, :] = 0

    def process_img(self, img):
        img = np.transpose(img, (2, 0, 1))

        if self.color == "rgb":
            mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
            std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
            norm_img = (img - img.mean(axis=(1,2)).reshape(3,1,1))/img.std(axis=(1,2)).reshape(3,1,1)
            processed_img = norm_img*std + mean
        else:
            img = img.astype(float)

            processed_img = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
            processed_img /= 255.0
            processed_img = processed_img.reshape((1, 96, 96))

        return processed_img

    @staticmethod
    def make_action_space(action_mode):
        choices = [(0, -1, 1), (0, 1), (0, 0.2)]

        # One or more sub-actions
        if action_mode == "mutually_exclusive":
            actions = [np.zeros(len(choices)),]
            for i, choice in enumerate(choices):
                for val in choice:
                    if val != 0:
                        code = np.zeros(len(choices))
                        code[i] = val
                        actions.append(code)
            return np.stack(actions)
        # Any combination of sub-actions
        elif action_mode == "decoupled":
            actions = []
            code = np.zeros(len(choices))
            for choice_0 in choices[0]:
                code_0 = code.copy()
                code_0[0] = choice_0

                for choice_1 in choices[1]:
                    code_1 = code_0.copy()
                    code_1[1] = choice_1

                    for choice_2 in choices[2]:
                        code_2 = code_1.copy()
                        code_2[1] = choice_2
                        actions.append(code_2)
            return np.stack(actions)
        else:
            assert False, f"Action Mode of {action_mode} is not valid"

    def action_space_size(self):
        return len(self.index_to_action)

    def step(self, action_index, is_train=True):
        action = self.index_to_action[action_index]

        R = 0
        for _ in range(self.num_skip+1):
            self.env.render()
            next_img, r, done, info = self.env.step(action)
            R += r

            next_img = self.process_img(next_img)
            self.last_nt_images[1:, :, :, :] = self.last_nt_images[:-1, :, :, :].copy() # Numpy requires clone
            self.last_nt_images[0, :, :, :] = next_img

            if done:
                break

        output = self.last_nt_images.copy()[::self.t, :, :, :]

        if done: self.new_epsiode()


        return output, R, done, info

    def reset(self):

        img = self.env.reset()
        img = self.process_img(img)
        self.new_epsiode()

        self.last_nt_images[0, :, :, :] = img
        output = self.last_nt_images[::self.t].copy()

        return output

    def close(self):
        self.env.close()


if __name__ == "__main__":
    print( CarRacingEnv.make_action_space("decoupled") )
    print( type(CarRacingEnv.make_action_space("decoupled")) )
    print( CarRacingEnv.make_action_space("mutually_exclusive") )
    print( type(CarRacingEnv.make_action_space("mutually_exclusive")) )
