from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms

import pickle
import numpy as np

import os
import gzip

def process_img(img):
    img = np.ascontiguousarray(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(img)
    return img

def dataloader_train_val(opt):
    num_train = opt.num_train
    num_val = opt.num_val
    batch_size = opt.batch_size
    N_history = opt.N_history

    data_dir = opt.data_dir
    data_file = os.path.join(data_dir, 'data.pkl.gzip')
    with gzip.open(data_file,'rb') as f:
        samples = pickle.load(f)

    X = np.stack(samples["state"])
    Y = np.stack(samples["action"])
    terminal = np.stack(samples["terminal"])

    if num_train < 0:
        num_train = len(X) - num_val
    assert num_train + num_val <= len(X), f"Training and Validation set sizes is too large for dataset of size {len(X)}\n"
    
    X_train = X[:num_train]
    Y_train = Y[:num_train]
    terminal_train = terminal[:num_train]

    X_val = X[-num_val:]
    Y_val = Y[-num_val:]
    terminal_val = terminal[-num_val:]

    dataset_train = CarDataset(X_train, Y_train, terminal_train, N_history, opt.prediction_mode)
    dataset_val = CarDataset(X_val, Y_val, terminal_val, N_history, opt.prediction_mode)

    dataloader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)
    dataloader_val = DataLoader(dataset_val, shuffle=False, batch_size=batch_size)

    return dataloader_train, dataloader_val

class CarDataset(Dataset):

    def __init__(self, X, Y, terminal, N_history, pred_mode="regression"):

        self.N_history = N_history
        self.X = X
        if pred_mode == "regression":
            self.Y = Y
        elif pred_mode == "classification":
            self.Y = np.zeros(len(self.X), dtype=np.int64) # Nothing
            self.Y[ Y[:, 1] > 0.5 ] = 4 #Accelerate
            self.Y[ Y[:, 2] > 0.5 ] = 3 #Break
            self.Y[ Y[:, 0] > 0.5 ] = 2 #Right
            self.Y[ Y[:, 0] < -0.5 ] = 1 #Left
        else:
            raise ValueError(f"Invalid Mode {pred_mode}")
        self.terminal = terminal

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        same_episode= True
        states = []
        for history in range(self.N_history):
            i = idx-history

            if history > 0 and (i < 0 or self.terminal[i]):
                same_episode = False
            if same_episode:
                s = process_img(self.X[i])
            else:
                s = torch.zeros((self.X.shape[3], self.X.shape[1], self.X.shape[2]))
            states.append(s)

        return torch.stack(states), torch.from_numpy(np.array(self.Y[idx]))
