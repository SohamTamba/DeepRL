import torch.nn as nn


def make_mlp(num_hidden_layers, N_in=4, N_hidden=128, N_out=2, drop_prob=0.6, use_softmax=False):
    layers = []
    current_N = N_in
    for _ in range(num_hidden_layers):
        layers.append(nn.Linear(current_N, N_hidden))
        layers.append(nn.Tanh()) # Recommendation of What Matters in RL
        current_N = N_hidden
        if drop_prob > 0.0:
            layers.append(nn.Dropout(p=drop_prob))
    layers.append(nn.Linear(current_N, N_out))
    if use_softmax:
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)