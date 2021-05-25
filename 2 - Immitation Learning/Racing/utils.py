import torch

def get_device_name():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def get_device():
    return torch.device(get_device_name())
