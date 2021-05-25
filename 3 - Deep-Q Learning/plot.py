import matplotlib.pyplot as plt
import numpy as np

def plot_loss(out_path, train_losses, val_losses):

    plt.title("Training Curve")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(train_losses)), train_losses, label="Training Loss")
    if not val_losses is None: plt.plot(np.arange(len(val_losses)), val_losses, label="Validation Loss")
    plt.legend()

    plt.savefig(out_path)
    plt.close()

def plot_reward(out_path, rewards, avg_rewards):


    plt.title("Reward Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(np.arange(len(rewards)), rewards, label="Reward")
    plt.plot(np.arange(len(avg_rewards)), avg_rewards, label="Avg Rewards of last 100")
    plt.legend()

    plt.savefig(out_path)
    plt.close()