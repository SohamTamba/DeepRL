import matplotlib.pyplot as plt
import numpy as np

def plot_loss(out_path, policy_losses, critic_losses):

    plt.title("Training Curve")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.plot(np.arange(len(policy_losses)), policy_losses, label="Policy Loss")
    if not critic_losses is None:
        plt.plot(np.arange(len(critic_losses)), critic_losses, label="Critc Loss")
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