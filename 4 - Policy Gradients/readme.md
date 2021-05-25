
# Policy Gradients for Cart-Pole-v1

## Problem Statement

Implement the Reinforce and Proximal Policy Gradients algorithm to solve Open AI's Cart-Pole-v1 task.


## Code

To train the agent using Reinforce, execute:  `python reinforce_trainer.py`

To train the agent using PPO, execute:  `python ppo_trainer.py`


## Observations

Normalizing rewards was essential to solving this task. In hindsight, this was obvious because the Cart-Pole environment provides a reward of +1 for all non-terminal states and 0 for the terminal state. This significantly prolongs and destabilizes training: Once the model is randomly initialized, actions are chosen completely at random. Since almost all actions provide a positive reward, randomly chosen actions are rewarded, which further encourages actions that were chosen at random. Eventually better actions with lower probabilities will be sampled which will provide a higher discounted reward, but this did not happen even after training even for 3,000 episodes! <br>


Reinforce solved the task in 300 episdoes, while PPO solved the task in 200 episodes.

