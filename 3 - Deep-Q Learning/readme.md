# Deep-Q Learning for CarRacing-v0

## Problem Statement

Implement the Deep-Q Network (DQN) and DQN with Priority Experience Replay (PER) to solve Open AI's CarRacing-v0 task.


## Code

To train the agent using DQN, execute:  `python dqn.py --bufffer_mode vanilla`

To train the agent using DQN with PER, execute:  `python dqn.py --bufffer_mode priority`

Note that the folder folder containing the build_model and DQNAgent are missing, so the above code will not run


## Observations

We were allowed to use pre-trained models for this assignment. Facebook AI Research had recently published its [ResNet-50 model pre-trained with SWaV](https://github.com/facebookresearch/swav). I planned on comparing the performance of an ImageNet pre-trained ResNet-50 vs a SWaV pretrained ResNet-50, but both models performed terribly on the first run
and the models were too slow to hyper-parameter tune. I had the same issue even with ResNet-18.

Halving the frame-rate (num_skip=2) significantly improved performance, probably because it increased the sensitivity of each action and improved exploration.

For both experiments, I performed a grid search for hyper-parameter tuning with:
1. num_skip = [1, 2, 3]
2. discount_factor = [0.9, 0.95, 0.99]
3. decay = [0.9, 0.95, 0.99]

Both models achieved a running mean score of approximately 350. DQN with PER converged faster but performed a little worse than DQN.

