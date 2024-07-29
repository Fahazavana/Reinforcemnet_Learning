# REINFORCEMENT LEARNING
## TABULAR-METHODS
* SARSA
* Q-Learning
## DEEP-LEARNING TECHNIQUES
* Deep Q-Network
* Deep Q-Network, with image as input

# Dependencies
* Python3
* PyTorch
* Numpy
* gymnasium
* minigrid
* pickle
# How to
## Directories and scripts
* Utilty: Utility class and function
  * ``EpsilonStrategy`` and ``DQNEpsilonStrategy``: Helper class for $\epsilon$-Greedy and decresinag epsilon
  * ``MiniGrid``: Helper class for each use case of the minigrid:
    * ``MiniGridBase``: Defaults
    * ``MiniGridHash``: The observation become the MD5 hash of the observation
    * ``MiniGridRaw``: The observation become a normalized flattedend array
    * ``MiniGridImage``: The observation is the normalized grayscale of the partila observation
    * ``get_device``: a function that return ``mps``,  ``cuda`` or ``cpu`` depending on the plateform
  * ``ReplayMemory``: Contain the named Tuple ``Transition`` and a Class for the Replay Buffer
  * ``Plots``: Live training monitoring plots, and some helper function for the plots in ``Results.ipynb``
* ``Q_Learning``: Folder for Q-Learning algorithms
  * ``QLearning``: Main script/Module for Q-Learning
  * ``q_learning_table.pkl``: Optimal Q-table found using the Q-learning algorithm
* ``SARSA``: Folder for SARSA-Learning algorithms
  * ``SARSA``: Main script/Module for SARSA-Learning
  * ``sarsa_learning_table.pkl``: Optimal Q-table found using the SARSA-learning algorithm
* ``DQN``: Folder for Deep Q-Network approach
  * ``DQN``: Main scripts/Module for simple Deep Q-Network
  * ``dqn.pth``: Optimal weights of the policy netwok found using DQN
  * ``DQNIMAGE``: Main scripts/Module for CNN Deep Q-Network
  * ``dqn_image.pth``: Optimal weights of the policy netwok found using CNN-DQN

All the ``*_live_plot,json`` files contains the reward, td-error (squared, mse), steps taken per episod, thes value were
dynaicaly during training.

All the ``*_train.json`` files contains the reward, loss, steps take as a function of the steps done (global step).

Each Module/Script that correspond to a learning algorithn have a ``if __name__ == '__main__':``. It that section, 
we check if there is already a ``Q-table`` or ``Network Weight``, if so, we just evaluate, otherwise we train before evaluating.

The notebook ``Results.ipynb`` gather all the results, plots, performace of each algoriths. We can visualize the optimal
policy/path found by each algorthm, and we can use it if we want to check all evaluation result for each algorith.

**PS** The Live plot/monitoring during the training always take over all window during the training, we can close it without 
stoping the training if we want to perform other tasks.




# RESULTS
### Optimal Path found using Q-Learning
<img src="report/figures/QLearning_episode.png">

**SUMMARY**

|               | TRAIN  | EVALUATION |
|---------------|--------|------------|
|Episodes       | 512    | 1000       |
|Completion rate| 92.97% | 100%       |
|Average Reward | 0.808  | 0.958      |
|Average steps  | 52.510 | 12.000     |


<img src='Q_Learning/q_learning.gif' width='400px'>

### Optimal Path found using SARSA-Learning
<img src="report/figures/SARSALearning_episode.png">

**SUMMARY**

|               | TRAIN  | EVALUATION |
|---------------|--------|------------|
|Episodes       | 600    | 1000       |
|Completion rate| 95.67% | 100%       |
|Average Reward | 0.844  | 0.958      |
|Average steps  | 43.185 | 12.000     |


<img src='SARSA/sarsa.gif' width='400px'>


### Optimal Path found using DQN-Learning
<img src="report/figures/DQNLearning_episode.png">


**SUMMARY**

|               | TRAIN  | EVALUATION |
|---------------|--------|------------|
|Episodes       | 2000   | 1000       |
|Completion rate| 99.2%  | 100%       |
|Average Reward | 0.912  | 0.958      |
|Average steps  | 24.794 | 12.000     |


<img src='DQN/dqn.gif' width='400px'>

### Optimal Path found using DQN-Learning with RGB Image technique
<img src="report/figures/DQNRGBLearning_episode.png">

**SUMMARY**

|               | TRAIN  | EVALUATION |
|---------------|--------|------------|
|Episodes       | 1500   | 1000       |
|Completion rate| 98%    | 100%       |
|Average Reward | 0.892  | 0.961      |
|Average steps  | 30.068 | 11.000     |


<img src='DQN/dqn_img.gif' width='400px'>

# TO DO
* Add docstring
* Improve code readability
* Error Handling
* Make the live plot compatible with Jupyter if possible
# REFERENCES
