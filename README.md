# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

# Results
## Behavioral Cloning (q1)
Gather training data and labels by recording an expert perform actions and train a new model on these observations and decisions.
Run `./make_clone_results.bash` to recreate the graphs
![Alt text](results/cloning_graphs/Ant-v1.png)

## Dagger (q2)
Sample a larger distribution of states to learn how to react when the observations deviate from the optimum. Instead of learning by only observing experts, perform actions with the learner model, record expert actions, but perform learner actions. Then train on the expert actions in batches. In theory, the learner model should converge to the expert model.
Run `./make_dagger_results.bash` to recreate the graphs below
