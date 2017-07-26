import pickle
import argparse
import numpy as np
from collections import defaultdict
from model import init_model
import parse_expert
from keras.models import Sequential

parser = argparse.ArgumentParser()
parser.add_argument('expert_policy_file', type=str)
parser.add_argument('envname', type=str)

args = parser.parse_args()

def normalize(inputs, mean, sdev):
    std_inputs = (inputs - mean) / (sdev + 1e-6)
    return std_inputs

def render_and_eval(model, in_shape, out_shape, obs_mean, obs_sdev):
    import gym
    env = gym.make(args.envname)
    max_steps = 1000
    actions = []
    returns = []
    for i in range(10):
        print('iter', i)
        obs = env.reset()
        done = False
        steps = 0
        totalr = 0
        while not done:
            obs = obs.reshape(-1, in_shape)
            std_obs = normalize(obs, obs_mean, obs_sdev)
            action = model.predict(std_obs, batch_size=1)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            #if True:
            #    env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    return returns

def main():
    print("loading expert actions")
    expert_behavior = 'data/cloning/experts/{}.p'.format(args.envname)
    expert_policy = args.expert_policy_file
    
    observations, actions = parse_expert.actions(expert_behavior)

    obs_shape, action_shape = parse_expert.model_shapes(expert_behavior)

    actions = actions.reshape(-1, action_shape)
    print("standardize training data")

    obs_mean, obs_sdev = parse_expert.standardizers(expert_policy)
    obs_std = normalize(observations, obs_mean, obs_sdev)

    model = init_model(obs_shape, action_shape)
    
    results = defaultdict(list)
    for i in range(11):
        result = render_and_eval(model, obs_shape, action_shape, obs_mean, obs_sdev)
        results['clone_epoch'].append(result)
        model.fit(obs_std, actions, epochs=5, batch_size=256)

    results['expert'] = parse_expert.returns(expert_behavior)
    results['epochs_per_result'] = 5

    pickle.dump(results, open("results/cloning/{}.p".format(args.envname), "wb"))

if __name__ == '__main__':
    main()