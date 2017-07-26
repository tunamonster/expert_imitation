import tensorflow as tf
import argparse
from model import init_model
import parse_expert
import load_policy
import gym
import tf_util
from util import Progbar
import numpy as np
import keras 
from collections import defaultdict
import pickle

def normalize(inputs, mean, sdev):
    std_inputs = (inputs - mean) / (sdev + 1e-6)
    return std_inputs

def gather_observations(learning_model, expert_model, envname, obs_mean, obs_sdev):

	env = gym.make(envname)
	max_steps = env.spec.timestep_limit
	target_size = 5000
	observations = []
	expert_actions = []
	all_steps = []
	returns = []
	prog = Progbar(target=1 + target_size)
	while len(observations) < target_size:
		resets = 0
		obs = env.reset()
		obsize = obs.shape
		done = False
		totalr = 0.
		steps = 0
		while not done:
			expert_action = expert_model(obs[None,:])
			observations.append(obs)
			std_obs = normalize(obs, obs_mean, obs_sdev)
			expert_actions.append(expert_action)
			action = learning_model.predict(std_obs, batch_size=1)
			obs, r, done, _ = env.step(action)
			steps += 1
			totalr += r
			env.render()
			if steps >= max_steps:
				break
		prog.update(len(observations), [("in the gym", resets)])
		returns.append(totalr)

	return returns, np.array(observations), np.array(expert_actions)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('expert_policy_file', type=str)
	parser.add_argument('envname', type=str)
	args = parser.parse_args()
	expert_policy = args.expert_policy_file
	expert_behavior = 'data/cloning/experts/{}.p'.format(args.envname)
	print('loading and building expert policy')
	expert_model = load_policy.load_policy(expert_policy)
	print('loaded and built')

	obs_shape, action_shape = parse_expert.model_shapes(expert_policy)
	
	learning_model = init_model(obs_shape, action_shape)

	obs_mean, obs_sdev = parse_expert.standardizers(expert_policy)

	results = defaultdict(list)

	with tf.Session():
		tf_util.initialize()

		for i in range(50):
			print("iteration {}".format(i))
			returns, observations, expert_actions = gather_observations(learning_model, expert_model, 
																args.envname, obs_mean, obs_sdev)
			print('')
			print("n resets: {}".format(len(returns)))
			print("score: {} *-* {} *-* {}".format(np.mean(returns) - np.std(returns), np.mean(returns), np.mean(returns) + np.std(returns)))
			results['dagger_epoch'].append(returns)
			obs_std = normalize(observations, obs_mean, obs_sdev)
			expert_actions = expert_actions.reshape(-1, action_shape)
			learning_model.fit(obs_std, expert_actions, epochs=10, batch_size=16)

		results['expert'] = parse_expert.returns(expert_behavior)
		pickle.dump(results, open("results/dagger/{}.p".format(args.envname), "wb"))

if __name__ == '__main__':
	main()