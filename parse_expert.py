import pickle
import numpy as np

def actions(actions_file):
    with open(actions_file, 'rb') as f:
        data = pickle.loads(f.read())
        observations = data['observations']
        actions = data['actions']
        return observations, actions

def model_shapes(policy_file):
    with open(policy_file, 'rb') as f:
        data = pickle.loads(f.read())
        policy = data['GaussianPolicy']
        output_shape = policy['logstdevs_1_Da'].shape[1]
        input_shape = policy['obsnorm']['Standardizer']['mean_1_D'].shape[1]
        return input_shape, output_shape

def standardizers(policy_file):
    with open(policy_file, 'rb') as f:
        policy = pickle.loads(f.read())['GaussianPolicy']
        mean_std = policy['obsnorm']['Standardizer']['mean_1_D']
        sdev_std = policy['obsnorm']['Standardizer']['meansq_1_D']
        return mean_std, sdev_std

def returns(actions_file):
    with open(actions_file, 'rb') as f:
        data = pickle.loads(f.read())
        return data['returns']