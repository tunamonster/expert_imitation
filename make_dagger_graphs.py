import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import pickle
import numpy as np 

path = './results/dagger/'
for file in os.listdir(path):
    if file[-2:] == '.p':
    	with open(path+file, 'rb') as f:
    		data = pickle.loads(f.read())
    		model_name = file[:-2]

    		iterations = len(data['dagger_epoch'])
    		all_x = [i for i in range(iterations)]

    		expert_mean = np.mean(data['expert'])
    		expert_std = np.std(data['expert'])
    		expert_mean = [expert_mean] * iterations
    		expert_lower_std = [expert_mean - expert_std] * iterations
    		expert_upper_std = [expert_mean + expert_std] * iterations

    		clone_means = [np.mean(epoch) for epoch in data['dagger_epoch']]
    		clone_std = [np.std(epoch) for epoch in data['dagger_epoch']]
    		clone_lower_std = [clone_means[i] - clone_std[i] for i in range(iterations)]
    		clone_upper_std = [clone_means[i] + clone_std[i] for i in range(iterations)]
    		
    		plt.plot(all_x, expert_mean,  'g-')
    		plt.plot(all_x, expert_lower_std, 'r-')
    		plt.plot(all_x, expert_upper_std, 'r-')

    		plt.plot(all_x, clone_means, 'bo')
    		plt.plot(all_x, clone_lower_std, 'r-')
    		plt.plot(all_x, clone_upper_std, 'r-')

    		plt.title(model_name)
    		plt.xlabel('rollout iteration')
    		plt.ylabel('environment reward')
    		plt.savefig('results/dagger_graphs/{}.png'.format(model_name))
    		plt.gcf().clear()