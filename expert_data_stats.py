import os  
import pickle
import numpy as np

path = './data/cloning/experts/'
for file in os.listdir(path):
    if file[-2:] == '.p':
    	with open(path+file, 'rb') as f:
    		data = pickle.loads(f.read())
    	print(file)
    	print('action size: {}'.format(data['actions'][0].shape))
    	print('min action: {}'.format(np.min(data['actions'])))
    	print('max action: {}'.format(np.max(data['actions'])))
    	print('observation size: {}'.format(data['observations'][0].shape))
    	print('min observation: {}'.format(np.min(data['observations'])))
    	print('max observation: {}'.format(np.max(data['observations'])))
    	print("_____________________________")
    	print("")