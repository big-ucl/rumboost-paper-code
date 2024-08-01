from keras import backend as K
from swissmetro_paper import data_manager as swissDM
if __name__ == "__main__" and __package__ is None:
	from sys import path
	from os.path import dirname as dir
	path.append(dir(path[0]))
	splits = path[0].split('/')

parent = '/'.join(splits[:-1])
path.append(dir(parent))

#path.append('../GBM feature extraction/Multiclass problem')

import utilities.run_utils as ru
import utilities.grad_hess_utilities as ghu
import argparse
import time
import numpy as np
import pandas as pd
"""
	Run script for Swissmetro Real Datset Experiments
	For each model: 
		- Define architecture (# X inputs, # Q inputs, model architecture)
		- Create input with keras_input()
		- Run the appropriate function below
	
	Main() flags:
	------------
	models:		Estimates many models on Swissmetro dataset
	scan: 		Perform a architectural scan of neurons on L-MNL
"""
def normalize(data):
	return (data-data.mean(axis=0))/(data.std(axis=0))

parser = argparse.ArgumentParser(description='Choose Flags for training on experiments')
parser.add_argument('--scan', action='store_true', help='Trains multiple L-MNL models of increasing size on Swissmetro')
parser.add_argument('--models', action='store_true', help='Trains a full set of models on Swissmetro')
args = parser.parse_args()

models = args.models
scan = args.scan

choices_num = 4  # Train, SM, Car
batchSize = 50

def LPMCMixed(filePath, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=False,
					minima=None, train_betas=True, filePart='', saveName='',
					networkSize=100, hidden_layers=1, verbose=100, nEpoch=200):

	betas, saveExtension, model = ru.runMixed(filePath, fileInputName, beta_num, choices_num, nEpoch, train_data_name,
									   batchSize, extraInput, nExtraFeatures, minima, train_betas, filePart,
									   saveName=saveName, networkSize=networkSize, hidden_layers=hidden_layers,
									   verbose=verbose)
	
	return betas, saveExtension, model

if __name__ == '__main__':

	# splits data into train and test set
	# extensions = swissDM.train_test_split(filePath, seed = 32)

	filePath = 'swissmetro_paper/'
	extensions = ['_train', '_test']
	CV_extensions = ['_CVtrain_0', '_CVtrain_1', '_CVtrain_2', '_CVtrain_3', '_CVtrain_4', '_CVtest_0', '_CVtest_1', '_CVtest_2', '_CVtest_3', '_CVtest_4']

	if models:
		folderName = 'models/'
		fileInputName = 'lpmc'


		lmnlArchitecture = True
		beta_num = 17
		nExtraFeatures = 15

		layer_width = [50, 100, 200]
		hidden_layers = [1, 2, 3]
		results = {}
		avg_metrics = []
		best_ce = 1000
		
		for w in layer_width:
			for h in hidden_layers:
				metrics = []

				for i in range(1):
					init_time = time.time()

					train_data_name = filePath+folderName+'keras_input_'+fileInputName+CV_extensions[i]+'.npy'

					_, _, model = LPMCMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose=2, networkSize=w, hidden_layers=h, nEpoch=200)

					iteration_time = time.time()-init_time

					test_data_name = filePath+folderName+'keras_input_'+fileInputName+CV_extensions[i+5]+'.npy'

					CV_test_data = np.load(test_data_name)  
					CV_test_labels = CV_test_data[:,-1,:]
					CV_test_data = np.delete(CV_test_data, -1, axis = 1)

					CV_test_data = np.expand_dims(CV_test_data, -1)

					extra_data = np.load(test_data_name[:-4] + '_extra.npy')
					extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

					#CV_test_data = normalize(CV_test_data)
					extra_data = normalize(extra_data)

					cel, acc = ghu.get_likelihood_accuracy(model,[CV_test_data, extra_data], CV_test_labels)
					
					name = str(w) + '_' + str(h) + '_' + str(i)
					results[name] = [cel, acc, iteration_time]
					metrics.append(cel)

					print(f"CV iteration: {i} ({iteration_time:.2f}s)")
					print(f"Cross Entropy Loss: {cel:.4f}")

					K.clear_session()

				avg_metrics.append(np.mean(metrics))

				if avg_metrics[-1] < best_ce:
					best_ce = avg_metrics[-1]
					best_width = w
					best_hidden = h


		for i in range(5):
			init_time = time.time()

			train_data_name = filePath+folderName+'keras_input_'+fileInputName+CV_extensions[i]+'.npy'

			_, _, model = LPMCMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose=2, networkSize=best_width, hidden_layers=best_hidden)

			iteration_time = time.time()-init_time

			test_data_name = filePath+folderName+'keras_input_'+fileInputName+CV_extensions[i+5]+'.npy'

			CV_test_data = np.load(test_data_name)  
			CV_test_labels = CV_test_data[:,-1,:]
			CV_test_data = np.delete(CV_test_data, -1, axis = 1)

			CV_test_data = np.expand_dims(CV_test_data, -1)

			extra_data = np.load(test_data_name[:-4] + '_extra.npy')
			extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

			#CV_test_data = normalize(CV_test_data)
			extra_data = normalize(extra_data)

			cel, acc = ghu.get_likelihood_accuracy(model,[CV_test_data, extra_data], CV_test_labels)
			
			
			results[i] = [cel, acc, iteration_time]

			print(f"CV iteration: {i} ({iteration_time:.2f}s)")
			print(f"Cross Entropy Loss: {cel:.4f}")

			K.clear_session()


		print("L-MNL")
		init_time = time.time()

		train_data_name = filePath+folderName+'keras_input_'+fileInputName+extensions[0]+'.npy'
		_, _, model = LPMCMixed(filePath+folderName, fileInputName, beta_num, choices_num, nExtraFeatures, train_data_name, extraInput=True, verbose=2, networkSize=best_width, hidden_layers=best_hidden)

		iteration_time = time.time()-init_time

		test_data_name = filePath+folderName+'keras_input_'+fileInputName+extensions[1]+'.npy'
		
		test_data = np.load(test_data_name)
		test_labels = test_data[:,-1,:]
		test_data = np.delete(test_data, -1, axis = 1)

		test_data = np.expand_dims(test_data, -1)

		extra_data = np.load(test_data_name[:-4] + '_extra.npy')
		extra_data = np.expand_dims(np.expand_dims(extra_data, -1),-1)

		#test_data = normalize(test_data)
		extra_data = normalize(extra_data)

		cel, acc = ghu.get_likelihood_accuracy(model,[test_data, extra_data], test_labels)

		results["test"] = [cel, acc, iteration_time]

		print(f"Test set ({iteration_time:.2f}s)")
		print(f"Cross Entropy Loss: {cel:.4f}")

		results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Cross Entropy Loss', 'Accuracy', 'Time'])
		results_df.to_csv(filePath+folderName+'results.csv')

		K.clear_session()

