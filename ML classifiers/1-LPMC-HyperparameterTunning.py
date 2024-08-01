#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyperparameter Tunning of the Machine Learning methods for the LPMC dataset.

Authors:
- José Ángel Martín-Baos
- Julio Alberto López-Gomez
- Luis Rodríguez-Benítez
- Tim Hillel
- Ricardo García-Ródenas
"""

# Import packages
import pandas as pd  # For file input/output
import scipy
import time
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import hyperopt
import warnings

# Import the Classification models
from Models.DNN import DNN
from Models.NN import NN
from Models.LightGBM import LightGBM
from Models.ResLogit import ResLogit
#from MNL import MNL

import random
from collections import Counter, defaultdict

# Load common functions for the experiments
from expermients_functions import *



# Load the data
dataset_prefix = "LPMC"
data_dir = "../Data/Datasets/preprocessed/"
adjusted_hyperparms_dir = "../Data/adjusted-hyperparameters/"
dataset_name = dataset_prefix+"_train.csv"
mode_var = "travel_mode"
individual_id = 'household_id'
hyperparameters_file = dataset_prefix+"_hyperparameters"
crossval_pickle_file = data_dir+dataset_prefix+"_hyperparams_crossval.pickle"
reset_crossval_indices = 0 # Set to 0 for reproducibility of the experiment over multiple executions

scaled_fetures = ['day_of_week', 'start_time_linear', 'age', 'car_ownership',
                  'distance', 'dur_walking', 'dur_cycling', 'dur_pt_access', 'dur_pt_rail',
                  'dur_pt_bus', 'dur_pt_int_waiting', 'dur_pt_int_walking', 'pt_n_interchanges',
                  'dur_driving', 'cost_transit', 'cost_driving_fuel']

train = pd.read_csv(data_dir + dataset_name, sep=',')

# Divide the dataset into charasteristics and target variable
X = train.loc[:, train.columns != mode_var]
y = train[mode_var]

# Reduce dataset size to reduce computational cost of the hyperparameter estimation
_, X_sample, _, y_sample = train_test_split(X, y,
                                            stratify=y,
                                            test_size=0.25,
                                            random_state=2022)
X_sample = X_sample.reset_index(drop=True)
y_sample = y_sample.reset_index(drop=True)

# Extract the individual ID to later group observations using it
groups = np.array(X_sample[individual_id].values)
X_sample = X_sample.drop(columns=individual_id)

X_n_cols = X_sample.shape[1]
n_alternatives = y.nunique()


###########################   Set the parameters   ############################
# Number of iterations of the random search
n_iter = 1000
#n_iter = 1 # Uncomment for quick experiments
CV = 5 # Number of cross-validation

hyperparameters_file = hyperparameters_file +'_'+ str(n_iter) + '.csv'

# Set the hyperparameters search space
hyperparameters = {"NN"  : {"hidden_layer_sizes": hyperopt.pyll.scope.int(hyperopt.hp.quniform('hidden_layer_sizes', 10, 500,1)),
                            "activation" : hyperopt.hp.choice('activation', ["tanh"]), # TODO: Consider other activation functions (ReLU, LeakyReLU, etc.)
                            "solver" : hyperopt.hp.choice('solver', ["lbfgs","sgd","adam"]),
                            "learning_rate_init": hyperopt.hp.uniform('learning_rate_init', 0.0001, 1),
                            "learning_rate" : hyperopt.hp.choice('learning_rate', ["adaptive"]),
                            "max_iter": hyperopt.hp.choice('max_iter', [10000000]),
                            "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                            "tol" : hyperopt.hp.choice('tol', [1e-3]),
                           },
                   "DNN"  : {"input_dim": hyperopt.hp.choice('input_dim', [X_n_cols]),
                             "output_dim": hyperopt.hp.choice('output_dim', [n_alternatives]),
                             "depth": hyperopt.hp.choice('depth', [2,3,4,5,6,7,8,9,10]),
                             "width": hyperopt.hp.choice('width', [25,50,100,150,200]),
                             "drop": hyperopt.hp.choice('drop', [0.5, 0.3, 0.1]),
                             # TODO: Consider adding the activation functions for the hidden layers (thanh, ReLU, LeakyReLU, etc.)
                             "epochs": hyperopt.pyll.scope.int(hyperopt.hp.quniform('epochs', 50, 200,1)),
                             "batch_size": hyperopt.hp.choice('batch_size', [128,256,512,1024]),
                           },
                   "LightGBM" : {'num_leaves': hyperopt.pyll.scope.int(hyperopt.hp.quniform('num_leaves', 2, 100, 1)),
                                'min_gain_to_split':  hyperopt.hp.loguniform('min_gain_to_split', -9.21, 1.61), # interval: [0.0001, 5]
                                'min_sum_hessian_in_leaf': hyperopt.hp.loguniform('min_sum_hessian_in_leaf', 0, 4.6),
                                'min_data_in_leaf': hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_data_in_leaf', 1, 200, 1)),
                                'bagging_fraction': hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
                                'bagging_freq': hyperopt.hp.choice('bagging_freq', [1, 5, 10]),
                                'feature_fraction': hyperopt.hp.uniform('feature_fraction', 0.5, 1),
                                'lambda_l1': hyperopt.hp.loguniform('lambda_l1', -9.21, 2.30), # interval: [0.0001, 10]
                                'lambda_l2': hyperopt.hp.loguniform('lambda_l2', -9.21, 2.30), # interval: [0.0001, 10]
                                'max_bin': hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_bin', 100, 500, 1)),
                                'num_iterations': hyperopt.pyll.scope.int(hyperopt.hp.quniform('num_iterations', 1, 3000, 1))
                            },
                    "ResLogit":{"input_dim": hyperopt.hp.choice('input_dim', [X_n_cols]),
                             "output_dim": hyperopt.hp.choice('output_dim', [n_alternatives]),
                             "depth": hyperopt.hp.choice('depth', [4, 8, 16, 32]),
                             #"drop": hyperopt.hp.choice('drop', [0.5, 0.3, 0.1]),
                             # TODO: Consider adding the activation functions for the hidden layers (thanh, ReLU, LeakyReLU, etc.)
                             "epochs": hyperopt.hp.choice('epochs', [200]),
                             "batch_size": hyperopt.hp.choice('batch_size', [64, 128, 256]),
                           },
                   }


model_type_to_class = { "LightGBM": LightGBM, 
                        "NN": NN,
                        "DNN": DNN,
                        "ResLogit": "ResLogit",
                        }

STATIC_PARAMS = {'n_jobs': -1}

###############################################################################


## Obtain datasets for K-Fold cross validation (the same fold splits are used across all the iterations for all models)
train_indices = []
test_indices = []

try:
    train_indices, test_indices = pickle.load(open(crossval_pickle_file, "rb"))
    if reset_crossval_indices == 1: # Reset the indices
        raise FileNotFoundError
except (OSError, IOError) as e:
    print("Recomputing Cross-val indices...")
    for (train_index, test_index) in stratified_group_k_fold(X_sample, y_sample, groups, k=CV):
        train_indices.append(train_index)
        test_indices.append(test_index)
    pickle.dump([train_indices, test_indices], open(crossval_pickle_file, "wb"))

def alt_spec_data(dataset):

    dataset['zeros'] = np.zeros_like(dataset['distance'])

    dataset_alt_spec = np.stack(
        [dataset[['distance',  'zeros',    'zeros',    'zeros',    'dur_walking',  'zeros',        'zeros',        'zeros',        'zeros',        'zeros',        'zeros',                'zeros',                'zeros',            'zeros',        'zeros',            'zeros',            'zeros',                    'age', 'female', 'start_time_linear', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']].values,
        dataset[['zeros',      'distance', 'zeros',    'zeros',    'zeros',        'dur_cycling',  'zeros',        'zeros',        'zeros',        'zeros',        'zeros',                'zeros',                'zeros',            'zeros',        'zeros',            'zeros',            'zeros',                    'age', 'female', 'start_time_linear', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']].values,
        dataset[['zeros',      'zeros',    'distance', 'zeros',    'zeros',        'zeros',        'dur_pt_access','zeros',        'dur_pt_rail',  'dur_pt_bus',   'dur_pt_int_waiting',   'dur_pt_int_walking',   'pt_n_interchanges','cost_transit', 'zeros',            'zeros',            'zeros',                    'age', 'female', 'start_time_linear', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']].values,
        dataset[['zeros',      'zeros',    'zeros',    'distance', 'zeros',        'zeros',        'zeros',        'dur_driving',  'zeros',        'zeros',        'zeros',                'zeros',                'zeros',            'zeros',        'cost_driving_fuel','congestion_charge','driving_traffic_percent',  'age', 'female', 'start_time_linear', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']].values]
        )
    
    dataset_alt_spec = np.swapaxes(dataset_alt_spec, 0, 2)
    dataset_alt_spec = np.swapaxes(dataset_alt_spec, 0, 1)

    return dataset_alt_spec


def objective(space):
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # Ignore deprecation warnings

    # Create the classifier
    params = {**space, **STATIC_PARAMS}
    if classifier == "ResLogit":
        pass
    else:
        clf = model_type_to_class[classifier](**params)

    # Applying k-Fold Cross Validation
    loss = 0
    N_sum = 0

    for iteration in range(0, len(train_indices)):

        # Obtain training and testing data for this iteration (split of de k-Fold)
        X_train, X_test = X_sample.loc[train_indices[iteration]], X_sample.loc[test_indices[iteration]]
        y_train, y_test = y_sample.loc[train_indices[iteration]], y_sample.loc[test_indices[iteration]]

        # Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train[scaled_fetures])
        X_train.loc[:, scaled_fetures] = scaler.transform(X_train[scaled_fetures])
        X_test.loc[:, scaled_fetures] = scaler.transform(X_test[scaled_fetures])

        X_train = alt_spec_data(X_train)
        X_test = alt_spec_data(X_test)

        if classifier == "ResLogit":
            clf = ResLogit(X_train, y_train, X_train.shape[1], space["output_dim"], n_layers=space['depth'], batch_size=space['batch_size'], epochs=space['epochs'])
            loss_it, _, _ = clf.train(X_train, y_train, X_test, y_test)
            loss += loss_it * X_test.shape[0]
            N_sum += X_test.shape[0]
        else:
            clf.fit(X_train, y_train)

            proba = clf.clf.predict_proba(X_test)

            # Cross-Entropy Loss
            sum = 0
            i = 0
            for sel_mode in y_test.values:
                sum = sum + np.log(proba[i,sel_mode])
                i += 1
            N = i - 1
            loss += -sum  # Original: (-sum/N) * N
            N_sum += N

    loss = loss / N_sum
    return {'loss': loss, 'status': hyperopt.STATUS_OK}


#%%

# Read a previous adjusted-hyperparameters datafile or create a new one
try:
    adjusted_hyperparameters_file = pd.read_csv(adjusted_hyperparms_dir + hyperparameters_file, index_col=0)
    best_hyperparameters = adjusted_hyperparameters_file.to_dict()
except (OSError, IOError) as e:
    print("Creating new best_hyperparameters structure...")
    best_hyperparameters = {}

for classifier in model_type_to_class.keys():
    print("\n--- %s" % classifier)
    time_ini = time.perf_counter()

    trials = hyperopt.Trials()
    best_classifier = hyperopt.fmin(fn=objective,
                                    space=hyperparameters[classifier],
                                    algo=hyperopt.tpe.suggest,
                                    max_evals=n_iter,
                                    trials=trials)

    
    elapsed_time = time.perf_counter() - time_ini
    print("Tiempo ejecucción: %f" % elapsed_time)
    
    best_hyperparameters[classifier] = best_classifier
    best_hyperparameters[classifier]['_best_loss'] = trials.best_trial["result"]["loss"]
    best_hyperparameters[classifier]['_best_GMPCA'] = np.exp(-trials.best_trial["result"]["loss"])
    best_hyperparameters[classifier]['_elapsed_time'] = elapsed_time

    # Partially store the results (the best hyperparameters)
    best_hyperparameters_df = pd.DataFrame(best_hyperparameters)
    best_hyperparameters_df.to_csv(adjusted_hyperparms_dir + hyperparameters_file, sep=',', index=True)
