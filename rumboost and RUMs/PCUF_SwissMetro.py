from datasets import load_preprocess_SwissMetro
from models import SwissMetro
from benchmarks import return_dataset, prepare_model
from function_smoothing import optimal_knots_position, map_x_knots, smooth_predict, updated_utility_collection
from rumbooster import RUMBooster
from utils import cross_entropy, split_mixed_model, bio_to_rumboost
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


import hyperopt

dataset = load_preprocess_SwissMetro(full_data=True)
hh_id = dataset['ID']
X = dataset.drop(['choice'], axis=1)
y = dataset['choice']

models = 'SwissMetro_RUMBoost_'
bio_model = prepare_model([SwissMetro], [dataset])[0]
rnd_effects_attributes = ['age', 'female', 'start_time_linear', 'travel_day', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']
avg_metrics = {'Cross-entropy loss':[], 'Elapsed time':[]}

sss = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=1)

for i, (train_i, test_i) in enumerate(sss.split(X, y, groups = hh_id)):
    X_train, y_train = X.iloc[train_i], y.iloc[train_i]
    X_test, y_test = X.iloc[test_i], y.iloc[test_i]
    models_train_RUMB = RUMBooster(model_file='../Data/Results-RealDatasets/' + models + str(i) +'_test.json')

    models_train_RUMB.rum_structure = bio_to_rumboost(bio_model)

    param_space = {'0':{'TRAIN_COST': hyperopt.pyll.scope.int(hyperopt.hp.quniform('TRAIN_COST', 3, 7,1)),
                        'TRAIN_TT': hyperopt.pyll.scope.int(hyperopt.hp.quniform('TRAIN_TT', 3, 7,1))},
                '1':{'SM_COST': hyperopt.pyll.scope.int(hyperopt.hp.quniform('SM_COST', 3, 7,1)),
                        'SM_TT': hyperopt.pyll.scope.int(hyperopt.hp.quniform('SM_TT', 3, 7,1))},
                '2':{'CAR_CO': hyperopt.pyll.scope.int(hyperopt.hp.quniform('CAR_COST', 3, 7,1)),
                        'CAR_TT': hyperopt.pyll.scope.int(hyperopt.hp.quniform('CAR_TT', 3, 7,1))}
                    }


    def objective(space):

        weights = models_train_RUMB.weights_to_plot_v2()

        spline_utilities = {'0':['TRAIN_TT', 'TRAIN_COST'], '1':['SM_TT', 'SM_COST'], '2': ['CAR_TT', 'CAR_CO']}

        spline_collection = space

        x_opt, x_first, x_last, loss = optimal_knots_position(weights, X_train, X_train, y_train, spline_utilities, spline_collection, max_iter = 50, optimize = True, deg_freedom=True, n_iter=1)

        return {'loss': loss, 'x_opt':x_opt.x, 'status': hyperopt.STATUS_OK, 'x_first': x_first, 'x_last': x_last}
            #return {'loss': loss, 'x_opt':x_opt, 'status': hyperopt.STATUS_OK}
    n = 50

    time_init = time.perf_counter()

    trials = hyperopt.Trials()
    best_classifier = hyperopt.fmin(fn=objective,
                                    space=param_space,
                                    algo=hyperopt.tpe.suggest,
                                    max_evals=n,
                                    trials=trials)
    
    elapsed_time = time.perf_counter() - time_init

    best_classifier['_best_loss'] = trials.best_trial['result']['loss']
    best_classifier['_elapsed_time'] = elapsed_time
    best_classifier['_x_opt'] = trials.best_trial['result']['x_opt']
    best_classifier['_x_first'] = trials.best_trial['result']['x_first']
    best_classifier['_x_last'] = trials.best_trial['result']['x_last']


    weights = models_train_RUMB.weights_to_plot_v2()

    spline_collection = {'0':{'TRAIN_COST':int(best_classifier['TRAIN_COST']), 'TRAIN_TT':int(best_classifier['TRAIN_TT'])}, '1':{'SM_COST':int(best_classifier['SM_COST']), 'SM_TT':int(best_classifier['SM_TT'])}, '2': {'CAR_CO':int(best_classifier['CAR_COST']), 'CAR_TT':int(best_classifier['CAR_TT'])}}
    spline_utilities = {'0':['TRAIN_TT', 'TRAIN_COST'], '1':['SM_TT', 'SM_COST'], '2': ['CAR_TT', 'CAR_CO']}
    x_knots_dict = map_x_knots(trials.best_trial['result']['x_opt'], spline_collection, trials.best_trial['result']['x_first'], trials.best_trial['result']['x_last'])
    util_collection = updated_utility_collection(weights, X_train, spline_collection, spline_utilities, mean_splines=False, x_knots = x_knots_dict)
    y_pred = smooth_predict(X_test, util_collection)  
    CE_final = cross_entropy(y_pred, y_test)
    best_classifier['_on_test_set'] = CE_final
    avg_metrics['Cross-entropy loss'].append(CE_final)
    avg_metrics['Elapsed time'].append(elapsed_time)
    # Partially store the results (the best hyperparameters)
    best_num_knots_df = pd.DataFrame({'Number of knots': best_classifier})
    best_num_knots_df.to_csv(f'../Data/Results_PCUF/best_num_knots_{models}_{i}_test.csv', sep=',', index=True)

    #plot_spline(models_train_RUMB, dataset_train[0], spline_collection, utility_names={'0': 'Walking', '1': 'Cycling', '2':'PT', '3':'Driving'}, x_knots_dict = x_knots_dict, save_fig=False)

avg_metrics['Cross-entropy loss'].append(np.mean(avg_metrics['Cross-entropy loss']))
avg_metrics['Elapsed time'].append(np.mean(avg_metrics['Elapsed time']))
avg_metrics_df = pd.DataFrame(avg_metrics)
avg_metrics_df.to_csv(f'../Data/Results_PCUF/avg_metrics_{models}_test.csv', sep=',', index=True)
