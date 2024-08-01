from datasets import load_preprocess_LPMC
from models import LPMC
from benchmarks import return_dataset, prepare_model
from function_smoothing import optimal_knots_position, map_x_knots, smooth_predict, updated_utility_collection
from rumbooster import RUMBooster
from utils import cross_entropy, split_mixed_model, bio_to_rumboost
import pandas as pd
import time

import hyperopt
dataset_train, dataset_test = return_dataset([load_preprocess_LPMC], to_return = 'split')
models = [
    'LPMC_RUMBoost',
    'LPMC_RUMBoost_Full_Effect', 
    'LPMC_RUMBoost_Nested',
    ]
bio_model = prepare_model([LPMC], dataset_train)[0]
rnd_effects_attributes = ['age', 'female', 'start_time_linear', 'travel_day', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']


for model in models:
    models_train_RUMB = RUMBooster(model_file='../Data/Results-RealDatasets/' + model + '.json')
    if model == 'LPMC_RUMBoost_Full_Effect':
        models_train_RUMB.random_effects= True
        models_train_RUMB.rum_structure = bio_to_rumboost(bio_model, rnd_effect_attributes= rnd_effects_attributes)
    elif model == 'LPMC_RUMBoost_Nested':
        models_train_RUMB.mu = [1, 1, 1.16674677]
        models_train_RUMB.nest = {0:0, 1:1, 2:2, 3:2}
        models_train_RUMB.rum_structure = bio_to_rumboost(bio_model)
    else:
        models_train_RUMB.rum_structure = bio_to_rumboost(bio_model)

    param_space = {'0':{'distance': hyperopt.pyll.scope.int(hyperopt.hp.quniform('distance_0', 3, 7,1)),
                        'dur_walking': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_walking', 3, 7,1))},
                '1':{'distance': hyperopt.pyll.scope.int(hyperopt.hp.quniform('distance_1', 3, 7,1)),
                        'dur_cycling': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_cycling', 3, 7,1))},
                '2':{'cost_transit': hyperopt.pyll.scope.int(hyperopt.hp.quniform('cost_transit', 3, 7,1)),
                        #'distance': hyperopt.pyll.scope.int(hyperopt.hp.quniform('distance_2', 3, 7,1)),
                        'dur_pt_access': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_pt_access', 3, 7,1)),
                        'dur_pt_bus': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_pt_bus', 3, 7,1)),
                        'dur_pt_int_waiting': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_pt_int_waiting', 3, 7,1)),
                        'dur_pt_int_walking': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_pt_int_walking', 3, 7,1)),
                        'dur_pt_rail': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_pt_rail', 3, 7,1))},
                '3':{'cost_driving_fuel': hyperopt.pyll.scope.int(hyperopt.hp.quniform('cost_driving_fuel', 3, 7,1)),
                        'distance': hyperopt.pyll.scope.int(hyperopt.hp.quniform('distance_3', 3, 7,1)),
                        'driving_traffic_percent': hyperopt.pyll.scope.int(hyperopt.hp.quniform('driving_traffic_percent', 3, 7,1)),
                        'dur_driving': hyperopt.pyll.scope.int(hyperopt.hp.quniform('dur_driving', 3, 7,1))}
                    }

    def objective(space):
        #data_train = dataset_train[0].sample(frac=0.1)
        dataset_train, dataset_test = return_dataset([load_preprocess_LPMC], to_return = 'split')

        if model == 'LPMC_RUMBoost_Full_Effect':
            fe_model, re_model = split_mixed_model(models_train_RUMB)
            weights = fe_model.weights_to_plot_v2()
        else:
            weights = models_train_RUMB.weights_to_plot_v2()

        spline_utilities = {'0':['distance', 'dur_walking'],'1':['distance', 'dur_cycling'], '2':['distance', 'dur_pt_rail', 'dur_pt_bus', 'cost_transit', 'dur_pt_access', 'dur_pt_int_walking', 'dur_pt_int_waiting'], '3': ['distance', 'dur_driving','cost_driving_fuel', 'driving_traffic_percent']}

        target = 'choice'
        features = [col for col in dataset_train[0].columns if col is not target]

        #X_train, X_val, _, y_val = train_test_split(data_train[features], data_train[target], test_size=0.3, random_state=30)

        spline_collection = space

        #x_opt, x_first, x_last = optimal_knots_position(weights, dataset_train[0], dataset_train[0], dataset_train[0][target], spline_utilities, spline_collection, max_iter = 200, optimize = False, deg_freedom=True)
        if model == 'LPMC_RUMBoost_Full_Effect':
            x_opt, x_first, x_last, loss = optimal_knots_position(weights, dataset_train[0], dataset_train[0], dataset_train[0][target], spline_utilities, spline_collection, max_iter = 50, optimize = True, deg_freedom=True, n_iter=1, rde_model=re_model)
        elif model == 'LPMC_RUMBoost_Nested':
            mu = [1, 1, 1.16674677]
            nest = {0:0, 1:1, 2:2, 3:2}
            x_opt, x_first, x_last, loss = optimal_knots_position(weights, dataset_train[0], dataset_train[0], dataset_train[0][target], spline_utilities, spline_collection, max_iter = 50, optimize = True, deg_freedom=True, n_iter=1, mu=mu, nests=nest)
        else:
            x_opt, x_first, x_last, loss = optimal_knots_position(weights, dataset_train[0], dataset_train[0], dataset_train[0][target], spline_utilities, spline_collection, max_iter = 50, optimize = True, deg_freedom=True, n_iter=1)
        
        # x_knots_dict = map_x_knots(x_opt.x, spline_collection, x_first, x_last)

        # util_collection = updated_utility_collection(weights, dataset_train[0], spline_collection, spline_utilities, mean_splines=False, x_knots = x_knots_dict)

        # y_pred = smooth_predict(dataset_test[0], util_collection)

        # loss = cross_entropy(y_pred, dataset_test[0][target])

        return {'loss': loss, 'x_opt':x_opt.x, 'status': hyperopt.STATUS_OK, 'x_first': x_first, 'x_last': x_last}
            #return {'loss': loss, 'x_opt':x_opt, 'status': hyperopt.STATUS_OK}
    n = 25

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

    if model == 'LPMC_RUMBoost_Full_Effect':
        fe_model, re_model = split_mixed_model(models_train_RUMB)
        weights = fe_model.weights_to_plot_v2()
    else:
        weights = models_train_RUMB.weights_to_plot_v2()

    spline_collection = {'0':{'distance': int(best_classifier['distance_0']),
                        'dur_walking': int(best_classifier['dur_walking'])},
                '1':{'distance': int(best_classifier['distance_1']),
                        'dur_cycling': int(best_classifier['dur_cycling'])},
                '2':{'cost_transit': int(best_classifier['cost_transit']),
                        #'distance': int(best_classifier['distance_2']),
                        'dur_pt_access': int(best_classifier['dur_pt_access']),
                        'dur_pt_bus': int(best_classifier['dur_pt_bus']),
                        'dur_pt_int_waiting': int(best_classifier['dur_pt_int_waiting']),
                        'dur_pt_int_walking': int(best_classifier['dur_pt_int_walking']),
                        'dur_pt_rail': int(best_classifier['dur_pt_rail'])},
                '3':{'cost_driving_fuel': int(best_classifier['cost_driving_fuel']),
                        'distance': int(best_classifier['distance_3']),
                        'driving_traffic_percent': int(best_classifier['driving_traffic_percent']),
                        'dur_driving': int(best_classifier['dur_driving'])}
                    }
    #spline_utilities = {'0':['distance', 'dur_walking'],'1':['distance', 'dur_cycling'], '2':['distance', 'dur_pt_rail', 'dur_pt_bus', 'cost_transit', 'dur_pt_int_waiting', 'dur_pt_int_walking', 'dur_pt_access'], '3': ['distance', 'dur_driving','cost_driving_fuel', 'driving_traffic_percent']}
    spline_utilities = {'0':['distance', 'dur_walking'],'1':['distance', 'dur_cycling'], '2':['dur_pt_rail', 'dur_pt_bus', 'cost_transit', 'dur_pt_int_waiting', 'dur_pt_int_walking', 'dur_pt_access'], '3': ['distance', 'dur_driving','cost_driving_fuel', 'driving_traffic_percent']}
    x_knots_dict = map_x_knots(trials.best_trial['result']['x_opt'], spline_collection, trials.best_trial['result']['x_first'], trials.best_trial['result']['x_last'])
    util_collection = updated_utility_collection(weights, dataset_train[0], spline_collection, spline_utilities, mean_splines=False, x_knots = x_knots_dict)
    if model == 'LPMC_RUMBoost_Nested':
        nest = {0:0, 1:1, 2:2, 3:2}
        mu = [1, 1, 1.16674677]
        y_pred = smooth_predict(dataset_test[0], util_collection, mu=mu, nests=nest)
    elif model == 'LPMC_RUMBoost_Full_Effect':
        fe_model, re_model = split_mixed_model(models_train_RUMB)
        y_pred = smooth_predict(dataset_test[0], util_collection, rde_model=re_model)
    else:
        y_pred = smooth_predict(dataset_test[0], util_collection)
    CE_final = cross_entropy(y_pred, dataset_test[0]['choice'])
    best_classifier['_on_test_set'] = CE_final

    # Partially store the results (the best hyperparameters)
    best_num_knots_df = pd.DataFrame({'Number of knots': best_classifier})
    best_num_knots_df.to_csv(f'../Data/Results_PCUF/best_num_knots_{model}.csv', sep=',', index=True)

    #plot_spline(models_train_RUMB, dataset_train[0], spline_collection, utility_names={'0': 'Walking', '1': 'Cycling', '2':'PT', '3':'Driving'}, x_knots_dict = x_knots_dict, save_fig=False)
