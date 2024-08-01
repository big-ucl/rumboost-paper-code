import hyperopt
from benchmarks import return_dataset, prepare_model
from datasets import *
from models import *
from utils import bio_to_rumboost
import warnings
import os 
import sys
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
import lightgbm as lgb
import time
from rumbooster import rum_train



# Load the data
dataset_prefix = "LPMC"
data_dir = "../Data/"
adjusted_hyperparms_dir = "../Data/adjusted-hyperparameters/"
dataset_name = dataset_prefix+"_train.csv"
mode_var = "travel_mode"
individual_id = 'household_id'
hyperparameters_file = dataset_prefix+"_hyperparameters"
crossval_pickle_file = data_dir+dataset_prefix+"_hyperparams_crossval.pickle"
reset_crossval_indices = 0 # Set to 0 for reproducibility of the experiment over multiple executions


train = pd.read_csv(data_dir + dataset_name, sep=',')

# Divide the dataset into charasteristics and target variable
X = train.loc[:, train.columns != mode_var]
y = train[mode_var]

# Reduce dataset size to reduce computational cost of the hyperparameter estimation

X_sample = X
y_sample = y

# Extract the individual ID to later group observations using it
groups = np.array(X_sample[individual_id].values)
X_sample = X_sample.drop(columns=individual_id)

X_n_cols = X_sample.shape[1]
n_alternatives = y.nunique()

# Number of iterations of the random search
n_iter = 50
#n_iter = 1 # Uncomment for quick experiments
CV = 5 # Number of cross-validation

hyperparameters_file = hyperparameters_file +'_'+ str(n_iter) + '.csv'

hyperparameters = {#"RUMBoost" : {'learning_rate': hyperopt.hp.choice('learning_rate', [0.1]),
#                                 'max_depth': hyperopt.hp.choice('max_depth', [1]),
#                                 'min_gain_to_split':  hyperopt.hp.loguniform('min_gain_to_split', -9.21, 1.61), # interval: [0.0001, 5]
#                                 'min_sum_hessian_in_leaf': hyperopt.hp.loguniform('min_sum_hessian_in_leaf', 0, 4.6),
#                                 'min_data_in_leaf': hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_data_in_leaf', 1, 200, 1)),
#                                 'bagging_fraction': hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
#                                 'bagging_freq': hyperopt.hp.choice('bagging_freq', [0, 1, 5, 10]),
#                                 'feature_fraction': hyperopt.hp.uniform('feature_fraction', 0.5, 1),
#                                 'lambda_l1': hyperopt.hp.loguniform('lambda_l1', -9.21, 2.30), # interval: [0.0001, 10]
#                                 'lambda_l2': hyperopt.hp.loguniform('lambda_l2', -9.21, 2.30), # interval: [0.0001, 10]
#                                 'max_bin': hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_bin', 100, 500, 1))
#                             },
                        "Nested_RUMBoost" : {
                                'mu': hyperopt.hp.uniform('mu', 1, 2)
                            },
                        "Cross-nested_RUMBoost" : {
                                'mu1': hyperopt.hp.uniform('mu1', 1, 2),
                                'mu2': hyperopt.hp.uniform('mu2', 1, 2),
                                'alpha4': hyperopt.hp.uniform('alpha4', 0, 1)
                            },
                        "Full_Effect_RUMBoost" : {'learning_rate': hyperopt.hp.choice('learning_rate', [0.1]),
                                'num_leaves': hyperopt.pyll.scope.int(hyperopt.hp.quniform('num_leaves', 2, 100, 1)),
                                'min_gain_to_split':  hyperopt.hp.loguniform('min_gain_to_split', -9.21, 1.61), # interval: [0.0001, 5]
                                'min_sum_hessian_in_leaf': hyperopt.hp.loguniform('min_sum_hessian_in_leaf', 0, 4.6),
                                'min_data_in_leaf': hyperopt.pyll.scope.int(hyperopt.hp.quniform('min_data_in_leaf', 1, 200, 1)),
                                'bagging_fraction': hyperopt.hp.uniform('bagging_fraction', 0.5, 1),
                                'bagging_freq': hyperopt.hp.choice('bagging_freq', [0, 1, 5, 10]),
                                'feature_fraction': hyperopt.hp.uniform('feature_fraction', 0.5, 1),
                                'lambda_l1': hyperopt.hp.loguniform('lambda_l1', -9.21, 2.30), # interval: [0.0001, 10]
                                'lambda_l2': hyperopt.hp.loguniform('lambda_l2', -9.21, 2.30), # interval: [0.0001, 10]
                                'max_bin': hyperopt.pyll.scope.int(hyperopt.hp.quniform('max_bin', 100, 500, 1))
                            }
                }


STATIC_PARAMS = {'n_jobs': -1,
                'num_classes':4,
                'objective':'multiclass',
                'boosting': 'gbdt',
                'monotone_constraints_method': 'advanced',
                'verbosity': -1,
                'early_stopping_round':100,
                'num_iterations': 3000,
                'learning_rate':0.1
                }

model_type_to_class = {"RUMBoost": "RUMBoost",
                    "Nested_RUMBoost": "Nested_RUMBoost",
                    #'Cross-nested_RUMBoost':'Cross-nested_RUMBoost',
                    "Full_Effect_RUMBoost":"Full_Effect_RUMBoost" 
                    }

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


dataset_train, dataset_test = return_dataset([load_preprocess_LPMC], to_return = 'split')
models_train = prepare_model([LPMC], dataset_train)

#rum_structure, nests, and mu
rnd_effects_attributes = ['age', 'female', 'start_time_linear', 'travel_day', 'day_of_week', 'car_ownership', 'driving_license', 'purpose_B', 'purpose_HBE', 'purpose_HBO', 'purpose_HBW', 'purpose_NHBO', 'fueltype_Average', 'fueltype_Diesel', 'fueltype_Hybrid', 'fueltype_Petrol']

target = 'choice'
train_data = lgb.Dataset(X_sample, label=y_sample, free_raw_data=False, params={'verbosity':-1})


def objective(space):
    #warnings.filterwarnings(action='ignore', category=DeprecationWarning)  # Ignore deprecation warnings


    # Create the classifier
    
    clf = model_type_to_class[classifier]

    if clf == "RUMBoost":
        params = {**space, **STATIC_PARAMS}
        params['max_depth'] = 1
        rum_structure = bio_to_rumboost(models_train[0])
    elif clf == "Nested_RUMBoost":
        params = {**space, **STATIC_PARAMS}
        params['max_depth'] = 1
        rum_structure = bio_to_rumboost(models_train[0])
        nest = {0:0, 1:1, 2:2, 3:2}
        mu = [1, 1, params.pop("mu")]
    elif clf == "Cross-nested_RUMBoost":
        params = {**space, **STATIC_PARAMS}
        params['max_depth'] = 1
        rum_structure = bio_to_rumboost(models_train[0])
        a4 = params.pop("alpha4")
        alphas = np.array([[0, 1],
                    [0, 1],
                    [1, 0],
                    [a4, 1-a4]])
        mu = [params.pop("mu1"), params.pop("mu2")]
    elif clf == "Full_Effect_RUMBoost":
        rum_structure = bio_to_rumboost(models_train[0], rnd_effect_attributes=rnd_effects_attributes)
        params = {'verbosity': -1,
        'num_classes':4,
        'early_stopping_round':100,
        'learning_rate': space['learning_rate'],
        'max_depth': 1,
        #'num_leaves':31,
        'num_boost_round': 3000,
        'objective':'multiclass',
        'boosting': 'gbdt',
        'monotone_constraints_method': 'advanced'}
        params_rde = {**space, **STATIC_PARAMS}


    # Applying k-Fold Cross Validation
    loss = 0
    N_sum = 0

    for iteration in range(0, len(train_indices)):
        # Obtain training and testing data for this iteration (split of de k-Fold)
        X_train, X_test = X_sample.loc[train_indices[iteration]], X_sample.loc[test_indices[iteration]]
        y_train, y_test = y_sample.loc[train_indices[iteration]], y_sample.loc[test_indices[iteration]]

        train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False, params={'verbosity':-1})
        test_set = lgb.Dataset(X_test, label=y_test, free_raw_data=False, params={'verbosity':-1})

        if clf == "RUMBoost":
            clf_trained = rum_train(params, train_set, rum_structure, valid_sets=[test_set])
        elif clf == "Nested_RUMBoost":
            clf_trained = rum_train(params, train_set, rum_structure, valid_sets=[test_set], nests=nest, mu=mu)
        elif clf == "Cross-nested_RUMBoost":
            clf_trained = rum_train(params, train_set, rum_structure, valid_sets=[test_set], mu=mu, alphas=alphas)
        elif clf == "Full_Effect_RUMBoost":
            clf_trained = rum_train(params, train_set, rum_structure, valid_sets=[test_set], params_rde=params_rde)

        if clf == "Nested_RUMBoost":
            proba, _, _ = clf_trained.predict(test_set, nests=nest, mu=mu)
        if clf == "Cross-nested_RUMBoost":
            proba, _, _ = clf_trained.predict(test_set, mu=mu, alphas=alphas)
        else:
            proba = clf_trained.predict(test_set)

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
    print(f'loss: {loss}, with mu : {mu}, and alphas: {alphas}')
    return {'loss': loss, 'status': hyperopt.STATUS_OK, 'best_iteration': clf_trained.best_iteration}


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
    best_hyperparameters[classifier]['_best_iter'] = trials.best_trial["result"]["best_iteration"]

    # Partially store the results (the best hyperparameters)
    best_hyperparameters_df = pd.DataFrame(best_hyperparameters)
    best_hyperparameters_df.to_csv(adjusted_hyperparms_dir + hyperparameters_file, sep=',', index=True)