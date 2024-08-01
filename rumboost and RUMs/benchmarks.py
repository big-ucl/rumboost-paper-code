from datasets import *
from models import *
import pandas as pd
from utils import cross_entropy, bio_to_rumboost
from rumbooster import rum_train, rum_cv
import biogeme
import lightgbm as lgb

all_models = [SwissMetro, LPMC, Optima, Netherlands, Airplane, Telephone, Parking]
all_datasets = [load_preprocess_SwissMetro, load_preprocess_LPMC]

def estimate_models(models_to_estimate, mixed_logit = False):

    results_tot = []
    for m in models_to_estimate:
        results = m.estimate()
        print(results.shortSummary())
        pandas_results = results.getEstimatedParameters()
        print(pandas_results)
        results_tot.append(results)
        print('Loglikelihood: {}'.format(results.data.logLike))   

    return results_tot

def return_dataset(datasets_to_return, to_return = 'split'):
    
    datasets_train = []
    datasets_test = []
    datasets_full = []
    folds_tot = []
    for d in datasets_to_return:
        data_train, data_test, folds = d()
        data_full = pd.concat([data_train, data_test], ignore_index=True)
        datasets_train.append(data_train)
        datasets_test.append(data_test)
        datasets_full.append(data_full)
        folds_tot.append(folds)

    if to_return == 'split':
        return datasets_train, datasets_test
    elif to_return == 'folds':
        return folds_tot
    elif to_return == 'full':
        return datasets_full
    
def prepare_model(models, datasets, for_prob=False):
    
    bio_tot = [m(datasets[i], for_prob=for_prob) for i, m in enumerate(models)]

    return bio_tot

def prepare_data_validation(dataset_train, folds):
    validation_data = []
    for i, d in enumerate(dataset_train):
        val_dat = []
        for train_idx, test_idx in folds[i]:
            val_dat.append(biogeme.database.EstimationValidation(d.iloc[train_idx], d.iloc[test_idx]))
        validation_data.append(val_dat)

    return validation_data

def prepare_labels(dataset_tests):

    labels = [d['choice'] for d in dataset_tests]
    return labels

def validation(models_to_estimate, results, validation_data):
    CV = []

    for i, m in enumerate(models_to_estimate):
        validation_results = m.validate(results[i], validation_data[i])

        CV.append(validation_results)
        ll_cv=0
        for slide in validation_results:
            ll_cv += slide["Loglikelihood"].sum()
        print(f'Log likelihood for 5 folds of validation data: {ll_cv/5}')


    return CV

def predict_test(results, biosim, labels, return_prob = False):
    ll_test = []
    for i, r in enumerate(results):
        beta = r.getBetaValues()
        simulatedValues = biosim[i].simulate(beta)
        if return_prob:
            return simulatedValues.values
        likelihood = cross_entropy(simulatedValues.to_numpy(), labels[i], likelihood=True)
        print('Likelihood on the test set {}'.format(likelihood))
        ll_test.append(likelihood)

    return ll_test

def predict_proba(X, results, alternatives, utility_functions):
    """
    Predict class probabilities for X.
    """
    X = X.copy()

    database = db.Database('X_test', X)
    globals().update(database.variables)

    # Update previous estimated beta variables
    globals().update(results.getBetaValues())

    V = np.zeros((X.shape[0], len(alternatives)))
    for alt in alternatives:
        V[:, alt] = database.valuesFromDatabase(eval(utility_functions[alt]))

    P = np.zeros((X.shape[0], len(alternatives)))
    for alt in alternatives:
        P[:, alt] = np.exp(V[:, alt]) / np.sum(np.exp(V), axis=1)

    proba = np.where(P > 0.00001, P, 0.00001)
    return proba

def train_rumboost(params, models, dataset_trains, dataset_tests, CV = False, folds = None):
    model_trained_tot = []

    for i, m in enumerate(models):
        # params = {'verbosity': 1,
        #         'num_classes':4,
        #         'learning_rate': 0.1,
        #         'max_depth': 15,
        #         'num_boost_round': 3000,
        #         'objective':'multiclass',
        #         'early_stopping_rounds': 100,
        #         'boosting': 'gbdt',
        #         'monotone_constraints_method': 'advanced',
        #         # 'num_leaves': 132,
        #         # 'min_sum_hessian' : 1.915332651441787,
        #         # #'min_data_in_leaf' : 20,
        #         # 'max_bin' : 226,
        #         # #'min_data_in_bin' : 20,
        #         # 'bagging_fraction' : 0.8260819114210728,
        #         # 'feature_fraction': 0.5675666511288083,
        #         # 'feature_fraction_bynode': 0.8,
        #         # 'lambda_l1': 2.084801999240355,
        #         # 'lambda_l2': 9.796907877709668,
        #         # 'max_delta_step': 1.5716792505797443,
        #         # 'min_gain_to_split': 1.6889182833646266
        #             }
        param = params[i]
        rum_structure = bio_to_rumboost(m)

        # rum_structure[0]['interaction_constraints'] = [[0], [1, 4], [2, 3], [5]]
        # rum_structure[1]['interaction_constraints'] = [[0], [1], [2], [3], [4], [5], [6], [7], [8]]
        # rum_structure[2]['interaction_constraints'] = [[0], [1, 4], [2, 3], [5]]
        # rum_structure[0]['interaction_constraints'] = [[0, 16], [2], [5], [15], [1], [3], [4], [6], [7], [8], [9], [10], [11], [12], [13], [14]]
        # rum_structure[1]['interaction_constraints'] = [[0, 16], [2], [5], [15], [1], [3], [4], [6], [7], [8], [9], [10], [11], [12], [13], [14]]
        # rum_structure[2]['interaction_constraints'] = [[0, 17], [2], [5], [15], [1], [3], [4], [6], [7], [8], [9], [10], [11], [12], [13], [14], [16], [18], [19], [20], [21], [22]]
        # rum_structure[3]['interaction_constraints'] = [[0, 16], [2], [5], [15], [1], [3], [4], [6], [7], [8], [9], [10], [11], [12], [13], [14], [17]]

        target = 'choice'
        train_data = lgb.Dataset(dataset_trains[i], label=dataset_trains[i][target], free_raw_data=False)

        validate_data = lgb.Dataset(dataset_tests[i], label=dataset_tests[i][target], free_raw_data=False)

        if not CV:
            model_trained = rum_train(param,train_data, rum_structure, valid_sets = [validate_data])
            model_trained_tot.append(model_trained)
            print('Likelihood: -on train set {} -on test set {}'.format(model_trained.best_score_train * len(dataset_trains[i]['choice']), model_trained.best_score*len(dataset_tests[i]['choice'])))
        else:
            model_trained = rum_cv(param,train_data, folds=folds[i], rum_structure=rum_structure, return_cvbooster=True)
            model_trained_tot.append(model_trained)
            print(f'Likelihood: -with 5 fold CV- {model_trained["cvbooster"].best_score*len(dataset_trains[i]["choice"])/5}')

    return model_trained_tot

