import pandas as pd
import numpy as np
import random
from scipy.special import softmax
from collections import Counter, defaultdict
from rumbooster import RUMBooster


# Sample a dataset grouped by `groups` and stratified by `y`
# Source: https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices

def cross_entropy(preds, labels, likelihood = False):
    """
    Compute negative cross entropy for given predictions and data.
    
    Parameters
    ----------
    preds: numpy array
        Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
        to the prediction of data point i to belong to class j.
    labels: numpy array
        The labels of the original dataset, as int.

    Returns
    -------
    Cross entropy : float
        The negative cross-entropy, as float.
    """
    num_data = len(labels)
    data_idx = np.arange(num_data)

    if likelihood:
        return np.sum(np.log(np.where(preds[data_idx, labels]>1e-8, preds[data_idx, labels], 1e-8)))
    
    return - np.mean(np.log(np.where(preds[data_idx, labels]>1e-8, preds[data_idx, labels], 1e-8)))

def nest_probs(preds, mu, nests):

    n_obs = np.size(preds, 0)
    data_idx = np.arange(n_obs)
    n_alt = np.size(preds, 1)
    pred_i_m = np.array(np.zeros((n_obs, n_alt)))
    V_tilde_m = np.array(np.zeros((n_obs, len(mu))))
    for alt, nest in nests.items():
        nest_alt = [a for a, n in nests.items() if n == nest]

        pred_i_m[:, alt] = np.exp(mu[nest] * preds[data_idx, alt]) / np.sum(np.exp(mu[nest] * preds[data_idx, :][:, nest_alt]), axis=1)

        V_tilde_m[:, nest] = 1/mu[nest] * np.log(np.sum(np.exp(mu[nest] * preds[data_idx, :][:, nest_alt]), axis=1))

    pred_m = softmax(V_tilde_m, axis=1)

    preds = np.array([pred_i_m[:, i] * pred_m[:, nests[i]] for i in nests.keys()])

    return preds.T, pred_i_m, pred_m

def split_mixed_model(model):

    fixed_effect_model = RUMBooster()
    random_effect_model = RUMBooster()

    fixed_effect_model.boosters = [b for i, b in enumerate(model.boosters) if i%2 == 0]
    fixed_effect_model.rum_structure = model.rum_structure[::2]

    random_effect_model.boosters = [b for i, b in enumerate(model.boosters) if i%2 == 1]
    random_effect_model.rum_structure = model.rum_structure[1::2]

    return fixed_effect_model, random_effect_model
    

def utility_ranking(weights, spline_utilities):
    """
    Rank attributes utility importance by their utility range. The first rank is the attribute having the largest
    max(V(x)) - min(V(x)).

    Parameters
    ----------
    weights : dict
        A dictionary containing all the split points and leaf values for all attributes, for all utilities.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.

    Returns
    -------
    util_ranks_ascend : list of tupple
        A list of tupple where the first tupple is the one having the largest utility range. Tupples are composed of 
        their utility and the name of their attributes.
    """
    util_ranks = []
    util_ranges = []
    for u in spline_utilities:
        for f in spline_utilities[u]:
            #compute range
            util_ranges.append(np.max(weights[u][f]['Histogram values']) - np.min(weights[u][f]['Histogram values']))
            util_ranks.append((u, f))

    sort_idx = np.argsort(util_ranges)
    util_ranks = np.array(util_ranks)
    util_ranks_ascend = util_ranks[np.flip(sort_idx)]

    return util_ranks_ascend

def bio_to_rumboost(model, all_columns = False, monotonic_constraints = True, interaction_contraints = True, rnd_effect_attributes = []):
    '''
    Converts a biogeme model to a rumboost dict.

    Parameters
    ----------
    model : a BIOGEME object
        The model used to create the rumboost structure dictionary.
    all_columns : bool, optional (default = False)
        If True, do not consider alternative-specific features.
    monotonic_constraints : bool, optional (default = True)
        If False, do not consider monotonic constraints.
    interaction_contraints : bool, optional (default = True)
        If False, do not consider feature interactions constraints.
    max_depth : int, optional (default = 1)
        The maximum depth allowed in the RUMBoost object for decision trees.

    Returns
    -------
    rum_structure : dict
        A dictionary specifying the structure of a RUMBoost object.

    '''
    utilities = model.loglike.util #biogeme expression
    rum_structure = []

    #for all utilities
    for k, v in utilities.items():
        rum_structure.append({'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []})
        if len(rnd_effect_attributes) > 0:
            rum_structure_re = {'columns': [], 'monotone_constraints': [], 'interaction_constraints': [], 'betas': [], 'categorical_feature': []}
        for i, pair in enumerate(process_parent(v, [])): # get all the pairs of the utility
            
            if pair[1] in rnd_effect_attributes:
                rum_structure_re['columns'].append(pair[1]) #append variable name
                rum_structure_re['betas'].append(pair[0]) #append beta name
                if interaction_contraints:
                    # if (max_depth > 1) and (('weekend'in pair[0])|('dur_driving' in pair[0])|('dur_walking' in pair[0])|('dur_cycling' in pair[0])|('dur_pt_rail' in pair[0])): #('distance' in pair[0])): |('dur_pt_bus' in pair[0]))
                    #     interac_2d.append(i) #in the case of interaction constraint, append only the relevant continous features to be interacted
                    # else:             
                    #     rum_structure_re['interaction_constraints'].append([i]) #no interaction between features
                    rum_structure_re['interaction_constraints'].append(len(rum_structure_re['interaction_constraints'])) #no interaction between features
                #if ('fueltype' in pair[0]) | ('female' in pair[0]) | ('purpose' in pair[0]) | ('license' in pair[0]) | ('week' in pair[0]):
                    #rum_structure_re['categorical_feature'].append(len(rum_structure_re['categorical_feature'])) #register categorical features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(pair[0]) #get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure_re['monotone_constraints'].append(1) #register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure_re['monotone_constraints'].append(-1) #register negative monotonic constraint
                    else:
                        rum_structure_re['monotone_constraints'].append(0) #none
            
            else:
                rum_structure[-1]['columns'].append(pair[1]) #append variable name
                rum_structure[-1]['betas'].append(pair[0]) #append beta name
                if interaction_contraints:
                    # if (max_depth > 1) and (('weekend'in pair[0])|('dur_driving' in pair[0])|('dur_walking' in pair[0])|('dur_cycling' in pair[0])|('dur_pt_rail' in pair[0])): #('distance' in pair[0])): |('dur_pt_bus' in pair[0]))
                    #     interac_2d.append(i) #in the case of interaction constraint, append only the relevant continous features to be interacted
                    # else:             
                    #     rum_structure[-1]['interaction_constraints'].append([i]) #no interaction between features
                    if len(rnd_effect_attributes) > 0:
                        rum_structure[-1]['interaction_constraints'].append([len(rum_structure[-1]['interaction_constraints'])]) #no interaction between features
                    else:
                        rum_structure[-1]['interaction_constraints'].append([i]) #no interaction between features
                #if ('fueltype' in pair[0]) | ('female' in pair[0]) | ('purpose' in pair[0]) | ('license' in pair[0]) | ('week' in pair[0]):
                #    rum_structure[-1]['categorical_feature'].append(i) #register categorical features
                if monotonic_constraints:
                    bounds = model.getBoundsOnBeta(pair[0]) #get bounds on beta parameter for monotonic constraint
                    if (bounds[0] is not None) and (bounds[1] is not None):
                        raise ValueError("Only one bound can be not None")
                    if bounds[0] is not None:
                        if bounds[0] >= 0:
                            rum_structure[-1]['monotone_constraints'].append(1) #register positive monotonic constraint
                    elif bounds[1] is not None:
                        if bounds[1] <= 0:
                            rum_structure[-1]['monotone_constraints'].append(-1) #register negative monotonic constraint
                    else:
                        rum_structure[k]['monotone_constraints'].append(0) #none
        if len(rnd_effect_attributes) > 0:
            rum_structure.append(rum_structure_re)

    if all_columns:
        rum_structure[-1]['columns'] = [col for col in model.database.data.drop(['choice'], axis=1).columns.values.tolist()]
        # if max_depth > 1:
        #     rum_structure[-1]['interaction_constraints'].append(interac_2d)
        
    return rum_structure

def process_parent(parent, pairs):
    '''
    Dig into the biogeme expression to retrieve name of variable and beta parameter. Work only with simple utility specification (beta * variable).
    '''
    # final expression to be stored
    if parent.getClassName() == 'Times':
        pairs.append(get_pair(parent))
    else: #if not final
        try: #dig into the expression
            left = parent.left
            right = parent.right
        except: #if no left and right children
            return pairs 
        else: #dig further left and right
            process_parent(left, pairs)
            process_parent(right, pairs)
    return pairs

def get_pair(parent):
    '''
    Return beta and variable names on a tupple from a parent expression.
    '''
    left = parent.left
    right = parent.right
    beta = None
    variable = None
    for exp in [left, right]:
        if exp.getClassName() == 'Beta':
            beta = exp.name
        elif exp.getClassName() == 'Variable':
            variable = exp.name
    if beta and variable:
        return (beta, variable)
    else:
        raise ValueError("Parent does not contain beta and variable")


