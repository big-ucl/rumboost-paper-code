import numpy as np
from scipy.optimize import curve_fit, minimize, shgo, differential_evolution, LinearConstraint, basinhopping
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.special import softmax
from fit_functions import func_wrapper, logistic
from utils import cross_entropy, utility_ranking, nest_probs
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import lightgbm as lgb

def get_mid_pos(data, split_points, end='data'):
    '''
    return midpoint in-between two split points for a specific feature (used in pw linear predict)
    '''
    #getting position in the middle of splitting points intervals
    if len(split_points) > 1:
        mid_pos = [(sp2 + sp1)/2 for sp2, sp1 in zip(split_points[:-1], split_points[1:])]
    else:
        mid_pos = []
    
    if end == 'data':
        mid_pos.insert(0, min(data)) #adding first point
        mid_pos.append(max(data)) #adding last point
    elif end == 'split point':
        mid_pos.insert(0, min(split_points)) #adding first point
        mid_pos.append(max(split_points)) #adding last point
    elif end == 'mean_data':
        mid_pos.insert(0, data[data<split_points[0]].mean())
        mid_pos.append(data[data>split_points[-1]].mean())

    return mid_pos

def get_mean_pos(data, split_points):

    mean_data = [np.mean(data[(data < s_ii) & (data > s_i)]) for s_i, s_ii in zip(split_points[:-1], split_points[1:])]
    mean_data.insert(0, np.mean(data[data<split_points[0]]))
    mean_data.append(np.mean(data[data>split_points[-1]]))

    return mean_data

def data_leaf_value(data, weights_feature, technique='data_weighted'):

    if technique == 'data_weighted':
        data_ordered = np.sort(data)
        idx = np.searchsorted(np.array(weights_feature['Splitting points']), data_ordered)
        data_values = np.array(weights_feature['Histogram values'])[idx]

        return np.array(data_ordered), data_values

    if technique == 'mid_point':
        mid_points = np.array(get_mid_pos(data, weights_feature['Splitting points']))
        return mid_points, np.array(weights_feature['Histogram values'])
    elif technique == 'mean_data':
        mean_data = np.array(get_mean_pos(data, weights_feature['Splitting points']))
        return mean_data, np.array(weights_feature['Histogram values'])

    data_ordered = data.copy().sort_values()
    data_values = [weights_feature['Histogram values'][0]]*sum(data_ordered < weights_feature['Splitting points'][0])

    if technique == 'mid_point_weighted':
        mid_points = get_mid_pos(data, weights_feature['Splitting points'])
        mid_points_weighted = [mid_points[0]]*sum(data_ordered < weights_feature['Splitting points'][0])
    elif technique == 'mean_data_weighted':
        mean_data = get_mean_pos(data, weights_feature['Splitting points'])
        mean_data_weighted = [mean_data[0]]*sum(data_ordered < weights_feature['Splitting points'][0])

    for i, (s_i, s_ii) in enumerate(zip(weights_feature['Splitting points'][:-1], weights_feature['Splitting points'][1:])):
        data_values += [weights_feature['Histogram values'][i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))
        if technique == 'mid_point_weighted':
            mid_points_weighted += [mid_points[i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))
        elif technique == 'mean_data_weighted':
            mean_data_weighted += [mean_data[i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))

    data_values += [weights_feature['Histogram values'][-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
    if technique == 'mid_point_weighted':
        mid_points_weighted += [mid_points[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        return np.array(mid_points_weighted), np.array(data_values)
    elif technique == 'mean_data_weighted':
        mean_data_weighted += [mean_data[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        return np.array(mean_data_weighted), np.array(data_values)

    return np.array(data_ordered), np.array(data_values)

def get_grad(x, y, technique='mid_mean_point', sample_points=30, normalise = False):

    if len(y) <= 1:
        return 0
    
    x_values = x
    y_values = y

    if normalise:
        x_values = (x - np.min(x))/(np.max(x) - np.min(x))
        y_values = (y - np.min(y))/(np.max(y) - np.min(y))

    if technique == 'mid_mean_point'  :
        grad = [(y_values[i+1]-y_values[i])/(x_values[i+1]-x_values[i]) for i in range(0, len(x_values)-1)]
        #grad.insert(0, 0) #adding first slope
        grad.append(0) #adding last slope
    elif technique == 'sample_data':
        x_sample = np.linspace(np.min(x_values), np.max(x_values), sample_points)
        f = interp1d(x_values, y_values, kind='previous')
        y_sample = f(x_sample)
        grad = [(y_sample[i+1]-y_sample[i])/(x_sample[i+1]-x_sample[i]) for i in range(0, len(x_sample)-1)]
        #grad.insert(0, 0) #adding first slope
        grad.append(0) #adding last slope

        if normalise:
            x_sample = x_sample*(np.max(x) - np.min(x)) + np.min(x)
            y_sample = y_sample*(np.max(y) - np.min(y)) + np.min(y)

        return grad, x_sample, y_sample

    return grad

def get_angle_diff(x_values, y_values):

    slope = get_grad(x_values, y_values, normalise = True)
    angle = np.arctan(slope)
    diff_angle = [np.pi - np.abs(angle[0])]
    diff_angle += [np.pi - np.abs(a_1-a_0) for (a_1, a_0) in zip(angle[1:], angle[:-1])]

    return diff_angle

def find_disc(x_values, grad):
    
    diff_angle = get_angle_diff(x_values, grad)

    is_disc = [True if (angle < 0.2) and (np.abs(g) > 5) else False for angle, g in zip(diff_angle, grad)]

    disc = x_values[is_disc]
    disc_idx = np.nonzero(is_disc)[0]
    num_disc = np.sum(is_disc)

    return disc, disc_idx, num_disc
    
def fit_func(data, weight, technique='weighted_data'):

    data_ordered, data_values = data_leaf_value(data, weight, technique)

    best_fit = np.inf
    func_fitted = {}
    fit_score = {}

    func_for_fit = func_wrapper()

    for n, f in func_for_fit.items():
        try:
            param_opt, _, info, _, _ = curve_fit(f, data_ordered, data_values, full_output=True)
        except:
            continue
        func_fitted[n] = param_opt
        fit_score[n] = np.sum(info['fvec']**2)
        if np.sum(info['fvec']**2) < best_fit:
            best_fit = np.sum(info['fvec']**2)
            best_func = n
        print('Fitting residuals for the function {} is: {}'.format(n, np.sum(info['fvec']**2)))

    print('Best fit ({}) for function {}'.format(best_fit, best_func))

    return func_fitted, fit_score

def find_feat_best_fit(model, data, technique='weighted_data'):
    
    weights = model.weights_to_plot_v2()
    best_fit = {}
    for u in weights:
        best_fit[u] = {}
        for f in weights[u]:
            if model.rum_structure[int(u)]['columns'].index(f) in model.rum_structure[int(u)]['categorical_feature']:
                continue #add a function that is basic tree
            func_fitted, fit_score = fit_func(data[f], weights[u][f], technique=technique)
            best_fit[u][f] = {'best_func': min(fit_score, key = fit_score.get), 'best_params': func_fitted[min(fit_score, key = fit_score.get)], 'best_score': min(fit_score)}

    return best_fit

def linear_extrapolator_wrapper(pchip):
    '''
    A wrapper function that adds linear extrapolation to a PchipInterpolator object.

    Parameters
    ----------
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.

    Returns
    -------
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines with linear extrapolation.
    '''
    def pchip_linear_extrapolator(x):
        return np.where(x < pchip.x[0], pchip(pchip.x[0]) + (x - pchip.x[0]) * pchip.derivative()(pchip.x[0]), np.where(x > pchip.x[-1], pchip(pchip.x[-1]) + (x - pchip.x[-1]) * pchip.derivative()(pchip.x[-1]), pchip(x)))

    return pchip_linear_extrapolator

def monotone_spline(x_spline, weights, num_splines=5, x_knots = None, y_knots=None, linear_extrapolation=False):
    '''
    A function that apply monotonic spline interpolation on a given feature.

    Parameters
    ----------
    x : numpy array
        Data from the interpolated feature.
    y : numpy array
        V(x_value), the values of the utility at x.
    num_splines : int, optional (default = 15)
        The number of splines used for interpolation.
    x_knots : numpy array
        The positions of knots. If None, linearly spaced.
    linear_extrapolation : bool, optional (default = False)
        If True, the splines are linearly extrapolated.

    Returns
    -------
    x_spline : numpy array
        A vector of x values used to plot the splines.
    y_spline : numpy array
        A vector of the spline values at x_spline.
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.
    '''

    if x_knots is None:
        x_knots = np.linspace(np.min(x_spline), np.max(x_spline), num_splines+1)
        x_knots, y_knots = data_leaf_value(x_knots, weights)

    is_sorted = lambda a: np.all(a[:-1] < a[1:])
    if not is_sorted(x_knots):
        x_knots = np.sort(x_knots)
    is_equal = lambda a: np.any(a[:-1] == a[1:])
    if is_equal(x_knots):
        first_point = x_knots[0]
        x_knots = [x_ii + 1e-10 if x_i == x_ii else x_ii for x_i, x_ii in zip(x_knots[:-1], x_knots[1:])]
        x_knots.insert(0, first_point)

    # f = interp1d(x, y, kind='previous', fill_value=(y[0], y[-1]), bounds_error=False)
    # y_knots = f(x_knots)

    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    if linear_extrapolation:
        pchip = linear_extrapolator_wrapper(pchip)

    # x_spline = np.linspace(0, np.max(x)*1.05, 10000)
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots

def mean_monotone_spline(x_data, x_mean, y_data, y_mean, num_splines=15):
    '''
    A function that apply monotonic spline interpolation on a given feature.
    The difference with monotone_spline, is that the knots are on the closest stairs mean.

    Parameters
    ----------
    x_data : numpy array
        Data from the interpolated feature.
    x_mean : numpy array
        The x coordinate of the vector of mean points at each stairs
    y_data : numpy array
        V(x_value), the values of the utility at x.
    y_mean : numpy array
        The y coordinate of the vector of mean points at each stairs

    Returns
    -------
    x_spline : numpy array
        A vector of x values used to plot the splines.
    y_spline : numpy array
        A vector of the spline values at x_spline.
    pchip : scipy.interpolate.PchipInterpolator
        The scipy interpolator object from the monotonic splines.
    '''
    #case where there are more splines than mean data points
    if num_splines + 1 >= len(x_mean):
        x_knots = x_mean
        y_knots = y_mean

        #adding first and last point for extrapolation
        if x_knots[0] != x_data[0]:
            x_knots = np.insert(x_knots,0,x_data[0])
            y_knots = np.insert(y_knots,0,y_data[0])

        if x_knots[-1] != x_data[-1]:
            x_knots = np.append(x_knots,x_data[-1])
            y_knots = np.append(y_knots,y_data[-1])

        #create interpolator
        pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

        #for plot
        x_spline = np.linspace(0, np.max(x_data)*1.05, 10000)
        y_spline = pchip(x_spline)

        return x_spline, y_spline, pchip, x_knots, y_knots

    #candidate for knots
    x_candidates = np.linspace(np.min(x_mean)+1e-10, np.max(x_mean)+1e-10, num_splines+1)

    #find closest mean point
    idx = np.unique(np.searchsorted(x_mean, x_candidates, side='left') - 1)

    x_knots = x_mean[idx]
    y_knots = y_mean[idx]

    #adding first and last point for extrapolation
    if x_knots[0] != x_data[0]:
        x_knots = np.insert(x_knots,0,x_data[0])
        y_knots = np.insert(y_knots,0,y_data[0])

    if x_knots[-1] != x_data[-1]:
        x_knots = np.append(x_knots,x_data[-1])
        y_knots = np.append(y_knots,y_data[-1])

    #create interpolator
    pchip = PchipInterpolator(x_knots, y_knots, extrapolate=True)

    #for plot
    x_spline = np.linspace(0, np.max(x_data)*1.05, 10000)
    y_spline = pchip(x_spline)

    return x_spline, y_spline, pchip, x_knots, y_knots

def updated_utility_collection(weights, data, num_splines_feat, spline_utilities, mean_splines=False, x_knots = None, linear_extrapolation=False):
    '''
    Create a dictionary that stores what type of utility (smoothed or not) should be used for smooth_predict.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data : pandas DataFrame
        The pandas DataFrame used for training.
    num_splines_feat : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int. 
        There should be a key for all features where splines are used.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    mean_splines : bool, optional (default = False)
        If True, the splines are computed at the mean distribution of data for stairs.
    x_knots : dict
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding 
        utility and attributes
    linear_extrapolation : bool, optional (default = False)
        If True, the splines are linearly extrapolated.

    Returns
    -------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    '''
    #initialise utility collection
    util_collection = {}

    #for all utilities and features that have leaf values
    for u in weights:
        util_collection[u] = {}
        for f in weights[u]:
            #data points and their utilities
            x_dat, y_dat = data_leaf_value(data[f], weights[u][f])

            #if using splines
            if f in spline_utilities[u]:
                #if mean technique
                if mean_splines:
                    x_mean, y_mean = data_leaf_value(data[f], weights[u][f], technique='mean_data')
                    _, _, func, _, _ = mean_monotone_spline(x_dat, x_mean, y_dat, y_mean, num_splines=num_splines_feat[u][f])
                #else, i.e. linearly sampled points
                else:       
                    x_spline = np.linspace(np.min(data[f]), np.max(data[f]), num=10000)
                    x_knots_temp, y_knots = data_leaf_value(x_knots[u][f], weights[u][f])
                    _, _, func, _, _ = monotone_spline(x_spline, weights, num_splines=num_splines_feat[u][f], x_knots=x_knots_temp, y_knots=y_knots, linear_extrapolation=linear_extrapolation)
            #stairs functions
            else:
                func = interp1d(x_dat, y_dat, kind='previous', bounds_error=False, fill_value=(y_dat[0],y_dat[-1]))

            #save the utility function
            util_collection[u][f] = func
                
    return util_collection

def smooth_predict(data_test, util_collection, utilities=False, mu=None, nests=None, rde_model = None, target='choice'):
    '''
    A prediction function that used monotonic spline interpolation on some features to predict their utilities.
    The function should be used with a trained model only.

    Parameters
    ----------
    data_test : pandas DataFrame
        A pandas DataFrame containing the observations that will be predicted.
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    utilities : bool, optional (default = False)
        if True, return the raw utilities.
        
    Returns
    -------
    U : numpy array
        A numpy array containing the predictions for each class for each observation. Predictions are computed through the softmax function,
        unless the raw utilities are requested. A prediction for class j for observation n will be U[n, j].
    '''
    U = np.array(np.zeros((data_test.shape[0], len(util_collection))))
    for u in util_collection:
        for f in util_collection[u]:
            U[:, int(u)] += util_collection[u][f](data_test[f])
        
    if rde_model is not None:
        U += rde_model.predict(lgb.Dataset(data_test, label=data_test[target], free_raw_data=False), utilities = True)

    #softmax
    if mu is not None:
        preds, _, _ = nest_probs(U, mu, nests)
        return preds
    if not utilities:
        U = softmax(U, axis=1)

    return U

def find_best_num_splines(weights, data_train, data_test, label_test, spline_utilities, mean_splines=False, search_technique='greedy'):
    '''
    Find the best number of splines fro each features prespecified.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data_train : pandas DataFrame
        The pandas DataFrame used for training.
    data_test : pandas DataFrame
        The pandas DataFrame used for testing.
    label_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    mean_splines : bool, optional (default = False)
        If True, the splines are computed at the mean distribution of data for stairs.
    search_technique : str, optional (default = 'greedy')
        The technique used to search for the best number of splines. It can be 'greedy' (i.e., optimise one feature after each other, while storing the feature value),
        'greedy_ranked' (i.e., same as 'greedy' but starts with the feature with the largest utility range) or 'feature_independant'.

    Returns
    -------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    '''
    #initialisation
    spline_range = np.arange(3, 50)
    num_splines = {}
    best_splines = {}
    ce=1000

    #'greedy_ranked' search
    if search_technique == 'greedy_ranked':
        util_ranked = utility_ranking(weights, spline_utilities)
        for rank in util_ranked:
            if num_splines.get(rank[0], None) is None:
                num_splines[rank[0]] = {}
                best_splines[rank[0]] = {}
            for s in spline_range:
                num_splines[rank[0]][rank[1]] = s
                #compute new utility collection 
                utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_splines, spline_utilities=spline_utilities, mean_splines=mean_splines)

                #get new predictions
                smooth_preds = smooth_predict(data_test, utility_collection)

                #compute new CE
                ce_knot = cross_entropy(smooth_preds, label_test)
                
                #store best one
                if ce_knot < ce:
                    ce = ce_knot
                    best_splines[rank[0]][rank[1]] = s
                
                print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, rank[1], ce, best_splines))

            #keep best values for next features
            num_splines = copy.deepcopy(best_splines)
        
        return best_splines, ce
    #'greedy' search
    elif search_technique == 'greedy':
        for u in spline_utilities:
            best_splines[u] = {}
            num_splines[u] = {}
            for f in spline_utilities[u]:
                for s in spline_range:
                    num_splines[u][f] = s
                    #compute new utility collection 
                    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_splines, spline_utilities=spline_utilities, mean_splines=mean_splines)

                    #get new predictions
                    smooth_preds = smooth_predict(data_test, utility_collection)

                    #compute new CE
                    ce_knot = cross_entropy(smooth_preds, label_test)
                    
                    #store best one
                    if ce_knot < ce:
                        ce = ce_knot
                        best_splines[u][f] = s
                    
                    print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, f, ce, best_splines))
                
                #keep best values for next features
                num_splines = copy.deepcopy(best_splines)
            
        return best_splines, ce
    
    #'feature_independant search
    elif search_technique == 'feature_independant':
        for u in spline_utilities:
            best_splines[u] = {}
            for f in spline_utilities[u]:
                ce = 1000
                for s in spline_range:
                    temp_num_splines = {u:{f:s}}
                    #compute new utility collection
                    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=temp_num_splines, spline_utilities=spline_utilities, mean_splines=mean_splines)

                    #get new predictions
                    smooth_preds = smooth_predict(data_test, utility_collection)

                    #compute new CE
                    ce_knot = cross_entropy(smooth_preds, label_test)

                    #store best one
                    if ce_knot < ce:
                        ce = ce_knot
                        best_splines[u][f] = s
                    
                    print("CE = {} at iteration {} for feature {} ---- best CE = {} with best knots: {}".format(ce_knot, s-2, f, ce, best_splines))

        #computation of final cross entropy
        utility_collection_final = updated_utility_collection(weights, data_train, num_splines_feat=best_splines, spline_utilities=spline_utilities, mean_splines=mean_splines)
        smooth_preds_final = smooth_predict(data_test, utility_collection_final)
        ce_final = cross_entropy(smooth_preds_final, label_test)

        return best_splines, ce_final
    
    else:
        raise ValueError('search_technique must be greedy, greedy_ranked, or feature_independant.')
    
def map_x_knots(x_knots, num_splines_range, x_first = None, x_last = None):
    '''
    Map the 1d array of x_knots into a dictionary with utility and attributes as keys.

    Parameters
    ----------
    x_knots : 1d np.array
        The positions of knots in a 1d array, following this structure: 
        np.array([x_att1_1, x_att1_2, ... x_att1_m, x_att2_1, ... x_attn_m]) where m is the number of knots 
        and n the number of attributes that are interpolated with splines.
    num_splines_range: dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int. 
        There should be a key for all features where splines are used.
    x_first : list
        A list of all first knots in the order of the attributes from spline_utilities and num_splines_range.
    x_last : list
        A list of all last knots in the order of the attributes from spline_utilities and num_splines_range.

    Returns
    -------
    x_knots_dict : dict
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding 
        utility and attributes
    '''
    x_knots_dict = {}
    starter = 0
    i=0
    for u in num_splines_range:
        x_knots_dict[u]={}
        for f in num_splines_range[u]:
            if x_first is not None:
                x_knots_dict[u][f] = [x_first[i]]
                x_knots_dict[u][f].extend(x_knots[starter:starter+num_splines_range[u][f]-1])
                x_knots_dict[u][f].append(x_last[i])
                x_knots_dict[u][f] = np.array(x_knots_dict[u][f])
                starter += num_splines_range[u][f]-1
                i +=1
            else:
                x_knots_dict[u][f] = x_knots[starter:starter+num_splines_range[u][f]+1]
                starter += num_splines_range[u][f]+1

    return x_knots_dict
    
def optimise_splines(x_knots, weights, data_train, data_test, label_test, spline_utilities, num_spline_range, x_first = None, x_last = None, deg_freedom = None, mu=None, nests=None, rde_model=None, linear_extrapolation=False):
    '''
    Function wrapper to find the optimal position of knots for each feature. The optimal position is the one
    who minimises the CE loss.

    Parameters
    ----------
    x_knots ; 1d np.array
        The positions of knots in a 1d array, following this structure: 
        np.array([x_att1_1, x_att1_2, ... x_att1_m, x_att2_1, ... x_attn_m]) where m is the number of knots 
        and n the number of attributes that are interpolated with splines.
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    data_train : pandas DataFrame
        The pandas DataFrame used for training.
    data_test : pandas DataFrame
        The pandas DataFrame used for testing.
    label_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    num_splines_range : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int. 
        There should be a key for all features where splines are used.
    x_first : list
        A list of all first knots in the order of the attributes from spline_utilities and num_splines_range.
    x_last : list
        A list of all last knots in the order of the attributes from spline_utilities and num_splines_range.
    deg_freedom : int, optional (default = None)
        The degree of freedom to add to the cross entropy loss.
    mu : numpy array, optional (default = None)
        The nest parameters.
    nests : numpy array, optional (default = None)
        The nest structure.
    rde_model : LightGBM model, optional (default = None)
        The RDE model used for the prediction.
    linear_extrapolation : bool, optional (default = False)
        If True, the splines are linearly extrapolated.

    Returns
    -------
    ce_final: float
        The final cross entropy on the test set.
    '''
    x_knots_dict = map_x_knots(x_knots, num_spline_range, x_first, x_last)

    utility_collection = updated_utility_collection(weights, data_train, num_splines_feat=num_spline_range, spline_utilities=spline_utilities, x_knots=x_knots_dict, linear_extrapolation=linear_extrapolation)
    smooth_preds_final = smooth_predict(data_test, utility_collection, mu=mu, nests=nests, rde_model=rde_model)
    loss = cross_entropy(smooth_preds_final, label_test)
    if deg_freedom is not None:
        N = len(label_test)
        loss = 2 * N * loss + np.log(N) * deg_freedom
    # labels_binary = np.zeros(np.shape(smooth_preds_final))
    # num_data = len(label_test)
    # data_idx = np.arange(num_data)
    # labels_binary[data_idx, label_test] = 1
    # grad = smooth_preds_final-labels_binary
    # return ce_final, grad
    return loss

def optimal_knots_position(weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, max_iter = 100, optimize = True, deg_freedom = None, n_iter=1, x_first=None, x_last=None, mu=None, nests=None, rde_model=None, linear_extrapolation=False):
    '''
    Find the optimal position of knots for a given number of knots for given attributes.

    Parameters
    ----------
    weights : dict
        A dictionary containing all leaf values for all utilities and all features.
    dataset_train : pandas DataFrame
        The pandas DataFrame used for training.
    dataset_test : pandas DataFrame
        The pandas DataFrame used for testing.
    labels_test : pandas Series or numpy array
        The labels of the dataset used for testing.
    spline_utilities : dict
        A dictionary containing attributes where splines are applied. Must be in the form ]
        {utility_indx: [attributes1, attributes2, ...], ...}.
    num_splines_range : dict
        A dictionary of the same format than weights of features names for each utility that are interpolated with monotonic splines.
        The key is a spline interpolated feature name, and the value is the number of splines used for interpolation as an int. 
        There should be a key for all features where splines are used.
    max_iter : int
        The maximum number of iterations from the solver

    Returns
    -------
    x_opt : OptimizeResult
        The result of scipy.minimize.
    '''
    
    ce = 10000000
    # for u in spline_utilities:
    #     for f in spline_utilities[u]:
    #         first_point = np.min(dataset_train[f])
    #         last_point = np.max(dataset_train[f])
    #         #x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(0,num_spline_range[u][f]+1)])
    #         x_0.extend(list(np.linspace(first_point, last_point, num=num_spline_range[u][f]+1)))
    #         cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6} for j in range(0,num_spline_range[u][f])]
    #         last_split_point = weights[u][f]['Splitting points'][-1]
    #         cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2: x[i_knot] - last_split_point - 1e-6})
    #         #cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter: x[i_knot] - first_point})
    #         cons.append({'type':'eq', 'fun': lambda x, i_0 = starter, fp = first_point: x[i_0]-fp, 'keep_feasible' : True})
    #         cons.append({'type':'eq', 'fun': lambda x, i_final = starter + num_spline_range[u][f], lp = last_point: x[i_final]-lp, 'keep_feasible' : True})
    #         bounds = [(first_point, last_point)]*(num_spline_range[u][f]+1)
    #         all_cons.extend(cons)
    #         all_bounds.extend(bounds)
    #         starter += num_spline_range[u][f]+1

    # for u in spline_utilities:
    #     for f in spline_utilities[u]:
    #         x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
    #         #x_0.extend([np.quantile(weights[u][f]['Splitting points'], q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
    #         cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f]-2)]
    #         last_split_point = weights[u][f]['Splitting points'][-1]
    #         first_point = np.min(dataset_train[f])
    #         last_point = np.max(dataset_train[f])
    #         cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
    #         # cons.append({'type':'ineq', 'fun': lambda x, i_0 = starter, fp = first_point: x[i_0] - fp - 1e-6, 'keep_feasible':True})
    #         # cons.append({'type':'ineq', 'fun': lambda x, i_l = starter + num_spline_range[u][f] - 2, lp = last_point: lp - x[i_l] - 1e-6, 'keep_feasible':True})
    #         x_0[starter + num_spline_range[u][f] - 2] = last_split_point + 2e-6
    #         #cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter: x[i_knot] - first_point})
    #         # cons.append({'type':'eq', 'fun': lambda x, i_0 = starter: x[i_0]-first_point})
    #         # cons.append({'type':'eq', 'fun': lambda x, i_final = starter + num_spline_range[u][f]: x[i_final]-last_point})
    #         bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(1, num_spline_range[u][f])]
    #         all_cons.extend(cons)
    #         all_bounds.extend(bounds)
    #         x_first.append(first_point)
    #         x_last.append(last_point)
    #         starter += num_spline_range[u][f]-1

    # for u in num_spline_range:
    #     for f in num_spline_range[u]:
    #         first_split_point = weights[u][f]['Splitting points'][0]
    #         last_split_point = weights[u][f]['Splitting points'][-1]
    #         first_point = np.min(dataset_train[f])
    #         last_point = np.max(dataset_train[f])
    #         x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
    #         #x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(0,num_spline_range[u][f]+1)])
    #         #x_0.extend([np.quantile(weights[u][f]['Splitting points'], q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
    #         #x_0.extend(np.linspace(first_split_point-1e-6, last_split_point+2e-6, num=num_spline_range[u][f]-1))
    #         cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f]-2)]
    #         #cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f])]
    #         cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
    #         # cons.append({'type':'ineq', 'fun': lambda x, i_0 = starter, fp = first_point: x[i_0] - fp - 1e-6, 'keep_feasible':True})
    #         # cons.append({'type':'ineq', 'fun': lambda x, i_l = starter + num_spline_range[u][f] - 2, lp = last_point: lp - x[i_l] - 1e-6, 'keep_feasible':True})
    #         #x_0[starter + num_spline_range[u][f] - 2] = last_split_point + 2e-6
    #         # cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter: x[i_knot] - first_point})
    #         # cons.append({'type':'eq', 'fun': lambda x, i_0 = starter: x[i_0]-first_point})
    #         # cons.append({'type':'eq', 'fun': lambda x, i_final = starter + num_spline_range[u][f]: x[i_final]-last_point})
    #         bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(1, num_spline_range[u][f])]
    #         #bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(0, num_spline_range[u][f] + 1)]
    #         all_cons.extend(cons)
    #         all_bounds.extend(bounds)
    #         x_first.append(first_point)
    #         x_last.append(last_point)
    #         all_lsp.append(last_split_point)
    #         end_points.append(starter + num_spline_range[u][f]-2)
    #         starter += num_spline_range[u][f]-1
    #         #starter += num_spline_range[u][f]+1
    #np.random.seed(1)
    for n in range(n_iter):
        x_0 = []
        if x_first:
            x_first = []
            x_last = []
        all_cons = []
        all_bounds = []
        starter = 0
        # for u in num_spline_range:
        #     for f in num_spline_range[u]:
        #         first_point = np.min(dataset_train[f])
        #         last_point = np.max(dataset_train[f])
        #         regions = [(q)/(num_spline_range[u][f]-1) for q in range(0,num_spline_range[u][f])]
        #         #regions = [(q)/(num_spline_range[u][f]) for q in range(0,num_spline_range[u][f]+1)]
        #         randomness = np.random.rand(num_spline_range[u][f]-2)
        #         last_split_point = weights[u][f]['Splitting points'][-1]
        #         # x_0.extend([np.quantile(dataset_train[f].unique(), q) for q in regions])
        #         x_0.extend([first_point + q*(last_split_point+2e-6-first_point) for q in regions])
        #         x_0.append(last_point)
        #         #x_0[starter + num_spline_range[u][f]-1] = last_split_point + 2e-6
        #         x_0[starter+1:starter+num_spline_range[u][f]-1] = x_0[starter+1:starter+num_spline_range[u][f]-1]*(1 + (randomness-0.5)*0.1*(last_split_point-first_point))
        #         cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-3, 'keep_feasible':True} for j in range(0,num_spline_range[u][f])]
        #         cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 1, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
        #         cons.append({'type':'eq', 'fun': lambda x, i_knot = starter, fp = first_point: x[i_knot] - fp, 'keep_feasible':True})
        #         cons.append({'type':'eq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f], lp = last_point: x[i_knot] - lp, 'keep_feasible':True})
        #         bounds = [(first_point + q*1e-6, last_point-(num_spline_range[u][f]-q)*1e-6) for q in range(0, num_spline_range[u][f]+1)]
        #         all_cons.extend(cons)
        #         all_bounds.extend(bounds)
        #         starter += num_spline_range[u][f]+1

        for u in num_spline_range:
            for f in num_spline_range[u]:
                first_split_point = weights[u][f]['Splitting points'][0]
                last_split_point = weights[u][f]['Splitting points'][-1]
                first_point = np.min(dataset_train[f])
                last_point = np.max(dataset_train[f])

                if x_first:
                    x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
                    cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f]-2)]
                    cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
                    bounds = [(first_point + q*1e-7, last_point + (-num_spline_range[u][f] + 1 + q)*1e-7) for q in range(1, num_spline_range[u][f])]
                    starter += num_spline_range[u][f]-1
                    x_first.append(first_point)
                    x_last.append(last_point)       
                else:
                    x_0.extend([np.quantile(dataset_train[f].unique(), 0.95*q/(num_spline_range[u][f])) for q in range(0,num_spline_range[u][f]+1)])
                    cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f])]
                    bounds = [(first_point + q*1e-7, last_point + (-num_spline_range[u][f] - 1 + q)*1e-7) for q in range(0, num_spline_range[u][f]+1)]
                    starter += num_spline_range[u][f]+1
                #x_0.extend([np.quantile(dataset_train[f].unique(), q/(num_spline_range[u][f])) for q in range(0,num_spline_range[u][f]+1)])
                #x_0.extend([np.quantile(weights[u][f]['Splitting points'], q/(num_spline_range[u][f])) for q in range(1,num_spline_range[u][f])])
                #x_0.extend(np.linspace(first_split_point-1e-6, last_split_point+2e-6, num=num_spline_range[u][f]-1))
                #cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f]-2)]
                #cons = [{'type':'ineq', 'fun': lambda x, i_plus = starter + j+1, i_minus = starter + j: x[i_plus] - x[i_minus] - 1e-6, 'keep_feasible':True} for j in range(0,num_spline_range[u][f])]
                #cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter + num_spline_range[u][f] - 2, lsp = last_split_point: x[i_knot] - lsp - 1e-6, 'keep_feasible':True})
                # cons.append({'type':'ineq', 'fun': lambda x, i_0 = starter, fp = first_point: x[i_0] - fp - 1e-6, 'keep_feasible':True})
                # cons.append({'type':'ineq', 'fun': lambda x, i_l = starter + num_spline_range[u][f] - 2, lp = last_point: lp - x[i_l] - 1e-6, 'keep_feasible':True})
                #x_0[starter + num_spline_range[u][f] - 2] = last_split_point + 2e-6
                # cons.append({'type':'ineq', 'fun': lambda x, i_knot = starter: x[i_knot] - first_point})
                # cons.append({'type':'eq', 'fun': lambda x, i_0 = starter: x[i_0]-first_point})
                # cons.append({'type':'eq', 'fun': lambda x, i_final = starter + num_spline_range[u][f]: x[i_final]-last_point})
                #bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(1, num_spline_range[u][f])]
                #bounds = [(first_point + q*1e-7, last_point - q*1e-7) for q in range(0, num_spline_range[u][f] + 1)]
                all_cons.extend(cons)
                all_bounds.extend(bounds)
                #all_lsp.append(last_split_point)
                #end_points.append(starter + num_spline_range[u][f]-2)
                #starter += num_spline_range[u][f]-1
                #starter += num_spline_range[u][f]+1

        if deg_freedom is not None:
            deg_freedom = starter

        if optimize:
            #x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last), bounds=all_bounds, constraints=all_cons, method='trust-constr', options={'verbose':2, 'maxiter':max_iter, 'gtol':1e-7, 'disp':True})
            #x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom), bounds=all_bounds, constraints=all_cons, method='SLSQP', options={'maxiter':max_iter, 'disp':True})
            x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu, nests, rde_model, linear_extrapolation), bounds=all_bounds, constraints=all_cons, method='SLSQP', options={'maxiter':max_iter, 'disp':True})
            #x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last), jac=True, bounds=all_bounds, constraints=all_cons, method='SLSQP', options={'maxiter':max_iter, 'disp':True})
            #x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last), constraints=all_cons, method='SLSQP', options={'maxiter':max_iter, 'disp':True})
            #x_opt = minimize(optimise_splines, np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range), bounds=all_bounds, constraints=all_cons, method='trust-constr', options={'verbose':2, 'maxiter':max_iter, 'gtol':1e-7})
            #x_opt = shgo(optimise_splines, all_bounds, args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range),callback=callback, constraints=all_cons, minimizer_kwargs = {'method':'SLSQP'}, options={'disp': True} )
            #x_opt = differential_evolution(optimise_splines,x0=np.array(x_0), args = (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last), bounds=all_bounds, constraints=tuple(all_cons))
            #x_opt = basinhopping(optimise_splines,x0=np.array(x_0), niter=50, stepsize = 0.5, T = 0.01, niter_success=50, callback=print_fun, minimizer_kwargs= {'args': (weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last), 'method': 'SLSQP', 'bounds':tuple(all_bounds), 'constraints':tuple(all_cons), 'options':{'maxiter':2}}) #'options':{'disp':True}})


            #final_loss = optimise_splines(x_opt.x, weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom)
            ce_final = optimise_splines(x_opt.x, weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu=mu, nests=nests, rde_model=rde_model)
            if ce_final < ce:
                ce = ce_final
                x_opt_best = x_opt
                x_first_best = x_first
                x_last_best = x_last
            #print(f'Final CE with splines smoothing {final_loss} on {spline_utilities}')
            print(f'{n+1}/{n_iter}:{ce_final} with knots at: {x_opt.x}')
        else:
            final_loss = optimise_splines(np.array(x_0), weights, dataset_train, dataset_test, labels_test, spline_utilities, num_spline_range, x_first, x_last, deg_freedom, mu=mu, nests=nests, rde_model=rde_model)

            if final_loss < ce:
                ce = final_loss
                x_opt_best = x_0
            #print(f'Final CE with splines smoothing {final_loss} on {spline_utilities}')
            print(f'{n+1}/{n_iter}:{final_loss}')

        #return x_opt, x_first, x_last
    if optimize:
        return x_opt_best, x_first_best, x_last_best, ce


    return x_0, ce

def print_fun(x, f, accepted):
    print("at minimum %.4f accepted %d" % (f, int(accepted)))

def plot_spline(model, data_train, spline_collection, utility_names, mean_splines = False, x_knots_dict = None, save_fig = False, lpmc_tt_cost=False, sm_tt_cost=False, save_file=''):
    '''
    Plot the spline interpolation for all utilities interpolated.

    Parameters
    ----------
    model : RUMBoost
        A RUMBoost object.
    data_train : pandas Dataframe
        The full training dataset.
    spline_collection : dict
        A dictionary containing the optimal number of splines for each feature interpolated of each utility
    mean_splines : bool, optional (default = False)
        Must be True if the splines are computed at the mean distribution of data for stairs.
    x_knots_dict : dict
        A dictionary in the form of {utility: {attribute: x_knots}} where x_knots are the spline knots for the corresponding 
        utility and attributes
    '''
    #get weights ordered by features
    weights = model.weights_to_plot_v2()
    tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True, 
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8,
            'scatter.edgecolors': 'none'
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
        #"font.sans-serif": "Computer Modern Roman",
    })

    if lpmc_tt_cost:
        x_plot_w, y_plot_w = data_leaf_value(data_train['dur_walking'], weights['0']['dur_walking'], 'data_weighted')
        y_plot_norm_w = [y - y_plot_w[0] for y in y_plot_w]
        x_spline_w = np.linspace(np.min(data_train['dur_walking']), np.max(data_train['dur_walking']), num=10000)
        x_knots_temp_w, y_knots_w = data_leaf_value(x_knots_dict['0']['dur_walking'], weights['0']['dur_walking'])
        _, y_spline_w, _, x_knot_w, y_knot_w = monotone_spline(x_spline_w, weights['0']['dur_walking'], num_splines=spline_collection['0']['dur_walking'], x_knots=x_knots_temp_w, y_knots=y_knots_w)
        y_spline_norm_w = [y - y_plot_w[0] for y in y_spline_w]
        y_knot_norm_w = [y - y_plot_w[0] for y in y_knot_w]

        

        
        plt.figure(figsize=(3.49, 2.09), dpi=1000)

        #data
        plt.scatter(x_plot_w, y_plot_norm_w, color='b', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_w, y_spline_norm_w, color='b', label=f'Walking travel time ({spline_collection["0"]["dur_walking"]+1} knots)')

        #knots position
        plt.scatter(x_knot_w, y_knot_norm_w, color='k', s=1)

        x_plot_c, y_plot_c = data_leaf_value(data_train['dur_cycling'], weights['1']['dur_cycling'], 'data_weighted')
        y_plot_norm_c = [y - y_plot_c[0] for y in y_plot_c]
        x_spline_c = np.linspace(np.min(data_train['dur_cycling']), np.max(data_train['dur_cycling']), num=10000)
        x_knots_temp_c, y_knots_c = data_leaf_value(x_knots_dict['1']['dur_cycling'], weights['1']['dur_cycling'])
        _, y_spline_c, _, x_knot_c, y_knot_c = monotone_spline(x_spline_c, weights['1']['dur_cycling'], num_splines=spline_collection['1']['dur_cycling'], x_knots=x_knots_temp_c, y_knots=y_knots_c)
        y_spline_norm_c = [y - y_plot_c[0] for y in y_spline_c]
        y_knot_norm_c = [y - y_plot_c[0] for y in y_knot_c]

        #data
        plt.scatter(x_plot_c, y_plot_norm_c, color='r', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_c, y_spline_norm_c, color='r', label=f'Cycling travel time ({spline_collection["1"]["dur_cycling"]+1} knots)')

        #knots position
        plt.scatter(x_knot_c, y_knot_norm_c, color='k', s=1)

        x_plot_p, y_plot_p = data_leaf_value(data_train['dur_pt_rail'], weights['2']['dur_pt_rail'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['dur_pt_rail']), np.max(data_train['dur_pt_rail']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['2']['dur_pt_rail'], weights['2']['dur_pt_rail'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['2']['dur_pt_rail'], num_splines=spline_collection['2']['dur_pt_rail'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'Rail travel time ({spline_collection["2"]["dur_pt_rail"]+1} knots)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['dur_driving'], weights['3']['dur_driving'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['dur_driving']), np.max(data_train['dur_driving']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['3']['dur_driving'], weights['3']['dur_driving'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['3']['dur_driving'], num_splines=spline_collection['3']['dur_driving'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving travel time ({spline_collection["3"]["dur_driving"]+1} knots)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 5])
        plt.xlabel('Travel time  [h]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("../Figures/RUMBoost/LPMC/splines_travel_time.png")
        plt.show()

        plt.figure(figsize=(3.49, 2.09), dpi=1000)

        x_plot_p, y_plot_p = data_leaf_value(data_train['cost_transit'], weights['2']['cost_transit'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['cost_transit']), np.max(data_train['cost_transit']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['2']['cost_transit'], weights['2']['cost_transit'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['2']['cost_transit'], num_splines=spline_collection['2']['cost_transit'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'PT cost ({spline_collection["2"]["cost_transit"]+1} knots)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['cost_driving_fuel'], weights['3']['cost_driving_fuel'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['cost_driving_fuel']), np.max(data_train['cost_driving_fuel']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['3']['cost_driving_fuel'], weights['3']['cost_driving_fuel'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['3']['cost_driving_fuel'], num_splines=spline_collection['3']['cost_driving_fuel'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving cost ({spline_collection["3"]["cost_driving_fuel"]+1} knots)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 10])
        plt.xlabel('Cost [£]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("../Figures/RUMBoost/LPMC/splines_cost.png")
        plt.show()

    if sm_tt_cost:

        x_plot_p, y_plot_p = data_leaf_value(data_train['TRAIN_TT'], weights['0']['TRAIN_TT'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['TRAIN_TT']), np.max(data_train['TRAIN_TT']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['0']['TRAIN_TT'], weights['0']['TRAIN_TT'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['0']['TRAIN_TT'], num_splines=spline_collection['0']['TRAIN_TT'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]
        
        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        #data
        plt.scatter(x_plot_p/60, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p/60, y_spline_norm_p, color='g', label=f'Rail travel time ({spline_collection["0"]["TRAIN_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_p/60, y_knot_norm_p, color='k', s=1)

        x_plot_s, y_plot_s = data_leaf_value(data_train['SM_TT'], weights['1']['SM_TT'], 'data_weighted')
        y_plot_norm_s = [y - y_plot_s[0] for y in y_plot_s]
        x_spline_s = np.linspace(np.min(data_train['SM_TT']), np.max(data_train['SM_TT']), num=10000)
        x_knots_temp_s, y_knots_s = data_leaf_value(x_knots_dict['1']['SM_TT'], weights['1']['SM_TT'])
        _, y_spline_s, _, x_knot_s, y_knot_s = monotone_spline(x_spline_s, weights['1']['SM_TT'], num_splines=spline_collection['1']['SM_TT'], x_knots=x_knots_temp_s, y_knots=y_knots_s)
        y_spline_norm_s = [y - y_plot_s[0] for y in y_spline_s]
        y_knot_norm_s = [y - y_plot_s[0] for y in y_knot_s]
        
        #data
        plt.scatter(x_plot_s/60, y_plot_norm_s, color='#6b8ba4', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_s/60, y_spline_norm_s, color='#6b8ba4', label=f'SwissMetro travel time ({spline_collection["1"]["SM_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_s/60, y_knot_norm_s, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['CAR_TT'], weights['2']['CAR_TT'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['CAR_TT']), np.max(data_train['CAR_TT']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['2']['CAR_TT'], weights['2']['CAR_TT'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['2']['CAR_TT'], num_splines=spline_collection['2']['CAR_TT'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d/60, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d/60, y_spline_norm_d, color='orange', label=f'Driving travel time ({spline_collection["2"]["CAR_TT"]} splines)')

        #knots position
        plt.scatter(x_knot_d/60, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 10])
        plt.xlabel('Travel time [h]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("../Figures/RUMBoost/SwissMetro/splines_travel_time.png")
        plt.show()


        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        x_plot_p, y_plot_p = data_leaf_value(data_train['TRAIN_COST'], weights['0']['TRAIN_COST'], 'data_weighted')
        y_plot_norm_p = [y - y_plot_p[0] for y in y_plot_p]
        x_spline_p = np.linspace(np.min(data_train['TRAIN_COST']), np.max(data_train['TRAIN_COST']), num=10000)
        x_knots_temp_p, y_knots_p = data_leaf_value(x_knots_dict['0']['TRAIN_COST'], weights['0']['TRAIN_COST'])
        _, y_spline_p, _, x_knot_p, y_knot_p = monotone_spline(x_spline_p, weights['0']['TRAIN_COST'], num_splines=spline_collection['0']['TRAIN_COST'], x_knots=x_knots_temp_p, y_knots=y_knots_p)
        y_spline_norm_p = [y - y_plot_p[0] for y in y_spline_p]
        y_knot_norm_p = [y - y_plot_p[0] for y in y_knot_p]

        #data
        plt.scatter(x_plot_p, y_plot_norm_p, color='g', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_p, y_spline_norm_p, color='g', label=f'Rail cost ({spline_collection["0"]["TRAIN_COST"]} splines)')

        #knots position
        plt.scatter(x_knot_p, y_knot_norm_p, color='k', s=1)

        x_plot_s, y_plot_s = data_leaf_value(data_train['SM_COST'], weights['1']['SM_COST'], 'data_weighted')
        y_plot_norm_s = [y - y_plot_s[0] for y in y_plot_s]
        x_spline_s = np.linspace(np.min(data_train['SM_COST']), np.max(data_train['SM_COST']), num=10000)
        x_knots_temp_s, y_knots_s = data_leaf_value(x_knots_dict['1']['SM_COST'], weights['1']['SM_COST'])
        _, y_spline_s, _, x_knot_s, y_knot_s = monotone_spline(x_spline_s, weights['1']['SM_COST'], num_splines=spline_collection['1']['SM_COST'], x_knots=x_knots_temp_s, y_knots=y_knots_s)
        y_spline_norm_s = [y - y_plot_s[0] for y in y_spline_s]
        y_knot_norm_s = [y - y_plot_s[0] for y in y_knot_s]

        #data
        plt.scatter(x_plot_s, y_plot_norm_s, color='#6b8ba4', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_s, y_spline_norm_s, color='#6b8ba4', label=f'SwissMetro cost ({spline_collection["1"]["SM_COST"]} splines)')

        #knots position
        plt.scatter(x_knot_s, y_knot_norm_s, color='k', s=1)

        x_plot_d, y_plot_d = data_leaf_value(data_train['CAR_CO'], weights['2']['CAR_CO'], 'data_weighted')
        y_plot_norm_d = [y - y_plot_d[0] for y in y_plot_d]
        x_spline_d = np.linspace(np.min(data_train['CAR_CO']), np.max(data_train['CAR_CO']), num=10000)
        x_knots_temp_d, y_knots_d = data_leaf_value(x_knots_dict['2']['CAR_CO'], weights['2']['CAR_CO'])
        _, y_spline_d, _, x_knot_d, y_knot_d = monotone_spline(x_spline_d, weights['2']['CAR_CO'], num_splines=spline_collection['2']['CAR_CO'], x_knots=x_knots_temp_d, y_knots=y_knots_d)
        y_spline_norm_d = [y - y_plot_d[0] for y in y_spline_d]
        y_knot_norm_d = [y - y_plot_d[0] for y in y_knot_d]

        #data
        plt.scatter(x_plot_d, y_plot_norm_d, color='orange', s=0.3, alpha=1, edgecolors='none')

        #splines
        plt.plot(x_spline_d, y_spline_norm_d, color='orange', label=f'Driving cost ({spline_collection["2"]["CAR_CO"]} splines)')

        #knots position
        plt.scatter(x_knot_d, y_knot_norm_d, color='k', s=1, label='Knots')

        #plt.title('Spline interpolation of {}'.format(f))
        plt.ylabel('Utility')
        plt.xlim([0, 500])
        plt.xlabel('Cost [chf]')
        plt.legend()
        plt.tight_layout()
        if save_fig:
            plt.savefig("../Figures/RUMBoost/SwissMetro/splines_cost.png")
        plt.show()


    for u in spline_collection:
        for f in spline_collection[u]:
            #data points and their utilities
            x_plot, y_plot = data_leaf_value(data_train[f], weights[u][f], 'data_weighted')
            y_plot_norm = [y - y_plot[0] for y in y_plot]
            x_spline = np.linspace(np.min(data_train[f]), np.max(data_train[f]), num=10000)

            #if using splines
            #if mean technique
            if mean_splines:
                x_mean, y_mean = data_leaf_value(data_train[f], weights[u][f], technique='mean_data')
                x_spline, y_spline, _, x_knot, y_knot = mean_monotone_spline(x_plot, x_mean, y_plot, y_mean, num_splines=spline_collection[u][f])
            #else, i.e. linearly sampled points
            else:
                if x_knots_dict is not None:
                    x_knots_temp, y_knots = data_leaf_value(x_knots_dict[u][f], weights[u][f])
                    _, y_spline, _, x_knot, y_knot = monotone_spline(x_spline, weights[u][f], num_splines=spline_collection[u][f], x_knots=x_knots_temp, y_knots=y_knots)
                else:
                    x_spline, y_spline, _, x_knot, y_knot = monotone_spline(x_plot, y_plot, num_splines=spline_collection[u][f])
            y_spline_norm = [y - y_plot[0] for y in y_spline]
            y_knot_norm = [y - y_plot[0] for y in y_knot]

            
            plt.figure(figsize=(3.49, 2.09), dpi=1000)

            #data
            plt.scatter(x_plot, y_plot_norm, color='k', s=0.3)

            #splines
            plt.plot(x_spline, y_spline_norm, color='#5badc7')

            #knots position
            plt.scatter(x_knot, y_knot_norm, color='#CC5500', s=1)

            plt.legend(['Data', 'Splines', f'Knots ({spline_collection[u][f]+1})'])
            #plt.title('Spline interpolation of {}'.format(f))
            plt.ylabel('{} utility'.format(utility_names[u]))     
            plt.tight_layout()        
            if 'dur' in f:
                plt.xlabel('{} [h]'.format(f))
            elif 'TIME' in f:
                plt.xlabel('{} [h]'.format(f))
            elif 'cost' in f:
                plt.xlabel('{} [£]'.format(f))
            elif 'CO' in f:
                plt.xlabel('{} [chf]'.format(f))
            elif 'distance' in f:
                plt.xlabel('{} [km]'.format(f))
            else:
                plt.xlabel('{}'.format(f))
            if save_fig:
                plt.savefig(save_file + '{} utility, {} feature.png'.format(u, f))
            plt.show()



def compute_VoT(util_collection, u, f1, f2):
    '''
    The function compute the Value of Time of the attributes specified in attribute_VoT.

    Parameters
    ----------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    u : str
        The utility number, as a str (e.g. '0', '1', ...).
    f1 : str
        The time related attribtue name.
    f2 : str
        The cost related attribtue name.

    Return
    ------
    VoT : lamda function
        The function calculating value of time for attribute1 and attribute2. 
    '''

    VoT = lambda x1, x2, u1 = util_collection[u][f1], u2 = util_collection[u][f2]: u1.derivative()(x1) / u2.derivative()(x2)

    return VoT

def plot_VoT(data_train, util_collection, attribute_VoT, utility_names, draw_range, save_figure = False, num_points = 1000):
    '''
    The function plot the Value of Time of the attributes specified in attribute_VoT.

    Parameters
    ----------
    util_collection : dict
        A dictionary containing the type of utility to use for all features in all utilities.
    attribute_VoT : dict
        A dictionary with keys being the utility number (as string) and values being a tuple of the attributes to compute the VoT on.
        The structure follows this form: {utility: (attribute1, attribute2)}
    '''

    tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True, 
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.1,
            "ytick.major.pad": 0.1,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
        #"font.sans-serif": "Computer Modern Roman",
    })
    
    for u in attribute_VoT: 
        f1, f2 = attribute_VoT[u]
        x_vect = np.linspace(draw_range[u][f1][0], draw_range[u][f1][1], num_points)
        y_vect = np.linspace(draw_range[u][f2][0], draw_range[u][f2][1], num_points)
        d_f1 = util_collection[u][f1].derivative()
        d_f2 = util_collection[u][f2].derivative()
        VoT = lambda x1, x2, df1 = d_f1, df2 = d_f2: df1(x1) / df2(x2)
        VoT_contour_plot = np.array(np.zeros((len(x_vect), len(y_vect))))
        X, Y = np.meshgrid(x_vect, y_vect, indexing='ij')
        for i in range(len(x_vect)):
            for j in range(len(y_vect)):
                if d_f2(Y[i, j]) == 0:
                    VoT_contour_plot[i, j] = 100
                elif VoT(X[i, j], Y[i, j]) > 100:
                    VoT_contour_plot[i, j] = 100
                elif VoT(X[i, j], Y[i, j]) < 0.1:
                    VoT_contour_plot[i, j] = 0.1
                else:
                    VoT_contour_plot[i, j] = VoT(X[i, j], Y[i, j])

        fig, axes = plt.subplots(figsize=(3.49,3.49), dpi=1000)

        #fig.suptitle(f'VoT ({f1} and {f2}) of {utility_names[u]}')

        res = 100

        c_plot = axes.contourf(X, Y, np.log(VoT_contour_plot)/np.log(10), levels=res, linewidths=0, cmap=sns.color_palette("Blues", as_cmap=True), vmin = -1, vmax = 2)

        #axes.set_title(f'{utility_names[u]}')
        axes.set_xlabel(f'{f1} [h]')
        axes.set_ylabel(f'{f2} [£]')

        cbar = fig.colorbar(c_plot, ax = axes, ticks=[-1, 0, 1, 2])
        cbar.set_ticklabels([0.1, 1, 10, 100])
        cbar.ax.set_ylabel('VoT [£/h]')
        cbar.ax.set_ylim([-1, 2])

        #plt.tight_layout()

        if save_figure:
            plt.savefig('../Figures/RUMBoost/LPMC/VoT_log_{}.png'.format(utility_names[u]))

        plt.show()


def plot_pop_VoT(data_test, util_collection, attribute_VoT, save_figure = False):

    tex_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True, 
            "font.family": "serif",
            "font.serif": "Computer Modern Roman",
            # Use 14pt font in plots, to match 10pt font in document
            "axes.labelsize": 7,
            "axes.linewidth":0.5,
            "axes.labelpad": 1,
            "font.size": 7,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 6,
            "legend.fancybox": False,
            "legend.edgecolor": "inherit",
            'legend.borderaxespad': 0.4,
            'legend.borderpad': 0.4,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "xtick.major.pad": 0.5,
            "ytick.major.pad": 0.5,
            "grid.linewidth": 0.5,
            "lines.linewidth": 0.8
        }
    sns.set_theme(font_scale=1, rc=tex_fonts)
    #sns.set_context(tex_fonts)
    sns.set_style("whitegrid")
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif"
        #"font.sans-serif": "Computer Modern Roman",
    })

    for u in attribute_VoT: 
        f1, f2 = attribute_VoT[u]
        d_f1 = util_collection[u][f1].derivative()
        d_f2 = util_collection[u][f2].derivative()

        VoT_pop = 60*d_f1(data_test[f1])/d_f2(data_test[f2])

        filtered_VoT_pop = VoT_pop[~np.isnan(VoT_pop)]

        limited_VoT_pop = filtered_VoT_pop[(filtered_VoT_pop>0) & (filtered_VoT_pop < np.quantile(filtered_VoT_pop, 0.99))]

        #fig, axes = plt.subplots(figsize=(10,8), layout='constrained')

        plt.figure(figsize=(3.49, 2.09), dpi=1000)
        sns.histplot(limited_VoT_pop, color='b', alpha = 0.5, kde=True, bins=50)
        plt.xlabel("VoT [chf/h]")
        plt.tight_layout()
        #plt.savefig("Figures/vot_{}.pdf".format(dataset["alt_names"][alt]))
        plt.show()

        if save_figure:
           plt.savefig('../Figures/RUMBoost/SwissMetro/pop_VoT_{}.png'.format(u))



