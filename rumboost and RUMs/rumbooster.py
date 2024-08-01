# coding: utf-8
"""Library with training routines of LightGBM."""
import collections
import copy
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from scipy.optimize import minimize
from multiprocessing import Pool

from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

from lightgbm import callback
from lightgbm.basic import Booster, Dataset, LightGBMError, _ConfigAliases, _InnerPredictor, _choose_param_value, _log_warning
from lightgbm.compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

from scipy.optimize import curve_fit

import concurrent.futures

_LGBM_CustomObjectiveFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[Union[List, np.ndarray], Union[List, np.ndarray]]
]
_LGBM_CustomMetricFunction = Callable[
    [Union[List, np.ndarray], Dataset],
    Tuple[str, float, bool]
]

class RUMBooster:
    """RUMBooster for doing Random Utility Modelling in LightGBM.
    
    Auxiliary data structure to implement boosters of ``rum_train()`` function for multiclass classification.
    This class has the same methods as Booster class.
    All method calls, except for the following methods, are actually performed for underlying Boosters.

    - ``model_from_string()``
    - ``model_to_string()``
    - ``save_model()``

    Attributes
    ----------
    boosters : list of Booster
        The list of underlying fitted models.
    valid_sets : None
        Validation sets of the RUMBooster. By default None, to avoid computing cross entropy if there are no 
        validation sets.
    """
    def __init__(self, model_file = None):
        """Initialize the RUMBooster.

        Parameters
        ----------
        model_file : str, pathlib.Path or None, optional (default=None)
            Path to the RUMBooster model file.
        """
        self.boosters = []
        self.valid_sets = None
        self.mu = None
        self.nests = None
        self.num_classes = None
        self.random_effects = None

        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))

    
    def f_obj(
            self,
            _,
            train_set
        ):
            """
            Objective function of the binary classification boosters, but based on softmax predictions

            Parameters
            ----------
            train_set: Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather
                another dataset containing the relevant features for that utility. It is the jth dataset in the
                RUMBooster object.

            """
            j = self._current_j
            preds = self._preds[:,j]
            factor = self.num_classes/(self.num_classes-1)
            eps = 1e-6
            labels = train_set.get_label()
            grad = preds - labels
            hess = np.maximum(factor * preds * (1 - preds), eps)
            return grad, hess
    
    def f_obj_nest(
            self,
            _,
            train_set
        ):
            """
            Objective function of the binary classification boosters, but based on softmax predictions

            Parameters
            ----------
            train_set: Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather
                another dataset containing the relevant features for that utility. It is the jth dataset in the
                RUMBooster object.

            """
            j = self._current_j
            pred_i_m = self.preds_i_m[:,j]
            pred_m = self.preds_m[:,self.nests[j]]
            eps = 1e-6

            grad = (self.labels == j) * (-self.mu[self.nests[j]] * (1 - pred_i_m) - pred_i_m * (1 - pred_m)) + \
                   (self.labels_nest == self.nests[j]) * (1 - (self.labels == j)) * (self.mu[self.nests[j]] * pred_i_m - pred_i_m * (1 - pred_m)) + \
                   (1 - (self.labels_nest == self.nests[j])) * (pred_i_m * pred_m)
            hess = (self.labels == j) * (np.maximum(-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m), eps)) + \
                   (self.labels_nest == self.nests[j]) * (1 - (self.labels == j)) * (np.maximum(-self.mu[self.nests[j]] * pred_i_m * (1 - pred_i_m) * (1 - self.mu[self.nests[j]] - pred_m) + pred_i_m**2 * pred_m * (1 - pred_m), eps)) + \
                   (1 - (self.labels_nest == self.nests[j])) * (np.maximum(-pred_i_m * pred_m * (pred_i_m * (self.mu[self.nests[j]] - 1) - self.mu[self.nests[j]] - pred_i_m * pred_m), eps))
            
            hess = np.maximum(hess, eps)

            return grad, hess
            
    def predict(
        self,
        data,
        start_iteration: int = 0,
        num_iteration: int = -1,
        raw_score: bool = True,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        utilities: bool = False,
        piece_wise: bool = False,
        nests: bool = False,
        mu = None
    ):
        """Predict logic.

        Parameters
        ----------
        data : str, pathlib.Path, numpy array, pandas DataFrame, H2O DataTable's Frame or scipy.sparse
            Data source for prediction.
            If str or pathlib.Path, it represents the path to a text file (CSV, TSV, or LibSVM).
        start_iteration : int, optional (default=0)
            Start index of the iteration to predict.
        num_iteration : int, optional (default=-1)
            Iteration used for prediction.
        raw_score : bool, optional (default=False)
            Whether to predict raw scores.
        pred_leaf : bool, optional (default=False)
            Whether to predict leaf index.
        pred_contrib : bool, optional (default=False)
            Whether to predict feature contributions.
        data_has_header : bool, optional (default=False)
            Whether data has header.
            Used only for txt data.
        validate_features : bool, optional (default=False)
            If True, ensure that the features used to predict match the ones used to train.
            Used only if data is pandas DataFrame.
        utilities : bool, optional (default=True)
            If True, return raw utilities for each class, without generating probabilities.
        piece_wise: bool, optional (default=False)
            If True, use piece-wise utility instead of stairs utility.

        Returns
        -------
        result : numpy array, scipy.sparse or list of scipy.sparse
            Prediction result.
            Can be sparse or a list of sparse objects (each element represents predictions for one class) for feature contributions (when ``pred_contrib=True``).
        """
        
        #compute utilities with corresponding features
        if piece_wise: #piece-wise utility
            U = self.pw_predict(data, utility= True)
        else:
            #split data
            new_data, _ = self._preprocess_data(data, return_data=True)
            #compute U
            U = [booster.predict(new_data[k].get_data(), 
                                 start_iteration, 
                                 num_iteration, 
                                 raw_score, 
                                 pred_leaf, 
                                 pred_contrib,
                                 data_has_header,
                                 validate_features) for k, booster in enumerate(self.boosters)]

        preds = np.array(U).T

        if self.random_effects:
            preds = preds.reshape((-1, self.num_classes, 2)).sum(axis=2)

        if nests:
            preds, pred_i_m, pred_m = self.nest_probs(preds, mu=mu)
            return preds, pred_i_m, pred_m

        #softmax
        if not utilities:
            preds = softmax(preds, axis=1)
   
        return preds
    
    def _inner_predict(
        self,
        data_idx: int = 0,
        utilities: bool = False,
        nests: bool = False,
        mu = None
    ):
        

        #getting dataset
        alt_pred = [booster._Booster__inner_predict(data_idx) for booster in self.boosters]

        #compute utilities with corresponding features
        preds = np.array(alt_pred).T

        if self.random_effects:
            preds = preds.reshape((-1, self.num_classes, 2)).sum(axis=2)

        if nests:
            preds, pred_i_m, pred_m = self.nest_probs(preds, mu=mu)
            return preds, pred_i_m, pred_m

        #softmax
        if not utilities:
            preds = softmax(preds, axis=1)

        return preds
        
    def nest_probs(self, preds, mu=None):

        if mu is None:
            mu = self.mu
        n_obs = np.size(preds, 0)
        data_idx = np.arange(n_obs)
        n_alt = np.size(preds, 1)
        pred_i_m = np.array(np.zeros((n_obs, n_alt)))
        V_tilde_m = np.array(np.zeros((n_obs, len(mu))))
        for alt, nest in self.nests.items():
            nest_alt = [a for a, n in self.nests.items() if n == nest]

            pred_i_m[:, alt] = np.exp(mu[nest] * preds[data_idx, alt]) / np.sum(np.exp(mu[nest] * preds[data_idx, :][:, nest_alt]), axis=1)

            V_tilde_m[:, nest] = 1/mu[nest] * np.log(np.sum(np.exp(mu[nest] * preds[data_idx, :][:, nest_alt]), axis=1))

        pred_m = softmax(V_tilde_m, axis=1)

        preds = np.array([pred_i_m[:, i] * pred_m[:, self.nests[i]] for i in self.nests.keys()])

        return preds.T, pred_i_m, pred_m

    def optimize_mu(self, mu, train_labels):

        new_preds, _, _ = self._inner_predict(nests=True, mu=mu)
        loss = self.cross_entropy(new_preds, train_labels)

        return loss
    
    def _get_mid_pos(self, data, split_points, end='data'):
        '''
        return midpoint in-between two split points for a specific feature (used in pw linear predict)

        Parameters
        ----------
        data: Panda.Series
            The column of the dataframe associated with the feature
        split_points: list
            The list of split points for that feature
        end: str
            How to compute the mid position of the first and last point, it can be:
                -'data': take min and max values of data
                -'split point': add first and last split points
                -'mean_data': add the mean of data before the first split point, and after the last split point

        Returns
        -------

        mid_pos: list
            a list of the points in the middle of every consecutive split points

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
            mid_pos.insert(0, data[data<split_points[0]].mean()) #adding first point
            mid_pos.append(data[data>split_points[-1]].mean()) #adding last point

        return mid_pos
    
    def _get_slope(self, mid_pos, leaf_values):
        '''
        get slope of the piece-wise utility function for a specific feature
        '''
        if len(leaf_values) <= 1:
            return 0
        
        slope = [(leaf_values[i+1]-leaf_values[i])/(mid_pos[i+1]-mid_pos[i]) for i in range(0, len(mid_pos)-1)]
        slope.insert(0, 0) #adding first slope
        slope.append(0) #adding last slope

        return slope

    def _stairs_to_pw(self, train_data, data_to_transform = None, util_for_plot = False):
        '''
        Transform a stair output to a piecewise linear prediction
        '''
        if type(train_data) is list:
            new_train_data = train_data[0].get_data()
            for data in train_data[1:]:
                new_train_data = new_train_data.join(data.get_data(), lsuffix='DROP').filter(regex="^(?!.*DROP)")

            train_data = new_train_data

        if data_to_transform is None:
            data_to_transform = train_data
        weights = self.weights_to_plot()
        pw_utility = []
        for u in weights:
            if util_for_plot:
                pw_util = []
            else:
                pw_util = np.zeros(data_to_transform.iloc[:, 0].shape)

            for f in weights[u]:
                leaf_values = weights[u][f]['Histogram values']
                split_points =  weights[u][f]['Splitting points']
                transf_data_arr = np.array(data_to_transform[f])

                if len(split_points) < 1:
                    break
                
                if self.rum_structure[int(u)]['columns'].index(f) not in self.rum_structure[int(u)]['categorical_feature']:

                    mid_pos = self._get_mid_pos(self.train_set[int(u)].get_data()[f],split_points)
                    
                    slope = self._get_slope(mid_pos, leaf_values)

                    conds = [(mp1 <= transf_data_arr) & (transf_data_arr < mp2) for mp1, mp2 in zip(mid_pos[:-1], mid_pos[1:])]
                    conds.insert(0, transf_data_arr<mid_pos[0])
                    conds.append(transf_data_arr >= mid_pos[-1])

                    values = [lambda x, j=j: leaf_values[j] + slope[j+1] * (x - mid_pos[j]) for j in range(0, len(leaf_values))]
                    values.insert(0, leaf_values[0])
                else:
                    conds = [(sp1 <= transf_data_arr) & (transf_data_arr < sp2) for sp1, sp2 in zip(split_points[:-1], split_points[1:])]
                    conds.insert(0, transf_data_arr < split_points[0])
                    conds.append(transf_data_arr >= split_points[-1])
                    values = leaf_values
                
                if util_for_plot:
                    pw_util.append(np.piecewise(transf_data_arr, conds, values))
                else:
                    pw_util = np.add(pw_util, np.piecewise(transf_data_arr, conds, values))
            pw_utility.append(pw_util)

        return pw_utility

    def pw_predict(self, data, data_to_transform = None, utility = False):
        '''
        compute predictions with piece-wise utility
        '''
        U = self._stairs_to_pw(data, data_to_transform)

        if utility:
            return U
        
        return softmax(np.array(U).T, axis=1)


    def accuracy(self, preds, labels):
        """
        Compute accuracy of the model

        Parameters
        ----------
        preds: ndarray
            Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
            to the prediction of data point i to belong to class j
        labels: ndarray
            The labels of the original dataset, as int

        Returns
        -------
        Accuracy: float
        """
        return np.mean(np.argmax(preds, axis=1) == labels)
    
    def cross_entropy(self, preds, labels):
        """
        Compute cross entropy of the RUMBooster model for the given predictions and data
        
        Parameters
        ----------
        preds: ndarray
            Predictions for all data points and each classes from a softmax function. preds[i, j] correspond
            to the prediction of data point i to belong to class j
        labels: ndarray
            The labels of the original dataset, as int
        Returns
        -------
        Cross entropy : float
        """
        num_data = len(labels)
        data_idx = np.arange(num_data)

        return - np.mean(np.log(preds[data_idx, labels]))
    
    def _preprocess_data(self, data, reduced_valid_set = None, return_data = False):
        """Set up J training (and, if specified, validation) datasets"""
        train_set_J = []
        reduced_valid_sets_J = []
        train_set_data_J = []
        reduced_valid_sets_data_J = []

        #to access data
        data.construct()
        self.labels = data.get_label()
        
        for j, struct in enumerate(self.rum_structure):
            if struct:
                if 'columns' in struct:
                    train_set_j_data = data.get_data()[struct['columns']] #only relevant features for the j booster
                    if self.random_effects:
                        new_label = np.where(data.get_label() == int(j/2), 1, 0) #new binary label
                    else:
                        new_label = np.where(data.get_label() == j, 1, 0)
                    if self.with_linear_trees:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'linear_trees':True})
                    else:
                        train_set_j = Dataset(train_set_j_data, label=new_label, free_raw_data=False, params={'verbosity':-1})
                    train_set_j.construct()
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = []
                        reduced_valid_sets_data_j = []
                        for valid_set in reduced_valid_set:
                            valid_set.construct()
                            valid_set_j_data = valid_set.get_data()[struct['columns']] #only relevant features for the j booster
                            if self.random_effects:
                                label_valid = np.where(valid_set.get_label() == int(j/2), 1, 0) #new binary label
                            else:
                                label_valid = np.where(valid_set.get_label() == j, 1, 0)
                            valid_set_j = Dataset(valid_set_j_data, label=label_valid, reference= train_set_j, free_raw_data=False, params={'verbosity':-1})
                            valid_set_j.construct()
                            reduced_valid_sets_j.append(valid_set_j)

                else:
                    new_label = np.where(data.get_label() == j, 1, 0)
                    train_set_j = Dataset(data.get_data(), label=new_label, free_raw_data=False)
                    if reduced_valid_set is not None:
                        reduced_valid_sets_j = reduced_valid_set[:]

            train_set_J.append(train_set_j)
            if reduced_valid_set is not None:
                reduced_valid_sets_J.append(reduced_valid_sets_j)

        #storing them in the RUMBooster object
        self.train_set = train_set_J
        self.valid_sets = np.array(reduced_valid_sets_J).T.tolist()
        if return_data:
            return train_set_J, reduced_valid_sets_J
    
    def _preprocess_params(self, params, return_params=False, params_rde=None):
        """Set up J set of parameters"""
        if params_rde is not None:
            params_J = [{**copy.deepcopy(params),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } if i%2 == 0 else 
                        {**copy.deepcopy(params_rde),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } for i, struct in enumerate(self.rum_structure)]
            self.with_linear_trees = params.get('linear_tree', False)
        else:
            params_J = [{**copy.deepcopy(params),
                        'verbosity': -1,
                        'objective': 'binary',
                        'num_classes': 1,
                        'monotone_constraints': struct.get('monotone_constraints', []) if struct else [],
                        'interaction_constraints': struct.get('interaction_constraints', []) if struct else [],
                        'categorical_feature': struct.get('categorical_feature', []) if struct else []
                        } for struct in self.rum_structure]
        
            self.with_linear_trees = params.get('linear_tree', False)

        self.params = params_J
        if return_params:
            return params_J
        
    def _preprocess_valids(self, train_set, params, valid_sets = None, valid_names = None):
        """Set up validation sets"""
        #construct training set to access data
        train_set.construct()

        #initializing variables
        is_valid_contain_train = False
        train_data_name = "training"
        reduced_valid_sets = []
        name_valid_sets = []

        if valid_sets is not None:
            if isinstance(valid_sets, Dataset):
                valid_sets = [valid_sets]
            if isinstance(valid_names, str):
                valid_names = [valid_names]
            for i, valid_data in enumerate(valid_sets):
                # reduce cost for prediction training data
                if valid_data is train_set:
                    is_valid_contain_train = True
                    if valid_names is not None:
                        train_data_name = valid_names[i]
                    continue
                if not isinstance(valid_data, Dataset):
                    raise TypeError("Training only accepts Dataset object")
                reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
                if valid_names is not None and len(valid_names) > i:
                    name_valid_sets.append(valid_names[i])
                else:
                    name_valid_sets.append(f'valid_{i}')

        return reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name     
    
    def _construct_boosters(self, train_data_name = "Training", is_valid_contain_train = False,
                            name_valid_sets = None):
        """Construct boosters of the RUMBooster model with corresponding set of parameters and training features"""
        #getting parameters, training, and validation sets
        params_J = self.params
        train_set_J = self.train_set
        reduced_valid_sets_J = self.valid_sets

        for j, (param_j, train_set_j) in enumerate(zip(params_J, train_set_J)):
            try: 
                #construct binary booster
                booster = Booster(params=param_j, train_set=train_set_j)
                if is_valid_contain_train:
                    booster.set_train_data_name(train_data_name)
                for valid_set, name_valid_set in zip(reduced_valid_sets_J, name_valid_sets):
                    booster.add_valid(valid_set[j], name_valid_set)
            finally:
                train_set_j._reverse_update_params()
                for valid_set in reduced_valid_sets_J:
                    valid_set[j]._reverse_update_params()

            booster.best_iteration = 0
            self._append(booster)

        self.best_iteration = 0
        self.best_score = 1e6
        self.best_score_train = 1e6


    def _append(self, booster: Booster) -> None:
        """Add a booster to RUMBooster."""
        self.boosters.append(booster)

    def _from_dict(self, models: Dict[str, Any]) -> None:
        """Load RUMBooster from dict."""
        self.best_iteration = models["best_iteration"]
        self.best_score = models["best_score"]
        self.boosters = []
        for model_str in models["boosters"]:
            self._append(Booster(model_str=model_str))

    def _to_dict(self, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        """Serialize RUMBooster to dict."""
        models_str = []
        for booster in self.boosters:
            models_str.append(booster.model_to_string(num_iteration=num_iteration, start_iteration=start_iteration,
                                                      importance_type=importance_type))
        return {"boosters": models_str, "best_iteration": self.best_iteration, "best_score": self.best_score}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        """Redirect methods call of RUMBooster."""
        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret
        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def getweights(self, model = None):
        """
        get leaf values from a RUMBooster or LightGBM model

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_df: DataFrame
            DataFrame containing all split points and their corresponding left and right leaves value, 
            for all features
        """
        #using self object or a given model
        if model is None:
            model_json = self.dump_model()
        else:
            model_json = [model.dump_model()]

        weights = []

        for i, b in enumerate(model_json):
            feature_names = b['feature_names']
            for trees in b['tree_info']:
                feature = feature_names[trees['tree_structure']['split_feature']]
                split_point = trees['tree_structure']['threshold']
                left_leaf_value = trees['tree_structure']['left_child']['leaf_value']
                right_leaf_value = trees['tree_structure']['right_child']['leaf_value']
                weights.append([feature, split_point, left_leaf_value, right_leaf_value, i])

        weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
        return weights_df
    
    def _create_name(self, features):
        """Create new feature names from a list of feature names"""
        new_name = features[0]
        for f_name in features[1:]:
            new_name += '-'+f_name
        return new_name
    
    def _get_child(self, weights, weights_2d, weights_market, tree, split_points, features, feature_names, i, market_segm, direction = None):
        """Dig into the tree to get splitting points, features, left and right leaves values"""
        min_r = 0
        max_r = 10000

        if feature_names[tree['split_feature']] not in features:
            features.append(feature_names[tree['split_feature']])

        split_points.append(tree['threshold'])

        # if features[-1] in self.rum_structure[i]['categorical_feature']:
        #     market_segm = True

        if 'leaf_value' in tree['left_child'] and 'leaf_value' in tree['right_child']:
            if direction is None:
                weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
            elif direction == 'left':
                if len(features) == 1:
                    weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    weights.append([feature_names[tree['split_feature']], split_points[0], 0, -tree['right_child']['leaf_value'], i])
                elif market_segm:
                    feature_name = self._create_name(features)
                    if features[0] in self.rum_structure[i]['categorical_feature']:
                        weights_market.append([features[-1]+'-0', tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    else:
                        weights_market.append([features[0]+'-0', split_points[0], tree['left_child']['leaf_value'], 0, i])
                        weights_market.append([features[0]+'-1', split_points[0], tree['right_child']['leaf_value'], 0, i])
                else:
                    feature_name = self._create_name(features)
                    weights_2d.append([feature_name, (min_r, split_points[0]), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                    weights_2d.append([feature_name, (min_r, split_points[0]), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
                    features.pop(-1)
                    split_points.pop(-1)
            elif direction == 'right':
                if len(features) == 1:
                    weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    weights.append([feature_names[tree['split_feature']], split_points[0], -tree['left_child']['leaf_value'], 0, i])
                elif market_segm:
                    feature_name = self._create_name(features)
                    if features[0] in self.rum_structure[i]['categorical_feature']:
                        weights_market.append([features[-1]+'-1', tree['threshold'], tree['left_child']['leaf_value'], tree['right_child']['leaf_value'], i])
                    else:
                        weights_market.append([features[0]+'-0', split_points[0], 0, tree['left_child']['leaf_value'], i])
                        weights_market.append([features[0]+'-1', split_points[0], 0, tree['right_child']['leaf_value'], i])
                else:
                    feature_name = self._create_name(features)
                    weights_2d.append([feature_name, (split_points[0], max_r), (min_r, tree['threshold']), tree['left_child']['leaf_value'], i])
                    weights_2d.append([feature_name, (split_points[0], max_r), (tree['threshold'], max_r), tree['right_child']['leaf_value'], i])
                    features.pop(-1)
                    split_points.pop(-1)
        elif 'leaf_value' in tree['left_child']:
            weights.append([feature_names[tree['split_feature']], tree['threshold'], tree['left_child']['leaf_value'], 0, i])
            self._get_child(weights, weights_2d, weights_market, tree['right_child'], split_points, features, feature_names, i, market_segm, direction='right')
        elif 'leaf_value' in tree['right_child']:
            weights.append([feature_names[tree['split_feature']], tree['threshold'], 0, tree['right_child']['leaf_value'], i])
            self._get_child(weights, weights_2d, weights_market, tree['left_child'], split_points, features, feature_names, i, market_segm, direction='left')
        else:
            self._get_child(weights, weights_2d, weights_market, tree['left_child'], split_points, features, feature_names, i, market_segm, direction='left')
            self._get_child(weights, weights_2d, weights_market, tree['right_child'], split_points, features, feature_names, i, market_segm, direction='right') 

    def getweights_v2(self, model = None):
        """
        get leaf values from a RUMBooster or LightGBM model

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_df: DataFrame
            DataFrame containing all split points and their corresponding left and right leaves value, 
            for all features
        """
        #using self object or a given model
        if model is None:
            model_json = self.dump_model()
        else:
            model_json = [model.dump_model()]

        weights = []
        weights_2d = []
        weights_market = []

        for i, b in enumerate(model_json):
            feature_names = b['feature_names']
            for trees in b['tree_info']:
                features = []
                split_points = []
                market_segm = False

                if "split_feature" not in trees['tree_structure']:
                    continue

                self._get_child(weights, weights_2d, weights_market, trees['tree_structure'], split_points, features, feature_names, i, market_segm)

        weights_df = pd.DataFrame(weights, columns= ['Feature', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
        weights_2d_df = pd.DataFrame(weights_2d, columns=['Feature', 'higher_lvl_range', 'lower_lvl_range', 'area_value', 'Utility'])
        weights_market_df = pd.DataFrame(weights_market, columns= ['Feature', 'Cat value', 'Split point', 'Left leaf value', 'Right leaf value', 'Utility'])
        return weights_df, weights_2d_df, weights_market_df
    
    def weights_to_plot_v2(self, model = None, market_segm=False):
        """
        Arrange weights by ascending splitting points and cumulative sum of weights

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_for_plot: dict
            Dictionary containing splitting points and corresponding cumulative weights value for all features
        """

        #get raw weights
        if market_segm:
            _, _, weights= self.getweights_v2()
        else:
            if model is None:
                weights, _, _= self.getweights_v2()
            else:
                weights, _, _ = self.getweights_v2(model=model)

        weights_for_plot = {}
        #for all features
        for i in weights.Utility.unique():
            weights_for_plot[str(i)] = {}
            
            for f in weights[weights.Utility == i].Feature.unique():
                
                split_points = []
                function_value = [0]

                #getting values related to the corresponding utility
                weights_util = weights[weights.Utility == i]
                
                #sort by ascending order
                feature_data = weights_util[weights_util.Feature == f]
                ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
                for j, s in enumerate(ordered_data['Split point']):
                    #new split point
                    if s not in split_points:
                        split_points.append(s)
                        #add a new right leaf value to the current right side value
                        function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    else:
                        #add right leaf value to the current right side value
                        function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                        
                weights_for_plot[str(i)][f] = {'Splitting points': split_points,
                                            'Histogram values': function_value}
                    
        return weights_for_plot

    def weights_to_plot(self, model = None):
        """
        Arrange weights by ascending splitting points and cumulative sum of weights

        Parameters
        ----------
        model: lightGBM model

        Returns
        -------
        weights_for_plot: dict
            Dictionary containing splitting points and corresponding cumulative weights value for all features
        """

        #get raw weights
        if model is None:
            weights = self.getweights()
        else:
            weights = self.getweights(model=model)

        weights_for_plot = {}
        weights_for_plot_double = {}
        #for all features
        for i in weights.Utility.unique():
            weights_for_plot[str(i)] = {}
            weights_for_plot_double[str(i)] = {}
            
            for f in weights[weights.Utility == i].Feature.unique():
                
                split_points = []
                function_value = [0]

                #getting values related to the corresponding utility
                weights_util = weights[weights.Utility == i]
                
                #sort by ascending order
                feature_data = weights_util[weights_util.Feature == f]
                ordered_data = feature_data.sort_values(by = ['Split point'], ignore_index = True)
                for j, s in enumerate(ordered_data['Split point']):
                    #new split point
                    if s not in split_points:
                        split_points.append(s)
                        #add a new right leaf value to the current right side value
                        function_value.append(function_value[-1] + float(ordered_data.loc[j, 'Right leaf value']))
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                    else:
                        #add right leaf value to the current right side value
                        function_value[-1] += float(ordered_data.loc[j, 'Right leaf value'])
                        #add left leaf value to all other current left leaf values
                        function_value[:-1] = [h + float(ordered_data.loc[j, 'Left leaf value']) for h in function_value[:-1]]
                        
                weights_for_plot[str(i)][f] = {'Splitting points': split_points,
                                               'Histogram values': function_value}
                    
        return weights_for_plot
    
    def non_lin_function(self, weights_ordered, x_min, x_max, num_points):
        """
        Create the nonlinear function for parameters, from weights ordered by ascending splitting points

        Parameters
        ----------
        weights_ordered : dict
            Dictionary containing splitting points and corresponding cumulative weights value for a specific 
            feature's parameter
        x_min : float, int
            Minimum x value for which the nonlinear function is computed
        x_max : float, int
            Maximum x value for which the nonlinear function is computed
        num_points: int
            Number of points used to draw the nonlinear function line

        Returns
        -------
        x_values: list
            X values for which the function will be plotted
        nonlin_function: list
            Values of the function at the corresponding x points
        """
        #create x points
        x_values = np.linspace(x_min, x_max, num_points)
        nonlin_function = []
        i = 0
        max_i = len(weights_ordered['Splitting points']) #all splitting points

        #handling no split points
        if max_i == 0:
            return x_values, float(weights_ordered['Histogram values'][i])*x_values

        for x in x_values:
            #compute the value of the function at x according to the weights value in between splitting points
            if x < float(weights_ordered['Splitting points'][i]):
                nonlin_function += [float(weights_ordered['Histogram values'][i])]
            else:
                nonlin_function += [float(weights_ordered['Histogram values'][i+1])]
                #go to next splitting points
                if i < max_i-1:
                    i+=1
        
        return x_values, nonlin_function
    
    def get_asc(self, weights, alt_to_normalise = 'Driving', alternatives = {'Walking':'0', 'Cycling':'1', 'Public Transport':'2', 'Driving':'3'}):
        '''Retrieve ASCs from a dictionary of all values from a dictionary of leaves values per alternative per feature'''
        ASCs = []
        for k, alt in alternatives.items():
            asc_temp = 0
            for feat in weights[alt]:
                asc_temp += weights[alt][feat]['Histogram values'][0]
            ASCs.append(asc_temp)

        return [a - ASCs[int(alternatives[alt_to_normalise])] for a in ASCs]

    def function_2d(self, weights_2d, x_vect, y_vect):
        """
        Create the nonlinear contour plot for parameters, from weights gathered in getweights_v2

        Parameters
        ----------
        weights_2d : dict
            Pandas DataFrame containing all possible rectangles with their corresponding area values, for the given feature and utility
        x_vect : np.linspace
            Vector of higher level feature
        y_vect : np.linspace
            Vector of lower level feature

        Returns
        -------
        contour_plot_values: np.darray
            Array with values at (x,y) points
        """
        contour_plot_values = np.zeros(shape=(len(x_vect), len(y_vect)))

        for k in range(len(weights_2d.index)):
            for i, x in enumerate(x_vect):
                if (x >= weights_2d['higher_lvl_range'].iloc[k][1]):
                    break

                if x >= weights_2d['higher_lvl_range'].iloc[k][0]:
                    for j, y in enumerate(y_vect):
                        if y >= weights_2d['lower_lvl_range'].iloc[k][1]:
                            break
                        if y >= weights_2d['lower_lvl_range'].iloc[k][0]:
                            if (weights_2d['lower_lvl_range'].iloc[k][1] == 10000) and (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                                contour_plot_values[i:, j:] += weights_2d['area_value'].iloc[k]
                                break
                            elif (weights_2d['lower_lvl_range'].iloc[k][1] == 10000):
                                contour_plot_values[i, j:] += weights_2d['area_value'].iloc[k]
                            elif (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                                contour_plot_values[i:, j] += weights_2d['area_value'].iloc[k]
                            else:
                                contour_plot_values[i, j] += weights_2d['area_value'].iloc[k]

        return contour_plot_values
    
    def function_2d_v2(self, weights_2d, x_vect, y_vect):
        """
        Create the nonlinear contour plot for parameters, from weights gathered in getweights_v2

        Parameters
        ----------
        weights_2d : dict
            Pandas DataFrame containing all possible rectangles with their corresponding area values, for the given feature and utility
        x_vect : np.linspace
            Vector of higher level feature
        y_vect : np.linspace
            Vector of lower level feature

        Returns
        -------
        contour_plot_values: np.darray
            Array with values at (x,y) points
        """
        contour_plot_values = np.zeros(shape=(len(x_vect), len(y_vect)))

        for k in range(len(weights_2d.index)):
            if (weights_2d['lower_lvl_range'].iloc[k][1] == 10000) and (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][0])
                i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][0])

                contour_plot_values[i_x:, i_y:] += weights_2d['area_value'].iloc[k]

            elif (weights_2d['lower_lvl_range'].iloc[k][1] == 10000):
                i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][1])
                i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][0])

                contour_plot_values[:i_x, i_y:] += weights_2d['area_value'].iloc[k]

            elif (weights_2d['higher_lvl_range'].iloc[k][1] == 10000):
                i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][0])
                i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][1])
                
                contour_plot_values[i_x:, :i_y] += weights_2d['area_value'].iloc[k]

            else:
                i_x = np.searchsorted(x_vect, weights_2d['higher_lvl_range'].iloc[k][1])
                i_y = np.searchsorted(y_vect, weights_2d['lower_lvl_range'].iloc[k][1])
                
                contour_plot_values[:i_x, :i_y] += weights_2d['area_value'].iloc[k]

        return contour_plot_values

    def plot_2d(self, feature1, feature2, min1, max1, min2, max2, num_points=100, model = None, save_figure=False):
        '''
        plot a 2d feature interaction as a contour plot.

        Parameters
        ----------
        feature1: str
            name of feature 1
        feature2: str
            name of feature 2
        max1: int
            maximum value of feature 1
        max2: int
            maximum value of feature 2
        model: RUMBoost, optional
            if specifies, use an existing model for the plot
        '''
        _, weights_2d, _ = self.getweights_v2(model = model)
        weights_ordered = self.weights_to_plot_v2()

        name1 = feature1 + "-" + feature2
        name2 = feature2 + "-" + feature1

        x_vect = np.linspace(min1, max1, num_points)
        y_vect = np.linspace(min2, max2, num_points)

        #to generalise
        utility_names = ['Walking', 'Cycling', 'PT', 'Driving']
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

        # fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,10), layout='constrained')
        # sns.set_theme()
        # fig.suptitle('Impact of {} and {} on the utility function'.format(feature1, feature2))
        # axes = axes.ravel()
        # for u in weights_2d.Utility.unique():
        #     weights_2d_util = weights_2d[weights_2d.Utility==u]
        #     contour_plot1 = self.function_2d(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
        #     contour_plot2 = self.function_2d(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)

        #     contour_plot = contour_plot1 + contour_plot2.T

        #     X, Y = np.meshgrid(x_vect, y_vect)

        #     c_plot = axes[int(u)].contourf(X, Y, contour_plot, levels=100, linewidths=0, cmap=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True), vmin=-6, vmax=4)

        #     axes[int(u)].set_title('{}'.format(utility_names[int(u)]))
        #     axes[int(u)].set_xlabel('{}'.format(feature1))
        #     axes[int(u)].set_ylabel('{}'.format(feature2))

        #     if int(u)==1:
        #         cbar = fig.colorbar(c_plot, ax = axes[int(u)])
        #         cbar.ax.set_ylabel('Utility value')

        for u in weights_2d.Utility.unique():
            weights_2d_util = weights_2d[weights_2d.Utility==u]
            contour_plot1 = self.function_2d_v2(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
            contour_plot2 = self.function_2d_v2(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)

            contour_plot = contour_plot1 + contour_plot2.T

            if np.sum(contour_plot) == 0:
                continue

            if (feature1 in weights_ordered[str(u)].keys()) and (feature2 in weights_ordered[str(u)].keys()):
                _, feature1_alone = self.non_lin_function(weights_ordered[str(u)][feature1], min1, max1, num_points)
                feature1_grid = np.repeat(feature1_alone, num_points).reshape((num_points, num_points))
                contour_plot += feature1_grid

                _, feature2_alone = self.non_lin_function(weights_ordered[str(u)][feature2], min2, max2, num_points)
                feature2_grid = np.repeat(feature2_alone, num_points).reshape((num_points, num_points)).T
                contour_plot += feature2_grid

            contour_plot -= contour_plot.max()

            colors = ['#F5E5E2', '#DF7057', '#A31D04']
            # Set your custom color palette
            customPalette = sns.set_palette(sns.color_palette(colors, as_cmap=True))

            if np.sum(contour_plot) != 0:
                X, Y = np.meshgrid(x_vect, y_vect)
                fig, axes = plt.subplots(figsize=(3.49,3), layout='constrained', dpi=1000)

                #fig.suptitle('Impact of {} and {} on the utility function'.format(feature1, feature2))

                res = 100

                c_plot = axes.contourf(X, Y, contour_plot.T, levels=res, linewidths=0, cmap=customPalette, vmin=-12, vmax=0)


                #axes.set_title(f'{utility_names[int(u)]}')
                axes.set_xlabel(f'{feature1} [h]')
                axes.set_ylabel(f'{feature2}')

                cbar = fig.colorbar(c_plot, ax = axes, ticks=[-10, -8, -6, -4, -2, 0])
                cbar.ax.set_ylabel('Utility')

                if save_figure:
                    plt.savefig('../Figures/FI RUMBoost/age_travel_time_{}.png'.format(utility_names[int(u)]))

                plt.show()

    def plot_2d_detailed(self, feature1, feature2, min1, max1, min2, max2, num_points=100, model = None, save_figure=False):
        '''
        plot a 2d feature interaction as a contour plot.

        Parameters
        ----------
        feature1: str
            name of feature 1
        feature2: str
            name of feature 2
        max1: int
            maximum value of feature 1
        max2: int
            maximum value of feature 2
        model: RUMBoost, optional
            if specifies, use an existing model for the plot
        '''
        _, weights_2d, _ = self.getweights_v2(model = model)
        weights_ordered = self.weights_to_plot_v2()

        name1 = feature1 + "-" + feature2
        name2 = feature2 + "-" + feature1

        x_vect = np.linspace(min1, max1, num_points)
        y_vect = np.linspace(min2, max2, num_points)

        #to generalise
        utility_names = ['Walking', 'Cycling', 'PT', 'Driving']
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

        # fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,10), layout='constrained')
        # sns.set_theme()
        # fig.suptitle('Impact of {} and {} on the utility function'.format(feature1, feature2))
        # axes = axes.ravel()
        # for u in weights_2d.Utility.unique():
        #     weights_2d_util = weights_2d[weights_2d.Utility==u]
        #     contour_plot1 = self.function_2d(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
        #     contour_plot2 = self.function_2d(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)

        #     contour_plot = contour_plot1 + contour_plot2.T

        #     X, Y = np.meshgrid(x_vect, y_vect)

        #     c_plot = axes[int(u)].contourf(X, Y, contour_plot, levels=100, linewidths=0, cmap=sns.cubehelix_palette(light=1, as_cmap=True, reverse=True), vmin=-6, vmax=4)

        #     axes[int(u)].set_title('{}'.format(utility_names[int(u)]))
        #     axes[int(u)].set_xlabel('{}'.format(feature1))
        #     axes[int(u)].set_ylabel('{}'.format(feature2))

        #     if int(u)==1:
        #         cbar = fig.colorbar(c_plot, ax = axes[int(u)])
        #         cbar.ax.set_ylabel('Utility value')

        for u in weights_2d.Utility.unique():
            weights_2d_util = weights_2d[weights_2d.Utility==u]
            contour_plot1 = self.function_2d_v2(weights_2d_util[weights_2d_util.Feature==name1], x_vect, y_vect)
            contour_plot2 = self.function_2d_v2(weights_2d_util[weights_2d_util.Feature==name2], y_vect, x_vect)

            contour_plot = contour_plot1 + contour_plot2.T

            if np.sum(contour_plot) == 0:
                continue

            if (feature1 in weights_ordered[str(u)].keys()) and (feature2 in weights_ordered[str(u)].keys()):
                x_1, feature1_alone = self.non_lin_function(weights_ordered[str(u)][feature1], min1, max1, num_points)
                feature1_alone = [f - feature1_alone[0] for f in feature1_alone]
                #feature1_grid = np.repeat(feature1_alone, num_points).reshape((num_points, num_points))
                #contour_plot += feature1_grid

                x_2, feature2_alone = self.non_lin_function(weights_ordered[str(u)][feature2], min2, max2, num_points)
                feature2_alone = [f - feature2_alone[0] for f in feature2_alone]
                #feature2_grid = np.repeat(feature2_alone, num_points).reshape((num_points, num_points)).T
                #contour_plot += feature2_grid

            contour_plot -= contour_plot.max()

            colors = ['#F5E5E2', '#DF7057', '#A31D04']
            # Set your custom color palette
            customPalette = sns.set_palette(sns.color_palette(colors, as_cmap=True))

            if np.sum(contour_plot) != 0:
                X, Y = np.meshgrid(x_vect, y_vect)
                res = 100
                #fig, axes = plt.subplots(figsize=(3.49,3), layout='constrained', dpi=1000)
                g = sns.JointGrid(xlim=(0, 2), height=4.5, space = 0.15)
                g.figure.set_dpi(1000)
                cplot = g.ax_joint.contourf(X, Y, contour_plot.T, levels=res, linewidths=0, cmap=customPalette, vmin=-12, vmax=0)
                g.ax_marg_x.plot(x_1, feature1_alone, color = 'k')
                g.ax_marg_x.tick_params(labelleft=True, labelsize=3, length=0.1)
                g.ax_marg_x.grid(True, axis='y', linewidth= 0.2)

                norm = Normalize(vmin=-12, vmax=0)
                mappable = ScalarMappable(norm=norm, cmap=customPalette)

                # Fill the marginal plots
                #g.ax_marg_x.fill_between(x_1, x_density, color='white')
                for i in range(len(x_1) - 1):
                    g.ax_marg_x.fill_between(x_1[i:i+2], feature1_alone[i:i+2], color=mappable.to_rgba((feature1_alone[i] + feature1_alone[i+1]) / 2))

                g.ax_marg_y.plot(feature2_alone, x_2, color = 'k')
                g.ax_marg_y.tick_params(labelbottom=True, labelrotation=-90, labelsize=4, length=0.1)
                g.ax_marg_y.grid(True, axis='x', linewidth= 0.2)

                for i in range(len(x_2) - 1):
                    g.ax_marg_y.fill_betweenx(x_2[i:i+2], feature2_alone[i:i+2], color=mappable.to_rgba((feature2_alone[i] + feature2_alone[i+1]) / 2))
                # 
                # fig, axes = plt.subplots(figsize=(3.49,3), layout='constrained', dpi=1000)

                #fig.suptitle('Impact of {} and {} on the utility function'.format(feature1, feature2))

                #res = 100

                #c_plot = axes.contourf(X, Y, contour_plot.T, levels=res, linewidths=0, cmap=customPalette, vmin=-12, vmax=0)

                plt.subplots_adjust(right=0.85)  # Adjust this value based on your layout needs

                # Create a new axes for the colorbar
                cbar_ax = g.figure.add_axes([0.87, 0.05, 0.02, 0.8])

                # Add colorbar with label 'Utility'
                cbar = plt.colorbar(cplot, cax=cbar_ax, orientation='vertical', ticks=[-5, -4, -3, -2, -1, 0])
                cbar.set_label('Utility')


                #axes.set_title(f'{utility_names[int(u)]}')
                g.set_axis_labels(xlabel=f'{feature1} [h]', ylabel=f'{feature2}')

                #cbar = g.figure.colorbar(c_plot, ax = axes, ticks=[-10, -8, -6, -4, -2, 0])
                #cbar.ax.set_ylabel('Utility')

                #plt.tight_layout()

                if save_figure:
                    plt.savefig('../Figures/FI RUMBoost/age_travel_time_detailed_{}_big.png'.format(utility_names[int(u)]))

                plt.show()
    
    def plot_parameters(self, params, X, utility_names, Betas = None , withPointDist = False, model_unconstrained = None, 
                        params_unc = None, with_pw = False, save_figure=False, asc_normalised = True, data_sep=False,
                        only_tt = False, only_1d = False, with_asc = False, with_cat=True, with_fit = False, fit_all=True, technique = 'weighted_data', sm_tt_cost=False,
                        save_file=''):
        """
        Plot the non linear impact of parameters on the utility function. When specified, unconstrained parameters
        and parameters from a RUM model can be added to the plot.

        Parameters
        ----------
        params : dict
            Dictionary containing parameters used to train the RUM booster.
        X : pandas dataframe
            Features used to train the model, in a pandas dataframe.
        utility_name : dict
            Dictionary mapping utilities to their names.
        Betas : list, optional (default = None)
            List of beta parameters value from a RUM. They should be listed in the same order as 
            in the RUMBooster model.
        withPointDist: Bool, optional (default = False)
            If True, the distribution of the training samples for the corresponding features will be plot 
            on the x axis
        model_unconstrained: LightGBM model, optional (default = None)
            The unconstrained model. Must be trained and compatible with dump_model().
        params_unc: dict, optional (default = None)
            Dictionary containing parameters used to train the unconstrained model
        with_pw: bool, optional (default = False)
            If the piece-wise function should be included in the graph
        save_figrue: bool
            if True, save the plot as a png file
        asc_normalised: bool
            if True, scale down utilities to be zero at the y axis
        only_tt: bool
            if true, plot only travel time and distance
        with_asc:
            if True, add full ASCs to the utility plot
        """
        #getting learning rate
        # if params['learning_rate'] is not None:
        #     lr = float(params['learning_rate'])
        # else:
        #     lr = 0.3
        
        # if model_unconstrained is not None:
        #     if params_unc['learning_rate'] is not None:
        #         lr_unc = float(params_unc['learning_rate'])
        #     else:
        #         lr_unc = 0.3
        
        #get and prepare weights
        weights_arranged = self.weights_to_plot_v2()

        if with_fit | data_sep:
            func_for_fit = func_wrapper()

        if with_pw:
            pw_func = self.plot_util_pw(X)
        
        if model_unconstrained is not None:
            weights_arranged_unc = self.weights_to_plot_v2(model=model_unconstrained)

        if with_asc:
            ASCs = self.get_asc(weights_arranged)

        if (not fit_all) and (with_fit):
            best_funcs = self.find_feat_best_fit(X, technique = technique, func_for_fit=func_for_fit)

        
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

        if sm_tt_cost:
            #plot for travel time on one figure
            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            x_w, non_lin_func_rail = self.non_lin_function(weights_arranged['0']['TRAIN_TT'], 0, 600, 10000)
            if asc_normalised:
                non_lin_func_rail = [n - non_lin_func_rail[0] for n in non_lin_func_rail]
            if with_asc:
                non_lin_func_rail = [n + ASCs[0] for n in non_lin_func_rail]

            x_c, non_lin_func_SM = self.non_lin_function(weights_arranged['1']['SM_TT'], 0, 600, 10000)
            if asc_normalised:
                non_lin_func_SM = [n - non_lin_func_SM[0] for n in non_lin_func_SM]
            if with_asc:
                non_lin_func_SM = [n + ASCs[1] for n in non_lin_func_SM]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['2']['CAR_TT'], 0, 600, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            sns.lineplot(x=x_w/60, y=non_lin_func_rail, color='g', label='Rail')
            sns.lineplot(x=x_c/60, y=non_lin_func_SM, color='#6b8ba4', label='Swissmetro')
            sns.lineplot(x=x_d/60, y=non_lin_func_driving, color='orange', label='Driving')

            #plt.title('Influence of alternative travel time on the utility function', fontdict={'fontsize':  16})
            plt.xlabel('Travel time [h]')
            plt.ylabel('Utility')

            plt.tight_layout()

            if save_figure:
                plt.savefig('../Figures/RUMBoost/SwissMetro/travel_time.png')

            #plot for travel time on one figure
            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            x_w, non_lin_func_rail = self.non_lin_function(weights_arranged['0']['TRAIN_COST'], 0, 500, 10000)
            if asc_normalised:
                non_lin_func_rail = [n - non_lin_func_rail[0] for n in non_lin_func_rail]
            if with_asc:
                non_lin_func_rail = [n + ASCs[0] for n in non_lin_func_rail]

            x_c, non_lin_func_SM = self.non_lin_function(weights_arranged['1']['SM_COST'], 0, 500, 10000)
            if asc_normalised:
                non_lin_func_SM = [n - non_lin_func_SM[0] for n in non_lin_func_SM]
            if with_asc:
                non_lin_func_SM = [n + ASCs[1] for n in non_lin_func_SM]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['2']['CAR_CO'], 0, 500, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            sns.lineplot(x=x_w, y=non_lin_func_rail, color='g', label='Rail')
            sns.lineplot(x=x_c, y=non_lin_func_SM, color='#6b8ba4', label='Swissmetro')
            sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')

            #plt.title('Influence of alternative cost on the utility function', fontdict={'fontsize':  16})

            plt.xlabel('Cost [chf]')
            plt.ylabel('Utility')

            plt.tight_layout()

            if save_figure:
                plt.savefig('../Figures/RUMBoost/SwissMetro/cost.png')

        if not only_1d:
            #plot for travel time on one figure
            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            x_w, non_lin_func_walk = self.non_lin_function(weights_arranged['0']['dur_walking'], 0, 2.5, 10000)
            if asc_normalised:
                non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
            if with_asc:
                non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

            x_c, non_lin_func_cycle = self.non_lin_function(weights_arranged['1']['dur_cycling'], 0, 2.5, 10000)
            if asc_normalised:
                non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
            if with_asc:
                non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

            x_ptb, non_lin_func_pt_bus = self.non_lin_function(weights_arranged['2']['dur_pt_bus'], 0, 2.5, 10000)
            if asc_normalised:
                non_lin_func_pt_bus = [n - non_lin_func_pt_bus[0] for n in non_lin_func_pt_bus]
            if with_asc:
                non_lin_func_pt_bus = [n + ASCs[2] for n in non_lin_func_pt_bus]

            x_ptr, non_lin_func_pt_rail = self.non_lin_function(weights_arranged['2']['dur_pt_rail'], 0, 2.5, 10000)
            if asc_normalised:
                non_lin_func_pt_rail = [n - non_lin_func_pt_rail[0] for n in non_lin_func_pt_rail]
            if with_asc:
                non_lin_func_pt_rail = [n + ASCs[2] for n in non_lin_func_pt_rail]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['3']['dur_driving'], 0, 2.5, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
            sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
            sns.lineplot(x=x_ptb, y=non_lin_func_pt_bus, color='#02590f', label='PT Bus')
            sns.lineplot(x=x_ptr, y=non_lin_func_pt_rail, color='g', label='PT Rail')
            sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


            #plt.title('Influence of alternative travel time on the utility function', fontdict={'fontsize':  16})
            plt.xlabel('Travel time [h]')
            plt.ylabel('Utility')

            plt.tight_layout()

            if save_figure:
                plt.savefig('../Figures/RUMBoost/LPMC/travel_time.png')
            
            #plot for distance on one figure
            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            # x_w, non_lin_func_walk = self.non_lin_function(weights_arranged['0']['distance'], 0, 1.05*max(X['distance']), 10000)
            # if asc_normalised:
            #     non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
            # if with_asc:
            #     non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

            # x_c, non_lin_func_cycle = self.non_lin_function(weights_arranged['1']['distance'], 0, 1.05*max(X['distance']), 10000)
            # if asc_normalised:
            #     non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
            # if with_asc:
            #     non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

            x_pt, non_lin_func_pt = self.non_lin_function(weights_arranged['2']['cost_transit'], 0, 10, 10000)
            if asc_normalised:
                non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
            if with_asc:
                non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['3']['cost_driving_fuel'], 0, 10, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            # sns.lineplot(x=x_w, y=non_lin_func_walk, lw=2, color='#fab9a5', label='Walking')
            # sns.lineplot(x=x_c, y=non_lin_func_cycle, lw=2, color='#B65FCF', label='Cycling')
            sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
            sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


            #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
            plt.xlabel('Cost [£]')
            plt.ylabel('Utility')

            plt.tight_layout()
            
            if save_figure:
                plt.savefig('../Figures/RUMBoost/LPMC/cost.png')

            plt.show()

            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            x_w, non_lin_func_walk = self.non_lin_function(weights_arranged['0']['age'], 0, 100, 10000)
            if asc_normalised:
                non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
            if with_asc:
                non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

            x_c, non_lin_func_cycle = self.non_lin_function(weights_arranged['1']['age'], 0, 100, 10000)
            if asc_normalised:
                non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
            if with_asc:
                non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

            x_pt, non_lin_func_pt = self.non_lin_function(weights_arranged['2']['age'], 0, 100, 10000)
            if asc_normalised:
                non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
            if with_asc:
                non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['3']['age'], 0, 100, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
            sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
            sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
            sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


            #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
            plt.xlabel('Age')
            plt.ylabel('Utility')

            plt.tight_layout()

            if save_figure:
                plt.savefig('../Figures/RUMBoost/LPMC/age.png')

            plt.show()

            plt.figure(figsize=(3.49, 3.49), dpi=1000)
            x_w, non_lin_func_walk = self.non_lin_function(weights_arranged['0']['start_time_linear'], 0, 24, 10000)
            if asc_normalised:
                non_lin_func_walk = [n - non_lin_func_walk[0] for n in non_lin_func_walk]
            if with_asc:
                non_lin_func_walk = [n + ASCs[0] for n in non_lin_func_walk]

            x_c, non_lin_func_cycle = self.non_lin_function(weights_arranged['1']['start_time_linear'], 0, 24, 10000)
            if asc_normalised:
                non_lin_func_cycle = [n - non_lin_func_cycle[0] for n in non_lin_func_cycle]
            if with_asc:
                non_lin_func_cycle = [n + ASCs[1] for n in non_lin_func_cycle]

            x_pt, non_lin_func_pt = self.non_lin_function(weights_arranged['2']['start_time_linear'], 0, 24, 10000)
            if asc_normalised:
                non_lin_func_pt = [n - non_lin_func_pt[0] for n in non_lin_func_pt]
            if with_asc:
                non_lin_func_pt = [n + ASCs[2] for n in non_lin_func_pt]

            x_d, non_lin_func_driving = self.non_lin_function(weights_arranged['3']['start_time_linear'], 0, 24, 10000)
            if asc_normalised:
                non_lin_func_driving = [n - non_lin_func_driving[0] for n in non_lin_func_driving]
            if with_asc:
                non_lin_func_driving = [n + ASCs[3] for n in non_lin_func_driving]

            sns.lineplot(x=x_w, y=non_lin_func_walk, color='b', label='Walking')
            sns.lineplot(x=x_c, y=non_lin_func_cycle, color='r', label='Cycling')
            sns.lineplot(x=x_pt, y=non_lin_func_pt, color='g', label='PT')
            sns.lineplot(x=x_d, y=non_lin_func_driving, color='orange', label='Driving')


            #plt.title('Influence of straight line distance on the utility function', fontdict={'fontsize':  16})
            plt.xlabel('Departure time')
            plt.ylabel('Utility')

            plt.tight_layout()

            if save_figure:
                plt.savefig('../Figures/RUMBoost/LPMC/departure_time.png')

            plt.show()

        #for all features parameters
        if not only_tt:
            for u in weights_arranged:
                for i, f in enumerate(weights_arranged[u]):
                    
                    if not with_cat:
                        is_cat = self.rum_structure[int(u)]['columns'].index(f) in self.rum_structure[int(u)]['categorical_feature']
                        if is_cat:
                            continue
                    else:
                        is_cat = False

                    #create nonlinear plot
                    x, non_lin_func = self.non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)

                    if asc_normalised:
                        val_0 = non_lin_func[0]
                        non_lin_func = [n - val_0 for n in non_lin_func]

                    if with_asc:
                        non_lin_func = [n + ASCs[int(u)] for n in non_lin_func]
                    
                    #non_lin_func_with_lr = [h/lr for h in non_lin_func]
                    
                    #plot parameters
                    plt.figure(figsize=(3.49, 2.09), dpi=1000)
                    #plt.title('Influence of {} on the predictive function ({} utility)'.format(f, utility_names[u]), fontdict={'fontsize':  16})
                    plt.ylabel('{} utility'.format(utility_names[u]))

                                        
                    if 'dur' in f:
                        plt.xlabel('{} [h]'.format(f))
                    elif 'TIME' in f:
                        plt.xlabel('{} [min]'.format(f))
                    elif 'cost' in f:
                        plt.xlabel('{} [£]'.format(f))
                    elif 'distance' in f:
                        plt.xlabel('{} [m]'.format(f))
                    elif 'CO' in f:
                        plt.xlabel('{} [chf]'.format(f))
                    else:
                        plt.xlabel('{}'.format(f))


                    if not is_cat:
                        sns.lineplot(x=x, y=non_lin_func, color='k', label='RUMBoost')

                    if (with_fit) and (not is_cat):
                        if fit_all:
                            opt_params, _ = self._fit_func(X[f], weights_arranged[u][f], technique = technique, func_for_fit=func_for_fit)
                            for func, p in opt_params.items():
                                y_smooth = func_for_fit[func](x, *p)
                                if asc_normalised:
                                    y_smooth += -val_0
                                if with_asc:
                                    y_smooth += ASCs[int(u)]
                                sns.lineplot(x=x, y=y_smooth, label=func)
                        else:
                            y_smooth = func_for_fit[best_funcs[u][f]['best_func']](x, *best_funcs[u][f]['best_params'])
                            if asc_normalised:
                                y_smooth += -val_0
                            if with_asc:
                                y_smooth += ASCs[int(u)]
                            sns.lineplot(x=x, y=y_smooth, label=best_funcs[u][f]['best_func'])

                    if (not is_cat) and (data_sep):
                        funcs, x_range = self.fit_sev_functions(X[f], weights_arranged[u][f])
                        for i, func in enumerate(funcs):
                            y_smooth = func_for_fit[list(func)[0]](x_range[i], *func[list(func)[0]])
                            if asc_normalised:
                                y_smooth += -val_0
                            if with_asc:
                                y_smooth += ASCs[int(u)]
                            sns.lineplot(x=x_range[i], y=y_smooth, label=list(func)[0])


                    #plot unconstrained model parameters
                    if model_unconstrained is not None:
                        _, non_lin_func_unc = self.non_lin_function(weights_arranged_unc[u][f], 0, 1.05*max(X[f]), 10000)
                        #non_lin_func_with_lr_unc =  [h_unc/lr_unc for h_unc in non_lin_func_unc]
                        sns.lineplot(x=x, y=non_lin_func_unc)

                    # if (with_pw) & (self.rum_structure[int(u)]['columns'].index(f)  not in self.rum_structure[int(u)]['categorical_feature']):
                    #     #pw_func_with_lr = [pw/lr for pw in pw_func[int(u)][i]]
                    #     sns.lineplot(x=x, y=pw_func[int(u)][i], lw=2)
                    
                    #plot RUM parameters
                    if Betas is not None:
                        sns.lineplot(x=x, y=Betas[i]*x)
                    
                    #plot data distribution
                    if withPointDist:
                        sns.scatterplot(x=x, y=0*x, s=100, alpha=0.1)
                    
                    plt.xlim([0-0.05*np.max(X[f]), np.max(X[f])*1.05])
                    plt.ylim([np.min(non_lin_func) - 0.05*(np.max(non_lin_func)-np.min(non_lin_func)), np.max(non_lin_func) + 0.05*(np.max(non_lin_func)-np.min(non_lin_func))])

                    plt.tight_layout()
                    #legend
                    # if Betas is not None:
                    #     if model_unconstrained is not None:
                    #         if withPointDist:
                    #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM', 'Data'])
                    #         else:
                    #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'With RUM'])
                    #     else:
                    #         if withPointDist:
                    #             plt.legend(labels = ['With GBM constrained', 'With RUM', 'Data'])
                    #         else:
                    #             plt.legend(labels = ['With GBM constrained', 'With RUM'])
                    # else:
                    #     if model_unconstrained is not None:
                    #         if withPointDist:
                    #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained', 'Data'])
                    #         else:
                    #             plt.legend(labels = ['With GBM constrained', 'With GBM unconstrained'])
                    #     else:
                    #         if withPointDist:
                    #             plt.legend(labels = ['With GBM constrained', 'Data'])
                    #         elif with_pw:
                    #             plt.legend(labels = ['With GBM constrained', 'With piece-wise linear function'])
                    #         else:
                    #             plt.legend(labels = ['RUMBooster'])
                        
                    if save_figure:
                        if with_fit:
                            plt.savefig('../Figures/{}{} utility, {} feature {} technique.png'.format(utility_names[u], f, technique))
                        else:
                            plt.savefig('../Figures/{}{} utility, {} feature.png'.format(save_file, utility_names[u], f))

                    #plt.show()

    def plot_market_segm(self, X, asc_normalised = True):

        utility_names = ['Walking', 'Cycling', 'Public Transport', 'Driving']
        sns.set_theme()

        weights_arranged = self.weights_to_plot_v2(market_segm=True)
        label = {0:'Weekdays',1:'Weekends'}
        color = ['r', 'b']

        for u in weights_arranged:
            plt.figure(figsize=(10, 6))

            for i, f in enumerate(weights_arranged[u]):

                #create nonlinear plot
                x, non_lin_func = self.non_lin_function(weights_arranged[u][f], 0, 1.05*max(X[f]), 10000)

                if asc_normalised:
                    val_0 = non_lin_func[0]
                    non_lin_func = [n - val_0 for n in non_lin_func]
                
                sns.lineplot(x=x, y=non_lin_func, lw=2, color=color[i], label=label[i])

                #plot parameters
                
                # plt.xlim([0-0.05*np.max(X[f]), np.max(X[f])*1.05])
                # plt.ylim([np.min(non_lin_func) - 0.05*(np.max(non_lin_func)-np.min(non_lin_func)), np.max(non_lin_func) + 0.05*(np.max(non_lin_func)-np.min(non_lin_func))])
            plt.title('Impact of travel time in weekdays and weekends on {} utility'.format(utility_names[u]), fontdict={'fontsize':  16})
            plt.ylabel('{} utility'.format(utility_names[u]))
            plt.xlabel('Travel time [h]')
            plt.show()
 #               plt.savefig('Figures/rumbooster_vfinal_lr3e-1 {} utility, {} feature.png'.format(utility_names[u], f))
                
    
    def plot_util(self, data_train, points=10000):
        '''
        plot the raw utility functions of all features
        '''
        sns.set_theme()
        for j, struct in enumerate(self.rum_structure):
            booster = self.boosters[j]
            for i, f in enumerate(struct['columns']):
                xin = np.zeros(shape = (points, len(struct['columns'])))
                xin[:, i] = np.linspace(0,1.05*max(data_train[f]),points)
                
                ypred = booster.predict(xin)
                plt.figure()
                plt.plot(np.linspace(0,1.05*max(data_train[f]),points), ypred)
                plt.title(f)

    def plot_util_pw(self, data_train, points = 10000):
        '''
        plot the piece-wise utility function
        '''
        features = data_train.columns
        data_to_transform = {}
        for f in features:
            xi = np.linspace(0, 1.05*max(data_train[f]), points)
            data_to_transform[f] = xi

        data_to_transform = pd.DataFrame(data_to_transform)

        pw_func = self._stairs_to_pw(data_train, data_to_transform, util_for_plot = True)

        return pw_func
    
    def _data_leaf_value(self, data, weights_feature, technique):

        if technique == 'mid_point':
            mid_points = np.array(self._get_mid_pos(data, weights_feature['Splitting points']))
            return mid_points, weights_feature['Histogram values']

        data_ordered = data.copy().sort_values()
        data_values = [weights_feature['Histogram values'][0]]*sum(data_ordered < weights_feature['Splitting points'][0])
        if technique == 'mid_point_weighted':
            mid_points = self._get_mid_pos(data, weights_feature['Splitting points'])
            mid_points_weighted = [mid_points[0]]*sum(data_ordered < weights_feature['Splitting points'][0])

        for i, (s_i, s_ii) in enumerate(zip(weights_feature['Splitting points'][:-1], weights_feature['Splitting points'][1:])):
            data_values += [weights_feature['Histogram values'][i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))
            if technique == 'mid_point_weighted':
                mid_points_weighted += [mid_points[i+1]]*sum((data_ordered < s_ii) & (data_ordered > s_i))

        data_values += [weights_feature['Histogram values'][-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
        if technique == 'mid_point_weighted':
            mid_points_weighted += [mid_points[-1]]*sum(data_ordered > weights_feature['Splitting points'][-1])
            return np.array(mid_points_weighted), data_values

        return data_ordered, data_values
        
    def _fit_func(self, data, weight, technique='weighted_data', split_data=False):

        data_ordered, data_values = self._data_leaf_value(data, weight, technique)

        func_for_fit = func_wrapper()

        best_fit = np.inf
        func_fitted = {}
        fit_score = {}

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
    
    def find_feat_best_fit(self, data, technique='weighted_data'):
        
        weights = self.weights_to_plot_v2()
        best_fit = {}
        for u in weights:
            best_fit[u] = {}
            for f in weights[u]:
                if self.rum_structure[int(u)]['columns'].index(f) in self.rum_structure[int(u)]['categorical_feature']:
                    continue #add a function that is basic tree
                func_fitted, fit_score = self._fit_func(data[f], weights[u][f], technique=technique, func_for_fit=func_for_fit)
                best_fit[u][f] = {'best_func': min(fit_score, key = fit_score.get), 'best_params': func_fitted[min(fit_score, key = fit_score.get)], 'best_score': min(fit_score)}

        return best_fit

    def data_to_fit(self, data, weights_feat, weight_w_data = False):
        split_points = weights_feat['Splitting points']
        leaves_values = weights_feat['Histogram values']
        mid_points = self._get_mid_pos(data, weights_feat['Splitting points'], end ='split point')

        split_range = np.max(split_points) - np.min(split_points)
        leaves_range = np.max(leaves_values) - np.min(leaves_values)

        new_func_idx = []
        start = 0

        for i, (s_1, s_2) in enumerate(zip(mid_points[:-1], mid_points[1:])):
            if (s_2-s_1) > 0.2*split_range:
                stop = i
                new_func_idx.append((start, stop))
                start = i+1
            elif np.abs(leaves_values[i] - leaves_values[i+1]) > 0.2*leaves_range:
                stop = i
                new_func_idx.append((start, stop))
                start = i+1

        return [{'Mid points':mid_points[i[0]:i[1]+1], 'Histogram values':leaves_values[i[0]:i[1]+1]} for i in new_func_idx]
    
    def fit_sev_functions(self, data, weight, technique='mid_point'):

        func_for_fit = func_wrapper()

        data_list = self.data_to_fit(data, weight, technique)
        best_funcs = []
        x_range = []
        for d in data_list:
            best_fit = np.inf
            for n, f in func_for_fit.items():
                try:
                    param_opt, _, info, _, _ = curve_fit(f, d['Mid points'], d['Histogram values'], full_output=True)
                except:
                    continue
                if np.sum(info['fvec']**2) < best_fit:
                    best_fit = np.sum(info['fvec']**2)
                    best_func = n
                    best_params = param_opt
                print('Fitting residuals for the function {} is: {}'.format(n, np.sum(info['fvec']**2)))

            if best_fit == np.inf:
                continue
            print('Best fit ({}) for function {}'.format(best_fit, best_func))
            best_funcs.append({best_func:best_params})
            x_range.append(np.linspace(np.min(d['Mid points']), np.max(d['Mid points']), 10000))

        return best_funcs, x_range

    def model_from_string(self, model_str: str):
        """Load RUMBooster from a string.

        Parameters
        ----------
        model_str : str
            Model will be loaded from this string.

        Returns
        -------
        self : RUMBooster
            Loaded RUMBooster object.
        """
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> str:
        """Save RUMBooster to JSON string.

        Parameters
        ----------
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        str_repr : str
            JSON string representation of RUMBooster.
        """
        return json.dumps(self._to_dict(num_iteration, start_iteration, importance_type))

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = 'split'
    ) -> "RUMBooster":
        """Save RUMBooster to a file as JSON text.

        Parameters
        ----------
        filename : str or pathlib.Path
            Filename to save RUMBooster.
        num_iteration : int or None, optional (default=None)
            Index of the iteration that should be saved.
            If None, if the best iteration exists, it is saved; otherwise, all iterations are saved.
            If <= 0, all iterations are saved.
        start_iteration : int, optional (default=0)
            Start index of the iteration that should be saved.
        importance_type : str, optional (default="split")
            What type of feature importance should be saved.
            If "split", result contains numbers of times the feature is used in a model.
            If "gain", result contains total gains of splits which use the feature.

        Returns
        -------
        self : RUMBooster
            Returns self.
        """
        with open(filename, "w") as file:
            json.dump(self._to_dict(num_iteration, start_iteration, importance_type), file)

        return self

def rum_train(
    params: Dict[str, Any],
    train_set: Dataset,
    rum_structure: List[Dict[str, Any]],
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Optional[Union[_LGBM_CustomMetricFunction, List[_LGBM_CustomMetricFunction]]] = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    feature_name: Union[List[str], str] = 'auto',
    categorical_feature: Union[List[str], List[int], str] = 'auto',
    keep_training_booster: bool = False,
    callbacks: Optional[List[Callable]] = None,
    pw_utility: bool = False,
    nests: dict = None,
    mu: list = None,
    optimize_mu: bool = False,
    params_rde: dict = None 
) -> RUMBooster:
    """Perform the RUM training with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for training. Values passed through ``params`` take precedence over those
        supplied via arguments.
    train_set : Dataset
        Data to be trained on.
    rum_structure : dict
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
        'cols': list of columns included in that class
        'monotone_constraints': list of monotonic constraints on parameters
        'interaction_constraints': list of interaction constraints on features
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    valid_sets : list of Dataset, or None, optional (default=None)
        List of data to be evaluated on during training.
    valid_names : list of str, or None, optional (default=None)
        Names of ``valid_sets``.
    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, eval_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : numpy 1-D array or numpy 2-D array (for multi-class task)
                The predicted values.
                For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes].
                If custom objective function is used, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            eval_data : Dataset
                A ``Dataset`` to evaluate.
            eval_name : str
                The name of evaluation function (without whitespaces).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        To ignore the default metric corresponding to the used objective,
        set the ``metric`` parameter to the string ``"None"`` in ``params``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features will be cast to int32 and thus should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
        Floating point numbers in categorical features will be rounded towards 0.
    keep_training_booster : bool, optional (default=False)
        Whether the returned Booster will be used to keep training.
        If False, the returned value will be converted into _InnerPredictor before returning.
        This means you won't be able to use ``eval``, ``eval_train`` or ``eval_valid`` methods of the returned Booster.
        When your model is very large and cause the memory error,
        you can try to set this param to ``True`` to avoid the model conversion performed during the internal call of ``model_to_string``.
        You can still use _InnerPredictor as ``init_model`` for future continue training.
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    pw_utility: bool, optional (default=False)
        If true, compute continuous feature utility in a piece-wise linear way.

    Note
    ----
    A custom objective function can be provided for the ``objective`` parameter.
    It should accept two parameters: preds, train_data and return (grad, hess).

        preds : numpy 1-D array or numpy 2-D array (for multi-class task)
            The predicted values.
            Predicted values are returned before any transformation,
            e.g. they are raw margin instead of probability of positive class for binary task.
        train_data : Dataset
            The training dataset.
        grad : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the first order derivative (gradient) of the loss
            with respect to the elements of preds for each sample point.
        hess : numpy 1-D array or numpy 2-D array (for multi-class task)
            The value of the second order derivative (Hessian) of the loss
            with respect to the elements of preds for each sample point.

    For multi-class task, preds are numpy 2-D array of shape = [n_samples, n_classes],
    and grad and hess should be returned in the same format.

    Returns
    -------
    rum_booster : RUMBooster
        The trained RUMBooster model.
    """
    for alias in _ConfigAliases.get("verbosity"): 
        if alias in params:
            verbosity = params[alias]
    # create predictor first
    params = copy.deepcopy(params)
    params = _choose_param_value(
        main_param_name='objective',
        params=params,
        default_value=None
    )
    fobj: Optional[_LGBM_CustomObjectiveFunction] = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            num_boost_round = params.pop(alias)
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
    params["num_iterations"] = num_boost_round
    # setting early stopping via global params should be possible
    params = _choose_param_value(
        main_param_name="early_stopping_round",
        params=params,
        default_value=None
    )
    if params["early_stopping_round"] is None:
        params["early_stopping_round"] = 10000
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    predictor: Optional[_InnerPredictor] = None
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    init_iteration = predictor.num_total_iteration if predictor is not None else 0
    # check dataset
    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")


    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    # process callbacks
    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks_set = set(callbacks)

    #if "early_stopping_round" in params:
    #    callbacks_set.add(
    #        callback.early_stopping(
    #            stopping_rounds=params["early_stopping_round"],
    #            first_metric_only=first_metric_only,
    #            verbose=_choose_param_value(
    #                main_param_name="verbosity",
    #                params=params,
    #                default_value=1
    #            ).pop("verbosity") > 0
    #        )
    #    )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter('order'))

    #construct boosters
    rum_booster = RUMBooster()

    
    if 'num_classes' not in params:
        raise ValueError("Specify the number of classes in the dictionary of parameters with the key num_classes")
    if len(rum_structure) == 2 * params['num_classes']:
        rum_booster.random_effects = True
    elif len(rum_structure) == params['num_classes']:
        rum_booster.random_effects = False
    else:
        raise ValueError('The length of rum_structure must be equal to the number of classes or twice the number of class (for random effects)')
    
    reduced_valid_sets, \
    name_valid_sets, \
    is_valid_contain_train, \
    train_data_name = rum_booster._preprocess_valids(train_set, params, valid_sets) #prepare validation sets
    rum_booster.rum_structure = rum_structure #saving utility structure
    rum_booster.num_classes = params.pop('num_classes')
    rum_booster._preprocess_params(params, params_rde = params_rde) #preparing J set of parameters
    rum_booster._preprocess_data(train_set, reduced_valid_sets, return_data=True) #preparing J datasets with relevant features
    rum_booster._construct_boosters(train_data_name, is_valid_contain_train, name_valid_sets) #building boosters with corresponding params and dataset

    if nests is not None:
        rum_booster.mu = mu
        rum_booster.nests = nests
        f_obj = rum_booster.f_obj_nest
        rum_booster.labels_nest = np.array([nests[l] for l in rum_booster.labels])
        rum_booster._preds, rum_booster.preds_i_m, rum_booster.preds_m = rum_booster._inner_predict(nests=True)

        bounds = [(1, 10) if m != 1 else (1, 1) for m in mu]
    else:
        f_obj = rum_booster.f_obj
        rum_booster._preds = rum_booster._inner_predict()

    #start training
    for i in range(init_iteration, init_iteration + num_boost_round):
        #updating all binary boosters of the rum_booster

        # num_processes = rum_booster.num_classes

        # pool = Pool(processes=num_processes)

        # arguments = [(i, j, booster, rum_booster)
        #             for j, booster in enumerate(rum_booster.boosters)]

        # pool.starmap(update_booster, arguments)

        # pool.close()
        # pool.join()

        for j, booster in enumerate(rum_booster.boosters):
            # for cb in callbacks_before_iter:
            #     cb(callback.CallbackEnv(model=booster,
            #                             params=rum_booster.params[j],
            #                             iteration=i,
            #                             begin_iteration=init_iteration,
            #                             end_iteration=init_iteration + num_boost_round,
            #                             evaluation_result_list=None))       
    
            #update booster with custom binary objective function, and relevant features
            if rum_booster.random_effects:
                rum_booster._current_j = int(j/2)
            else:
                rum_booster._current_j = j
            booster.update(train_set=rum_booster.train_set[j], fobj=f_obj)
            
            # check evaluation result. (from lightGBM initial code, check on all J binary boosters)
            # evaluation_result_list = []
            # if valid_sets is not None:
            #     if is_valid_contain_train:
            #         evaluation_result_list.extend(booster.eval_train(feval))
            #     evaluation_result_list.extend(booster.eval_valid(feval))
            # try:
            #     for cb in callbacks_after_iter:
            #         cb(callback.CallbackEnv(model=booster,
            #                                 params=rum_booster.params[j],
            #                                 iteration=i,
            #                                 begin_iteration=init_iteration,
            #                                 end_iteration=init_iteration + num_boost_round,
            #                                 evaluation_result_list=evaluation_result_list))
            # except callback.EarlyStopException as earlyStopException:
            #     booster.best_iteration = earlyStopException.best_iteration + 1
            #     evaluation_result_list = earlyStopException.best_score

        #make predictions after boosting round to compute new cross entropy and for next iteration grad and hess
        if nests is not None:
            rum_booster._preds, rum_booster.preds_i_m, rum_booster.preds_m = rum_booster._inner_predict(nests=True)
            if optimize_mu and i % 100 == 0:
                mu_opt = minimize(rum_booster.optimize_mu, rum_booster.mu, args=(train_set.get_label().astype(int)), method='TNC', bounds=bounds)
                rum_booster.mu = mu_opt.x
                print(f'New mu: {mu_opt.x}')
        else:
            rum_booster._preds = rum_booster._inner_predict()
        #compute cross validation on training or validation test
        if valid_sets is not None:
            if is_valid_contain_train:
                cross_entropy = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
            else:
                for k, _ in enumerate(valid_sets):
                    if nests is not None:
                        preds_valid, _, _ = rum_booster._inner_predict(k+1, nests=True)
                    else:
                        preds_valid = rum_booster._inner_predict(k+1)
                    cross_entropy_train = rum_booster.cross_entropy(rum_booster._preds, train_set.get_label().astype(int))
                    cross_entropy = rum_booster.cross_entropy(preds_valid, valid_sets[0].get_label().astype(int))
        
            if cross_entropy < rum_booster.best_score:
                rum_booster.best_score = cross_entropy
                if is_valid_contain_train:
                    rum_booster.best_score_train = cross_entropy
                else:
                    rum_booster.best_score_train = cross_entropy_train
                rum_booster.best_iteration = i+1
        
            if (verbosity >= 1) and (i % 10 == 0):
                if is_valid_contain_train:
                    print('[{}] -- NCE value on train set: {}'.format(i + 1, cross_entropy))
                else:
                    print('[{}] -- NCE value on train set: {} \n     --  NCE value on test set: {}'.format(i + 1, cross_entropy_train, cross_entropy))
        
        #early stopping if early stopping criterion in all boosters
        if (params["early_stopping_round"] != 0) and (rum_booster.best_iteration + params["early_stopping_round"] < i + 1):
            if is_valid_contain_train:
                print('Early stopping at iteration {}, with a best score of {}'.format(rum_booster.best_iteration, rum_booster.best_score))
            else:
                print('Early stopping at iteration {}, with a best score on test set of {}, and on train set of {}'.format(rum_booster.best_iteration, rum_booster.best_score, rum_booster.best_score_train))
            break

    for booster in rum_booster.boosters:
        booster.best_score = collections.defaultdict(collections.OrderedDict)
        # for dataset_name, eval_name, score, _ in evaluation_result_list:
        #     booster.best_score[dataset_name][eval_name] = score
        if not keep_training_booster:
            booster.model_from_string(booster.model_to_string()).free_dataset()
    return rum_booster

# def update_booster(i, j, booster, rum_booster):
    # for cb in callbacks_before_iter:
    #     cb(callback.CallbackEnv(model=booster,
    #                             params=rum_booster.params[j],
    #                             iteration=i,
    #                             begin_iteration=init_iteration,
    #                             end_iteration=init_iteration + num_boost_round,
    #                             evaluation_result_list=None))       

    #update booster with custom binary objective function, and relevant features
    # rum_booster._current_j = j
    # booster.update(train_set=rum_booster.train_set[j], fobj=rum_booster.f_obj)
    
    # # check evaluation result. (from lightGBM initial code, check on all J binary boosters)
    # evaluation_result_list = []
    # if valid_sets is not None:
    #     if is_valid_contain_train:
    #         evaluation_result_list.extend(booster.eval_train(feval))
    #     evaluation_result_list.extend(booster.eval_valid(feval))
    # try:
    #     for cb in callbacks_after_iter:
    #         cb(callback.CallbackEnv(model=booster,
    #                                 params=rum_booster.params[j],
    #                                 iteration=i,
    #                                 begin_iteration=init_iteration,
    #                                 end_iteration=init_iteration + num_boost_round,
    #                                 evaluation_result_list=evaluation_result_list))
    # except callback.EarlyStopException as earlyStopException:
    #     booster.best_iteration = earlyStopException.best_iteration + 1
    #     evaluation_result_list = earlyStopException.best_score

class CVRUMBooster:
    """CVRUMBooster in LightGBM.

    Auxiliary data structure to hold and redirect all boosters of ``cv`` function.
    This class has the same methods as Booster class.
    All method calls are actually performed for underlying Boosters and then all returned results are returned in a list.

    Attributes
    ----------
    rum_boosters : list of RUMBooster
        The list of underlying fitted models.
    best_iteration : int
        The best iteration of fitted model.
    """

    def __init__(self):
        """Initialize the CVBooster.

        Generally, no need to instantiate manually.
        """
        self.rumboosters = []
        self.best_iteration = -1
        self.best_score = 100000

    def _append(self, rum_booster):
        """Add a booster to CVBooster."""
        self.rumboosters.append(rum_booster)

    def __getattr__(self, name):
        """Redirect methods call of CVBooster."""
        def handler_function(*args, **kwargs):
            """Call methods with each booster, and concatenate their results."""
            ret = []
            for rum_booster in self.rumboosters:
                for booster in rum_booster:
                    ret.append(getattr(booster, name)(*args, **kwargs))
                return ret
        return handler_function


def _make_n_folds(full_data, folds, nfold, params, seed, fpreproc=None, stratified=True,
                  shuffle=True, eval_train_metric=False, rum_structure=None):
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, '__iter__') and not hasattr(folds, 'split'):
            raise AttributeError("folds should be a generator or iterator of (train_idx, test_idx) tuples "
                                 "or scikit-learn splitter object with split method")
        if hasattr(folds, 'split'):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.array(group_info, dtype=np.int32, copy=False)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(params.get(obj_alias, "") in {"lambdarank", "rank_xendcg", "xendcg",
                                             "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
               for obj_alias in _ConfigAliases.get("objective")):
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for ranking cv')
            # ranking task, split according to groups
            group_info = np.array(full_data.get_group(), dtype=np.int32, copy=False)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError('scikit-learn is required for stratified cv')
            skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            kstep = int(num_data / nfold)
            test_id = [randidx[i: i + kstep] for i in range(0, num_data, kstep)]
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id)

    ret = CVRUMBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
        # run preprocessing on the data set if needed
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params.copy()
        #create RUMBoosters with corresponding training, validation, and parameters sets
        cvbooster = RUMBooster()
        if 'num_classes' in params:
            cvbooster.num_classes = tparam.pop('num_classes')
        cvbooster.rum_structure = rum_structure
        reduced_valid_sets, name_valid_sets, is_valid_contain_train, train_data_name = cvbooster._preprocess_valids(train_set, tparam, valid_set)
        cvbooster._preprocess_data(train_set, reduced_valid_sets)
        cvbooster._preprocess_params(tparam)
        cvbooster._construct_boosters(train_data_name, is_valid_contain_train,
                                      name_valid_sets=name_valid_sets)

        ret._append(cvbooster)
    return ret


def _agg_cv_result(raw_results, eval_train_metric=False):
    """Aggregate cross-validation results."""
    cvmap = collections.OrderedDict()
    metric_type = {}
    for one_result in raw_results:
        for one_line in one_result:
            if eval_train_metric:
                key = f"{one_line[0]} {one_line[1]}"
            else:
                key = one_line[1]
            metric_type[key] = one_line[3]
            cvmap.setdefault(key, [])
            cvmap[key].append(one_line[2])
    return [('cv_agg', k, np.mean(v), metric_type[k], np.std(v)) for k, v in cvmap.items()]


def rum_cv(params, train_set, num_boost_round=100,
       folds=None, nfold=5, stratified=True, shuffle=True,
       metrics=None, fobj=None, feval=None, init_model=None,
       feature_name='auto', categorical_feature='auto',
       early_stopping_rounds=None, fpreproc=None,
       verbose_eval=None, show_stdv=True, seed=0,
       callbacks=None, eval_train_metric=False,
       return_cvbooster=False, rum_structure=None):
    """Perform the cross-validation with given parameters.

    Parameters
    ----------
    params : dict
        Parameters for Booster.
    train_set : Dataset
        Data to be trained on.
    num_boost_round : int, optional (default=100)
        Number of boosting iterations.
    folds : generator or iterator of (train_idx, test_idx) tuples, scikit-learn splitter object or None, optional (default=None)
        If generator or iterator, it should yield the train and test indices for each fold.
        If object, it should be one of the scikit-learn splitter classes
        (https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
        and have ``split`` method.
        This argument has highest priority over other data split arguments.
    nfold : int, optional (default=5)
        Number of folds in CV.
    stratified : bool, optional (default=True)
        Whether to perform stratified sampling.
    shuffle : bool, optional (default=True)
        Whether to shuffle before splitting data.
    metrics : str, list of str, or None, optional (default=None)
        Evaluation metrics to be monitored while CV.
        If not None, the metric in ``params`` will be overridden.
    fobj : callable or None, optional (default=None)
        Customized objective function.
        Should accept two parameters: preds, train_data,
        and return (grad, hess).

            preds : list or numpy 1-D array
                The predicted values.
                Predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task.
            train_data : Dataset
                The training dataset.
            grad : list or numpy 1-D array
                The value of the first order derivative (gradient) of the loss
                with respect to the elements of preds for each sample point.
            hess : list or numpy 1-D array
                The value of the second order derivative (Hessian) of the loss
                with respect to the elements of preds for each sample point.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is score[j * num_data + i]
        and you should group grad and hess in this way as well.

    feval : callable, list of callable, or None, optional (default=None)
        Customized evaluation function.
        Each evaluation function should accept two parameters: preds, train_data,
        and return (eval_name, eval_result, is_higher_better) or list of such tuples.

            preds : list or numpy 1-D array
                The predicted values.
                If ``fobj`` is specified, predicted values are returned before any transformation,
                e.g. they are raw margin instead of probability of positive class for binary task in this case.
            train_data : Dataset
                The training dataset.
            eval_name : str
                The name of evaluation function (without whitespace).
            eval_result : float
                The eval result.
            is_higher_better : bool
                Is eval result higher better, e.g. AUC is ``is_higher_better``.

        For multi-class task, the preds is group by class_id first, then group by row_id.
        If you want to get i-th row preds in j-th class, the access way is preds[j * num_data + i].
        To ignore the default metric corresponding to the used objective,
        set ``metrics`` to the string ``"None"``.
    init_model : str, pathlib.Path, Booster or None, optional (default=None)
        Filename of LightGBM model or Booster instance used for continue training.
    feature_name : list of str, or 'auto', optional (default="auto")
        Feature names.
        If 'auto' and data is pandas DataFrame, data columns names are used.
    categorical_feature : list of str or int, or 'auto', optional (default="auto")
        Categorical features.
        If list of int, interpreted as indices.
        If list of str, interpreted as feature names (need to specify ``feature_name`` as well).
        If 'auto' and data is pandas DataFrame, pandas unordered categorical columns are used.
        All values in categorical features should be less than int32 max value (2147483647).
        Large values could be memory consuming. Consider using consecutive integers starting from zero.
        All negative values in categorical features will be treated as missing values.
        The output cannot be monotonically constrained with respect to a categorical feature.
    early_stopping_rounds : int or None, optional (default=None)
        Activates early stopping.
        CV score needs to improve at least every ``early_stopping_rounds`` round(s)
        to continue.
        Requires at least one metric. If there's more than one, will check all of them.
        To check only the first metric, set the ``first_metric_only`` parameter to ``True`` in ``params``.
        Last entry in evaluation history is the one from the best iteration.
    fpreproc : callable or None, optional (default=None)
        Preprocessing function that takes (dtrain, dtest, params)
        and returns transformed versions of those.
    verbose_eval : bool, int, or None, optional (default=None)
        Whether to display the progress.
        If True, progress will be displayed at every boosting stage.
        If int, progress will be displayed at every given ``verbose_eval`` boosting stage.
    show_stdv : bool, optional (default=True)
        Whether to display the standard deviation in progress.
        Results are not affected by this parameter, and always contain std.
    seed : int, optional (default=0)
        Seed used to generate the folds (passed to numpy.random.seed).
    callbacks : list of callable, or None, optional (default=None)
        List of callback functions that are applied at each iteration.
        See Callbacks in Python API for more information.
    eval_train_metric : bool, optional (default=False)
        Whether to display the train metric in progress.
        The score of the metric is calculated again after each training step, so there is some impact on performance.
    return_cvbooster : bool, optional (default=False)
        Whether to return Booster models trained on each fold through ``CVBooster``.
    rum_structure : dict, optional (default=None)
        List of dictionaries specifying the RUM structure. 
        The list must contain one dictionary for each class, which describes the 
        utility structure for that class. 
        Each dictionary has three allowed keys. 
            'cols': list of columns included in that class
            'monotone_constraints': list of monotonic constraints on parameters
            'interaction_constraints': list of interaction constraints on features

    Returns
    -------
    eval_hist : dict
        Evaluation history.
        The dictionary has the following format:
        {'metric1-mean': [values], 'metric1-stdv': [values],
        'metric2-mean': [values], 'metric2-stdv': [values],
        ...}.
        If ``return_cvbooster=True``, also returns trained boosters via ``cvbooster`` key.
    """
    for alias in _ConfigAliases.get("verbosity"): 
        if alias in params:
            verbosity = params.pop(alias)

    if not isinstance(train_set, Dataset):
        raise TypeError("Training only accepts Dataset object")

    params = copy.deepcopy(params)
    if fobj is not None:
        for obj_alias in _ConfigAliases.get("objective"):
            params.pop(obj_alias, None)
        params['objective'] = 'none'
    for alias in _ConfigAliases.get("num_iterations"):
        if alias in params:
            _log_warning(f"Found `{alias}` in params. Will use it instead of argument")
            num_boost_round = params.pop(alias)
    params["num_iterations"] = num_boost_round
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        _log_warning("'early_stopping_rounds' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'early_stopping()' callback via 'callbacks' argument instead.")
    for alias in _ConfigAliases.get("early_stopping_round"):
        if alias in params:
            early_stopping_rounds = params.pop(alias)
    params["early_stopping_round"] = early_stopping_rounds
    if params["early_stopping_round"] is None:
        params["early_stopping_round"] = 10000
    first_metric_only = params.get('first_metric_only', False)

    if num_boost_round <= 0:
        raise ValueError("num_boost_round should be greater than zero.")
    if isinstance(init_model, (str, Path)):
        predictor = _InnerPredictor(model_file=init_model, pred_parameter=params)
    elif isinstance(init_model, Booster):
        predictor = init_model._to_predictor(dict(init_model.params, **params))
    else:
        predictor = None

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params['metric'] = metrics

    train_set._update_params(params) \
             ._set_predictor(predictor) \
             .set_feature_name(feature_name) \
             .set_categorical_feature(categorical_feature)

    results = collections.defaultdict(list)
    cvfolds = _make_n_folds(train_set, folds=folds, nfold=nfold,
                            params=params, seed=seed, fpreproc=fpreproc,
                            stratified=stratified, shuffle=shuffle,
                            eval_train_metric=eval_train_metric, rum_structure=rum_structure)

    # setup callbacks
    if callbacks is None:
        callbacks = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault('order', i - len(callbacks))
        callbacks = set(callbacks)
    if early_stopping_rounds is not None and early_stopping_rounds > 0:
        callbacks.add(callback.early_stopping(early_stopping_rounds, first_metric_only, verbose=False))
    if verbose_eval is not None:
        _log_warning("'verbose_eval' argument is deprecated and will be removed in a future release of LightGBM. "
                     "Pass 'log_evaluation()' callback via 'callbacks' argument instead.")
    if verbose_eval is True:
        callbacks.add(callback.log_evaluation(show_stdv=show_stdv))
    elif isinstance(verbose_eval, int):
        callbacks.add(callback.log_evaluation(verbose_eval, show_stdv=show_stdv))

    callbacks_before_iter = {cb for cb in callbacks if getattr(cb, 'before_iteration', False)}
    callbacks_after_iter = callbacks - callbacks_before_iter
    callbacks_before_iter = sorted(callbacks_before_iter, key=attrgetter('order'))
    callbacks_after_iter = sorted(callbacks_after_iter, key=attrgetter('order'))

    for i in range(num_boost_round):
        cross_ent = []
        raw_results = []
        #train all rumboosters
        for rumbooster in cvfolds.rumboosters:
            rumbooster._preds = rumbooster._inner_predict()
            for j, booster in enumerate(rumbooster.boosters):
                for cb in callbacks_before_iter:
                    cb(callback.CallbackEnv(model=booster,
                                            params=rumbooster.params[j],
                                            iteration=i,
                                            begin_iteration=0,
                                            end_iteration=num_boost_round,
                                            evaluation_result_list=None))
                rumbooster._current_j = j
                booster.update(train_set = rumbooster.train_set[j], fobj=rumbooster.f_obj)

            valid_sets = rumbooster.valid_sets
            for valid_set in valid_sets:
                preds_valid = rumbooster._inner_predict(data = valid_set)
                raw_results.append(preds_valid)
                cross_ent.append(rumbooster.cross_entropy(preds_valid, valid_set[0].get_label().astype(int)))

        results[f'Cross entropy --- mean'].append(np.mean(cross_ent))
        results[f'Cross entropy --- stdv'].append(np.std(cross_ent))
        if (verbosity > 0) and i % 10 == 0:
            print('[{}] -- Cross entropy mean: {}, with std: {}'.format(i + 1, np.mean(cross_ent), np.std(cross_ent)))
        
        if np.mean(cross_ent) < cvfolds.best_score:
            cvfolds.best_score = np.mean(cross_ent)
            cvfolds.best_iteration = i + 1 

        if early_stopping_rounds is not None and cvfolds.best_iteration + early_stopping_rounds < i+1:
            print('Early stopping at iteration {} with a cross entropy best score of {}'.format(cvfolds.best_iteration,cvfolds.best_score))
            for k in results:
                results[k] = results[k][:cvfolds.best_iteration]
            break
        #res = _agg_cv_result(raw_results, eval_train_metric)
        #try:
        #    for cb in callbacks_after_iter:
        #        cb(callback.CallbackEnv(model=cvfolds,
        #                                params=params,
        #                                iteration=i,
        #                                begin_iteration=0,
        #                                end_iteration=num_boost_round,
        #                                evaluation_result_list=res))
        #except callback.EarlyStopException as earlyStopException:
        #    cvfolds.best_iteration = earlyStopException.best_iteration + 1
        #    for k in results:
        #        results[k] = results[k][:cvfolds.best_iteration]
        #    break

    if return_cvbooster:
        results['cvbooster'] = cvfolds

    return dict(results)