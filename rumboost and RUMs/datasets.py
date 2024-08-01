import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
# from rumboost.utils import stratified_group_k_fold
from utils import stratified_group_k_fold
import os


def load_preprocess_LPMC():
    '''
    Load and preprocess the LPMC dataset.

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    folds : zip(list, list)
        5 folds of indices grouped by household for CV.
    '''
    path_folder = os.getcwd()
    #source: https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice
    data_train = pd.read_csv('../Data/LPMC_train.csv')
    data_test = pd.read_csv('../Data/LPMC_test.csv')

    # data_train_2 = pd.read_csv('Data/LTDS_train.csv')
    # data_test_2 = pd.read_csv('Data/LTDS_test.csv')

    #distance in km
    data_train['distance'] = data_train['distance']/1000
    data_test['distance'] = data_test['distance']/1000
    
    # #cyclical start time
    # data_train['start_time_linear_cos'] = np.cos(data_train['start_time_linear']*(2.*np.pi/24))
    # data_train['start_time_linear_sin'] = np.sin(data_train['start_time_linear']*(2.*np.pi/24))
    # data_test['start_time_linear_cos'] = np.cos(data_test['start_time_linear']*(2.*np.pi/24))
    # data_test['start_time_linear_sin'] = np.sin(data_test['start_time_linear']*(2.*np.pi/24))

    # #cyclical travel month
    # data_train['travel_month_cos'] = np.cos(data_train_2['travel_month']*(2.*np.pi/12))
    # data_train['travel_month_sin'] = np.sin(data_train_2['travel_month']*(2.*np.pi/12))
    # data_test['travel_month_cos'] = np.cos(data_test_2['travel_month']*(2.*np.pi/12))
    # data_test['travel_month_sin'] = np.sin(data_test_2['travel_month']*(2.*np.pi/12))

    #for market segmentation
    # data_train['weekend'] = (data_train['day_of_week'] > 5).apply(int)  
    # data_test['weekend'] = (data_test['day_of_week'] > 5).apply(int)  

    #rename label
    label_name = {'travel_mode': 'choice'}
    dataset_train = data_train.rename(columns = label_name)
    dataset_test = data_test.rename(columns = label_name)

    #get all features
    target = 'choice'
    features = [f for f in dataset_test.columns if f != target]

    #get household ids
    hh_id = np.array(data_train['household_id'].values)

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open('../Data/strat_group_k_fold_london.pickle', "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(data_train[features], data_train['travel_mode'], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open('../Data/strat_group_k_fold_london.pickle', "wb"))

    folds = zip(train_idx, test_idx)

    return dataset_train, dataset_test, folds

def load_preprocess_SwissMetro(test_size: float = 0.3, random_state: int = 42, full_data=False):
    '''
    Load and preprocess the SwissMetro dataset.

    Parameters
    ----------
    test_size : float, optional (default = 0.3)
        The proportion of data used for test set.
    random_state : int, optional (default = 42)
        For reproducibility in the train-test split

    Returns
    -------
    dataset_train : pandas Dataframe
        The training dataset ready to use.
    dataset_test : pandas Dataframe
        The training dataset ready to use.
    '''
    df = pd.read_csv(r"../Data/swissmetro.dat", sep='\t')

    label_name = {'CHOICE': 'choice'}

    #remove irrelevant choices and purposes
    keep = ((df['CHOICE']!=0)*(df['CAR_AV']==1)) > 0
    df = df[keep]

    #apply cost to people without GA
    df.loc[:, 'TRAIN_COST'] = df['TRAIN_CO'] * (df['GA']==0)
    df.loc[:, 'SM_COST'] = df['SM_CO'] * (df['GA']==0)

    #rescale choice from 0 to 2
    df.loc[:,'CHOICE'] = df['CHOICE'] - 1

    #luggage dummies
    df.loc[:, 'SEV_LUGGAGES'] = (df['LUGGAGE']==3).astype(int)

    #origin
    df.loc[:, 'ORIG_ROM'] = df['ORIGIN'].apply(lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0)
    df.loc[:, 'ORIG_TIC'] = df['ORIGIN'].apply(lambda x: 1 if x in [21] else 0)

    #dest
    df.loc[:, 'DEST_ROM'] = df['DEST'].apply(lambda x: 1 if x in [10, 22, 23, 24, 25, 26] else 0)
    df.loc[:, 'DEST_TIC'] = df['DEST'].apply(lambda x: 1 if x in [21] else 0)

    #purpose
    df.loc[:, 'PURPOSE_1'] = (df['PURPOSE']==1).astype(int)
    df.loc[:, 'PURPOSE_2'] = (df['PURPOSE']==2).astype(int)
    df.loc[:, 'PURPOSE_3'] = (df['PURPOSE']==3).astype(int)
    df.loc[:, 'PURPOSE_4'] = (df['PURPOSE']==4).astype(int)
    df.loc[:, 'PURPOSE_5'] = (df['PURPOSE']==5).astype(int)
    df.loc[:, 'PURPOSE_6'] = (df['PURPOSE']==6).astype(int)
    df.loc[:, 'PURPOSE_7'] = (df['PURPOSE']==7).astype(int)
    df.loc[:, 'PURPOSE_8'] = (df['PURPOSE']==8).astype(int)

    #age
    df.loc[:, 'AGE_1'] = (df['AGE']==1).astype(int)
    df.loc[:, 'AGE_2'] = (df['AGE']==2).astype(int)
    df.loc[:, 'AGE_3'] = (df['AGE']==3).astype(int)
    df.loc[:, 'AGE_4'] = (df['AGE']==4).astype(int)

    #final dataset
    df_final = df[['ID', 'TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO', 'MALE', 'FIRST', 'PURPOSE_1', 'PURPOSE_2', 'PURPOSE_3', 'PURPOSE_4', 'PURPOSE_5', 'PURPOSE_6', 'PURPOSE_7', 'PURPOSE_8', 'AGE_1', 'AGE_2', 'CHOICE']] #'SM_SEATS', 'SEV_LUGGAGES','ORIG_ROM', 'ORIG_TIC', 'DEST_ROM', 'DEST_TIC', 'AGE_3', 'AGE_4', 

    df_final = df_final.rename(columns= label_name)

    if full_data:
        return df_final
    #split dataset
    df_train, df_test  = train_test_split(df_final, test_size=test_size, random_state=random_state)

    hh_id = df_train.index.tolist()

    #k folds sampled by households for cross validation
    train_idx = []
    test_idx = []
    try:
        train_idx, test_idx = pickle.load(open(r"../Data/strat_group_k_fold_swissmetro.pickle", "rb"))
    except FileNotFoundError:
        for (train_i, test_i) in stratified_group_k_fold(df_train[['TRAIN_TT', 'TRAIN_COST', 'TRAIN_HE', 'SM_TT', 'SM_COST', 'SM_HE', 'CAR_TT', 'CAR_CO']], df_train['choice'], hh_id, k=5):
            train_idx.append(train_i)
            test_idx.append(test_i)
        pickle.dump([train_idx, test_idx], open(r"../Data/strat_group_k_fold_swissmetro.pickle", "wb"))

    folds = zip(train_idx, test_idx)
    
    return df_train, df_test, folds

