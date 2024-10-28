# rumboost-paper-code
All the code needed to reproduce the RUMBoost paper results. The code in ML classifiers is mostly adapted from [prediction-behavioural-analysis-ml-travel-mode-choice
](https://github.com/JoseAngelMartinB/prediction-behavioural-analysis-ml-travel-mode-choice), from [reslogit](https://github.com/LiTrans/reslogit) for the ResLogit and from [EnhancedDCM](https://github.com/BSifringer/EnhancedDCM) for the L-MNL.

 - The rumboost and RUMs folder contains:
    * the underlying rumboost code (rumbooster.py, utils.py, models.py, function_smoothing.py, datasets.py and benchmarks.py). This mostly corresponds to [rumboost version 1.0.2](https://github.com/big-ucl/rumboost/tree/v1.0.2).
    * the jupyter notebooks to run all rumboost and RUMs models on the LPMC dataset (lpmc_experiments_rumboost.ipynb)
    * the python script used to tune hyperparameters for RUMBoost-Nested and RUMBoost-FE (rumboost_hyperparameter_search.py)
    * the python scripts to run PCUF on Swissmetro and LPMC datasets (PCUF_Swissmetro.py and PCUF_LPMC.py)
    * the jupyter notebook used to generate most figures (figures.ipynb)
    * the jupyter notebook used for the boostrapping experiment (bootstrap.ipynb)
 - The ML classifiers model contains:
    * Lightgbm, NN, DNN, ResLogit and L-MNL underlying code (Models/LightGBM.py, Models/NN.py, Models/DNN.py, Models/ResLogit and EnhancedDCM/utilities)
    * The jupyter notebook used to run the LPMC experiments for LightGBM, NN, DNN, and ResLogit (3-Experiment-4-RealDatasets.ipynb)
    * The python script to tune hyperparameters on the LPMC datasets for LightGBM, NN, DNN, and ResLogit (1-LPMC-HyperparameterTuning.py)
    * The script to run hyperparameter tuning and testing on LPMC for L-MNL (EnhancedDCM/ready_example/lpmc_paper_run.py). Note that you need to run the jupyter notebook create_dataset (EnhancedDCM/ready_example/swissmetro_paper/create_dataset.ipynb) first to preprocess the LPMC dataset.
    * The jupyter notebook to tune hyperparameters and testing for all models on the Swissmetro dataset (SWISSMETRO.ipynb)
    * The jupyter notebook to run the semi-synthetic experiment (synthetic_experiment.ipynb)
- The Data folder contains the dataset needed for experiments, and all model results.
- The Figure folder contains all figures put in the paper, and additional ones, including gifs representing how the model is learning.
