import numpy as np
import pandas as pd

N = 9036
cv_loss = 0
best_ce = 0
time = 0

for i in range(20):
    data = pd.read_csv(f'best_num_knots_SwissMetro_RUMBoost__{i}_test.csv', index_col=0)

    n_knots = data.loc['CAR_COST'].astype(float) + data.loc['TRAIN_COST'].astype(float) + data.loc['SM_COST'].astype(float) + data.loc['TRAIN_TT'].astype(float) + data.loc['SM_TT'].astype(float) + data.loc['CAR_TT'].astype(float)

    cv_it = (data.loc['_best_loss'].astype(float) - (n_knots - 6) * np.log(N*0.7)) / (2*N*0.7)
    best_ce += data.loc['_on_test_set'].astype(float)
    time += data.loc['_elapsed_time'].astype(float)
    cv_loss += cv_it

print(cv_loss/20)
print(best_ce/20)
print(time/20)
