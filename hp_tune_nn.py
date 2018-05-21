from skopt import BayesSearchCV
import uuid
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_squared_error

from preprocessing.data import DataManager, ReSamplerForBatchTraining
import preprocessing.config as cfg
import preprocessing.file_utils as futils
from hot_nn import build_keras_model, reshape_input_for_batch_train,\
    CustomKerasRegressor


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(opt_search.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(opt_search.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(opt_search.best_score_, 4),
        opt_search.best_params_
    ))

    # Save all model results
    clf_name = opt_search.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


if __name__ == '__main__':
    # config
    gpu_available = len(futils.get_available_gpus()) >= 1
    batch_size = cfg.keras_cfg['params']['batch_size']
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager(cfg.data_cfg['file_path'], create_hold_out=False)

    # featurize dataset (feature engineering)
    tra_df, _, tst_df = dm.get_featurized_sets()
    full_df = tra_df.append(tst_df, ignore_index=True)

    # reorder samples
    resampler = ReSamplerForBatchTraining(batch_size)
    full_df = resampler.fit_transform(full_df)  # there is nothing fitted

    model = CustomKerasRegressor(build_fn=build_keras_model,
                                 verbose=1,
                                 use_gpu=gpu_available)
    tscv = TimeSeriesSplit()

    hyper_params = cfg.keras_cfg['hp_skopt_space']
    opt_search = \
        BayesSearchCV(model, n_iter=2, search_spaces=hyper_params,
                      iid=False, cv=tscv, random_state=2018)
    opt_search.fit(reshape_input_for_batch_train(full_df[dm.cl.x_cols]),
                   full_df[dm.cl.y_cols],
                   callback=status_print)
