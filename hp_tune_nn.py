from skopt import BayesSearchCV
import uuid
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, \
    cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from preprocessing.data import DataManager
import preprocessing.config as cfg
from hot_nn import build_keras_model


def status_print(optim_result):
    """Status callback during bayesian hyperparameter search"""

    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(opt_search.cv_results_)

    # Get current parameters and the best parameters
    best_params = pd.Series(opt_search.best_params_)
    print('Model #{}\nBest L2: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(opt_search.best_score_, 4),
        opt_search.best_params_
    ))

    # Save all model results
    clf_name = opt_search.estimator.__class__.__name__
    all_models.to_csv(clf_name + "_cv_results.csv")


if __name__ == '__main__':
    # config
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager(cfg.data_cfg['file_path'], create_hold_out=False)

    # featurize dataset (feature engineering)
    tra_df, _, tst_df = dm.get_featurized_sets()
    tra_df = tra_df.append(tst_df, ignore_index=True)

    model = build_keras_model()
    tscv = TimeSeriesSplit()

    hyper_params = cfg.lgbm_cfg['hp_skopt_space']
    opt_search = \
        BayesSearchCV(model, n_iter=2, search_spaces=hyper_params,
                      iid=False, cv=tscv, random_state=2018)
    opt_search.fit(tra_df[dm.cl.x_cols],
                   tra_df[dm.cl.y_cols[0]],
                   callback=status_print)
