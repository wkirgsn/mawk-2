from os.path import join
from hyperopt import hp, tpe
from hyperopt.fmin import fmin
import uuid

from tensorflow.python.client import device_lib
import lightgbm
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, \
    cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error

from kirgsn.data import DataManager
import kirgsn.config as cfg
import kirgsn.file_utils as futils


def hyperopt_objective(params):
    # todo: Ugly shit
    param_dict = {'num_leaves': False,
                  'max_depth': False,
                  'scale_pos_weight': True,
                  'colsample_bytree': True,
                  'min_child_weight': True,
                  'random_state': False
                  }
    params = {}
    for p_name, p_range in cfg.lgbm_cfg['hp_hyperopt_space'].items():
        if param_dict[p_name]:
            # True -> integer
            params[p_name] = '{:.3f}'.format(p_range)
        else:
            # False -> real valued
            params[p_name] = int(p_range)
    clf = lightgbm.LGBMRegressor(n_estimators=10000, **params)
    score = cross_val_score(clf, X, Y, scoring=make_scorer(mean_squared_error),
                            cv=TimeSeriesSplit())
    print("MSE: {:.3f} params {}".format(score, params))
    return score


if __name__ == '__main__':
    # config
    gpu_available = len(futils.get_available_gpus()) >= 1
    if cfg.debug_cfg['choose_debug_on_gpu_availability']:
        DEBUG = not gpu_available
    else:
        DEBUG = cfg.debug_cfg['DEBUG']
    if DEBUG:
        print('## DEBUG MODE ON ##')
    n_debug = cfg.debug_cfg['n_debug']
    batch_size = cfg.keras_cfg['batch_size']
    n_epochs = cfg.keras_cfg['n_epochs']
    if cfg.data_cfg['save_predictions']:
        model_uuid = str(uuid.uuid4())[:6]
        print('model uuid: {}'.format(model_uuid))

    dm = DataManager(cfg.data_cfg['file_path'], create_hold_out=False)

    # featurize dataset (feature engineering)
    tra_df, _, tst_df = dm.get_featurized_sets()

    if True:
        # hyperopt
        X = tra_df[dm.cl.x_cols]
        Y = tra_df[dm.cl.y_cols[0]]
        space = cfg.lgbm_cfg['hp_hyperopt_space']

        best = fmin(fn=hyperopt_objective,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=50)
        print("Hyperopt estimated optimum {}".format(best))
    else:
        # random search

        model = lightgbm.LGBMRegressor(n_estimators=10000)
        tscv = TimeSeriesSplit()

        hyper_params = cfg.lgbm_cfg['hp_tuning']
        # todo: Try hyperopt
        rnd_search = \
            RandomizedSearchCV(model, n_iter=100,
                               param_distributions=hyper_params,
                               iid=False, cv=tscv, )
        rnd_search.fit(tra_df[dm.cl.x_cols], tra_df[dm.cl.y_cols[0]])

        print('best params: {}'.format(rnd_search.best_params_))
        print('best score: {}'.format(rnd_search.best_score_))
