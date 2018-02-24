from os.path import join
import uuid

from tensorflow.python.client import device_lib
import lightgbm
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from kirgsn.data import DataManager
import kirgsn.config as cfg
import kirgsn.file_utils as futils

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

    model = lightgbm.LGBMRegressor(n_estimators=10000)
    tscv = TimeSeriesSplit()

    hyper_params = cfg.lgbm_cfg['hp_tuning']
    # todo: Try hyperopt
    rnd_search = \
        RandomizedSearchCV(model,
                           param_distributions=hyper_params,
                           iid=False, cv=tscv, )
    rnd_search.fit(tra_df[dm.cl.x_cols], tra_df[dm.cl.y_cols[0]])

    print('best params: {}'.format(rnd_search.best_params_))
    print('best score: {}'.format(rnd_search.best_score_))
