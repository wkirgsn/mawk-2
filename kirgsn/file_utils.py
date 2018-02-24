import sqlite3
import numpy as np
from tensorflow.python.client import device_lib
import kirgsn.config as cfg


def save_predictions(id, pred):
    if cfg.data_cfg['save_predictions']:
        with sqlite3.connect(cfg.data_cfg['db_path']) as con:
            # create table if not exists
            query = """CREATE TABLE IF NOT EXISTS 
                    predictions(id text, idx int, {} real, {} real, {} real, 
                    {} real)""".format(*pred.columns)
            con.execute(query)

            # format prediction
            df_to_db = pred.copy()
            df_to_db['id'] = id
            df_to_db['idx'] = pred.index
            entries = [tuple(x) for x in np.roll(df_to_db.values,
                                                 shift=2, axis=1)]
            con.executemany('INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?)',
                            entries)
            print('Predictions of model with uuid {} saved to db.'.format(id))


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']