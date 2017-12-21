"""This script was for initial data preparation into a dataframe"""

from os.path import join

import pandas as pd
import numpy as np


Input_param_names = ['ambient',
                     'coolant',
                     'u_d',
                     'u_q',
                     'motor_speed',
                     'torque',
                     'i_d',
                     'i_q']

Target_param_names = ['pm',
                      'stator_yoke',
                      'stator_tooth',
                      'stator_winding']

profile_id_colname = 'profile_id'

valset = ['31', ]
testset = ['20', ]
loadsets = ['4', '6', '10', '11', '20', '27', '29', '30',
            '31', '32', '36']


def load_data(path):
    """load data"""

    train_df_list = []
    for p in loadsets:
        train_df_list.append(pd.read_csv(join(path, '{}.csv'.format(p)),
                                         dtype=np.float32))
    # add profile id and merge to one df
    for i, p in enumerate(train_df_list):
        p[profile_id_colname] = str(loadsets[i])
    return pd.concat(train_df_list, axis=0).reset_index(drop=True)


if __name__ == '__main__':
    df = load_data('input')
    df.to_csv('input/measures.csv', index=False)
