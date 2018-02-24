import numpy as np

debug_cfg = {'DEBUG': False,
             'choose_debug_on_gpu_availability': False,
             'n_debug': 100,  # first n timestamps to use if debug
             }

data_cfg = {
    'Input_param_names': [#'ambient',
                          #'coolant',
                          'u_d',
                          'u_q',
                          'motor_speed',
                          'torque',
                          'i_d',
                          'i_q'
                           ],
    'Target_param_names': ['pm',
                           'stator_yoke',
                           'stator_tooth',
                           'stator_winding'],
    'lookback': 1,
    'valset': ['31', ],
    'testset': ['20', ],
    'loadsets': ['4', '6',
                 '10', '11',
                 '20', '27',
                 '29', '30',
                 '31', '32', '36'],
    # paths
    'file_path': "input/measures.csv",
    'db_path': 'results.db',
    'save_predictions': True,
    }


plot_cfg = {'do_plot': True, }

keras_cfg = {
    'do_train': False,
    'add_shifts': False,
    'batch_size': 64,
    'n_neurons': 64,
    'n_epochs': 200,
    'arch': 'gru',  # gru, lstm or rnn
    'hp_tuning': {'tbd': None},
}

lgbm_cfg = {
    'params': {'n_estimators': 1000,
               'colsample_bytree': 0.48,
               'num_leaves': 20,
               'scale_pos_weight': 7107,
               'max_depth': 53,
               'min_child_weight': 131.32,
               },
    'hp_tuning': {'num_leaves': list(range(2, 256, 2)),
                    'max_depth': list(range(2, 64)),
                    'scale_pos_weight': list(range(1, 10000)),
                    'colsample_bytree': list(np.linspace(0.3, 1.0)),
                    'min_child_weight': list(np.linspace(0.01, 1000, 100)),
                    'random_state': list(range(2000, 3000, 20))  # arbitrary
                    }
}

train_cfg = {
    'early_stop_patience': 30,
    'l2_reg_w': 0.01,
}


