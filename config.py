
debug_cfg = {'DEBUG': False,
             'choose_debug_on_gpu_availability': True,
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
    }

path_cfg = {
    'file_path': "/home/wilhelmk/Messdaten/PMSM_Lastprofile/hdf"
                 "/all_load_profiles",
}

plot_cfg = {'do_plot': True, }

keras_cfg = {
    'do_train': False,
    'add_shifts': False,
    'batch_size': 64,
    'n_neurons': 64,
    'n_epochs': 200,
    'arch': 'gru'  # gru, lstm or rnn
}

train_cfg = {
    'early_stop_patience': 30,
    'l2_reg_w': 0.01,
}


