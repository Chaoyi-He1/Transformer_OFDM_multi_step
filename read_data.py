import numpy as np
import pandas as pd
import config


def read_data(file_name):
    data = np.array(pd.read_csv(file_name, header=None))
    input_data = data[:, :config.vec_size].copy()
    label = np.reshape(data[:, config.type_position:], (-1, config.input_num_symbol, config.output_size))
    # r, c = np.shape(input_data)
    # signal_type_one_hot = np.zeros([r, 2])
    # signal_type_one_hot[np.arange(r), (data[:, config.type_position]).astype(int)] = 1
    # spec = data[:, config.spectrum_use_position]
    return input_data, label


def reshape_data(data_in):
    # data_out = []
    # r, c = np.shape(data_in)
    # if config.vec_size != np.shape(np.reshape(data_in[0], config.sig_shape))[2]:
    #     raise ValueError("input length is wrong")   # raise Exception("xxx")
    # for i in range(r):
        # data_out.append(np.reshape(data_in[i], config.sig_shape))
    return np.reshape(data_in, config.sig_shape)


def training_read(file_name):
    training_input, training_type = read_data(file_name)
    training_input = reshape_data(training_input)
    return training_input, training_type


def validating_read(file_name):
    validating_input, validating_type = read_data(file_name)
    validating_input = reshape_data(validating_input)
    return validating_input, validating_type


def testing_read(file_name):
    testing_input, testing_type = read_data(file_name)
    testing_input = reshape_data(testing_input)
    return testing_input, testing_type
