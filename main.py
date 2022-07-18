import numpy as np
import config
import model
import read_data


def data_reader(train=False, val=False, test=False):
    data_dic = dict()
    if train:
        training_input, training_type = read_data.training_read(config.training_data_path)
        data_dic.update(zip(["training_input", "training_type"],
                            [training_input, training_type]))
    if val:
        validating_input, validating_type = read_data.validating_read(
            config.validation_data_path)
        data_dic.update(zip(["validating_input", "validating_type"],
                            [validating_input, validating_type]))
    if test:
        testing_input, testing_type = read_data.testing_read(config.testing_data_path)
        data_dic.update(zip(["testing_input", "testing_type"],
                            [testing_input, testing_type]))
    return data_dic


if __name__ == '__main__':
    dic_data = data_reader(train=True, val=True, test=False)

    Transformer_model = model.Transformer_model()
    Transformer_model.train(dic_data=dic_data)
    # Transformer_model.test_or_validate(dic_data=dic_data,
    #                                    checkpoint_num_list=[100, 200, 300, 400, 500, 600, 700, 800, 900])
