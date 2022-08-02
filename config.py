import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = int(96 * 2)
input_num_symbol = 32
batch_size = 128
output_size = 96
vec_size = input_size * input_num_symbol
type_position = vec_size
num_classes = 2
sig_shape = [-1, input_num_symbol, input_size]

weight_decay = 0.8
learning_rate = 0.1
save_interval = 100
max_epoch = 300

embedded_dim = int(input_size / 2)
num_heads = 8

acc_print_num_frame = 24

##-----activation-----
coder_act = 'relu'
MLP_act = 'relu'

##-----file direction--
training_data_path = './Data/mtx_32f_B_30_35dB_small.csv'
testing_data_path = './Data/mtx_32f_B_20_25dB.csv'
validation_data_path = './Data/mtx_32f_B_15_20dB_small.csv'
model_dir = './Model'
Transformer_dir = model_dir + '/Transformer_whole'
Transformer_weight_save_dir = Transformer_dir + '/weight_mtx'

##-----encoder---------
num_encoder_block = 4
encoder_drop_rate = 0.5
encoder_dense_dim = 1024
encoder_norm_mode = 'layer'     #'batch' or 'layer'
##-----decoder---------
num_decoder_block = 4
decoder_drop_rate = 0.5
decoder_dense_dim = 1024
decoder_norm_mode = 'layer'     #'batch' or 'layer'
##-----LSTM---------
LSTM_num_layers = 2
LSTM_drop_rate = 0.5
bidirectional_LSTM = True
