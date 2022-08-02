import time
import config
from torch import nn
import os
import torch
import network
from tqdm import tqdm
import numpy as np
import pandas as pd
from collections import OrderedDict
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


class Transformer_model(nn.Module):
    def __init__(self):
        super(Transformer_model, self).__init__()
        self.Transformer_autoencoder = network.Transformer_net(num_encoder_block=config.num_encoder_block,
                                                               num_decoder_block=config.num_decoder_block,
                                                               act_mode=config.MLP_act).to(
            device=config.device)
        self.writer = SummaryWriter('runs/Transformer_test_1')
        self.criterion = nn.BCELoss().to(device=config.device)
        self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)

    def save(self, epoch):
        checkpoint_path = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % (epoch))
        if not os.path.exists(config.Transformer_dir):
            os.makedirs(config.Transformer_dir, exist_ok=True)
        torch.save(self.Transformer_autoencoder.state_dict(), checkpoint_path)
        print("Checkpoint has been created.")

    def load(self, checkpoint_name):
        ckpt = torch.load(checkpoint_name, map_location=config.device)
        self.Transformer_autoencoder.load_state_dict(ckpt, strict=True)
        print("Restored model parameters from {}".format(checkpoint_name))

    def accuracy_calculate(self, predict, target, predict_frame_num):
        acc = 0
        # predict = np.round(predict)
        for i in range(predict.shape[0]):
            predict[i, np.sum(predict[i, :, 0:32], axis=1) >= 16, 0:32] = np.ones(32)
            predict[i, np.sum(predict[i, :, 0:32], axis=1) < 16, 0:32] = np.zeros(32)

            predict[i, np.sum(predict[i, :, 32:64], axis=1) >= 16, 32:64] = np.ones(32)
            predict[i, np.sum(predict[i, :, 32:64], axis=1) < 16, 32:64] = np.zeros(32)

            predict[i, np.sum(predict[i, :, 64:96], axis=1) >= 16, 64:96] = np.ones(32)
            predict[i, np.sum(predict[i, :, 64:96], axis=1) < 16, 64:96] = np.zeros(32)

            if (predict[i][:predict_frame_num, :] == target[i][:predict_frame_num, :]).all():
                acc += 1
        return acc / predict.shape[0]

    def info_for_tensorboard(self, dic_data, loss_train, epochs):
        num_test = 100
        train_perform = []
        val_perform = []
        train_acc_return = 0
        val_acc_return = 0
        decoder_inputs = dic_data["training_type"]
        decoder_inputs = np.concatenate((-1 * np.ones((num_test, 1, config.output_size)),
                                         decoder_inputs[:num_test, :-1, :]), axis=1)
        val_decoder_inputs = dic_data["validating_type"]
        val_decoder_inputs = np.concatenate((-1 * np.ones((num_test, 1, config.output_size)),
                                             val_decoder_inputs[:num_test, :-1, :]), axis=1)
        for i in range(num_test):
            train_out = torch.round(
                self.Transformer_autoencoder(torch.reshape(torch.tensor(dic_data["training_input"][i, :, :],
                                                                        dtype=torch.float,
                                                                        device=config.device),
                                                           (-1, config.input_num_symbol, config.input_size)),
                                             torch.reshape(torch.tensor(decoder_inputs[i, :, :],
                                                                        dtype=torch.float,
                                                                        device=config.device),
                                                           (-1, config.input_num_symbol, config.embedded_dim))))
            train_perform.append(train_out.detach().cpu().numpy())

            val_out = torch.round(
                self.Transformer_autoencoder(torch.reshape(torch.tensor(dic_data["validating_input"][i, :, :],
                                                                        dtype=torch.float,
                                                                        device=config.device),
                                                           (-1, config.input_num_symbol, config.input_size)),
                                             torch.reshape(torch.tensor(val_decoder_inputs[i, :, :],
                                                                        dtype=torch.float,
                                                                        device=config.device),
                                                           (-1, config.input_num_symbol, config.embedded_dim))))
            val_perform.append(val_out.detach().cpu().numpy())

        train_perform = np.reshape(np.array(train_perform),
                                   newshape=(num_test, config.input_num_symbol, config.output_size))
        val_perform = np.reshape(np.array(val_perform),
                                 newshape=(num_test, config.input_num_symbol, config.output_size))

        loss_val = self.criterion(torch.tensor(val_perform, dtype=torch.float, device=config.device),
                                  torch.tensor(dic_data["validating_type"][:num_test, :], dtype=torch.float,
                                               device=config.device))

        self.writer.add_scalars('Loss', {'Train Loss': loss_train,
                                         'Val Loss': loss_val}, epochs)

        for predict_frame_num in [1, 2, 4, 8, 16, 24, 32]:
            train_acc = self.accuracy_calculate(train_perform, dic_data["training_type"][:num_test, :, :],
                                                predict_frame_num)
            val_acc = self.accuracy_calculate(val_perform, dic_data["validating_type"][:num_test, :, :],
                                              predict_frame_num)

            self.writer.add_scalars('Accuracy for ' + str(predict_frame_num) + 'frames prediction',
                                    {'Train Accuracy': train_acc, 'Val Accuracy': val_acc}, epochs)

            if predict_frame_num == config.acc_print_num_frame:
                train_acc_return = train_acc
                val_acc_return = val_acc
        return train_acc_return, val_acc_return, loss_val

    def train(self, dic_data):
        self.Transformer_autoencoder.train()
        self.writer.add_graph(self.Transformer_autoencoder, [torch.rand((1, 32, config.input_size), device=config.device
                                                                        , dtype=torch.float),
                                                             torch.rand((1, 32, config.output_size),
                                                                        device=config.device, dtype=torch.float)])
        # summary(self.Transformer_autoencoder, [(32, 256), (32, 256)])
        encoder_inputs = dic_data["training_input"]
        decoder_inputs = dic_data["training_type"]
        decoder_inputs = np.concatenate((-1 * np.ones((encoder_inputs.shape[0], 1, config.output_size)),
                                         decoder_inputs[:, :-1, :]), axis=1)
        y_train = dic_data["training_type"]

        num_samples = encoder_inputs.shape[0]
        num_batches = num_samples // config.batch_size
        max_val_acc = 0

        print('### Training... ###')
        for epoch in range(1, config.max_epoch + 1):
            shuffle_index = np.random.permutation(num_samples)
            start_time = time.time()
            curr_encoder_inputs = encoder_inputs[shuffle_index]
            curr_decoder_inputs = decoder_inputs[shuffle_index]
            curr_y_train = y_train[shuffle_index]
            loss_ = 0

            if epoch == 1:
                config.learning_rate = 0.0001
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate)
            elif epoch == int(config.max_epoch * 0.5):
                config.learning_rate = 0.00005
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate)
            elif epoch == int(config.max_epoch * 0.6):
                config.learning_rate = 0.00001
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate)
            elif epoch == int(config.max_epoch * 0.9):
                config.learning_rate = 0.000001
                self.optimizer = torch.optim.Adam(params=self.Transformer_autoencoder.parameters(),
                                                  lr=config.learning_rate)

            for i in range(num_batches):
                current_batch_encoder_input = torch.tensor(curr_encoder_inputs[config.batch_size * i:
                                                                               config.batch_size * (i + 1), :, :],
                                                           dtype=torch.float, device=config.device)
                current_batch_decoder_input = torch.tensor(curr_decoder_inputs[config.batch_size * i:
                                                                               config.batch_size * (i + 1), :, :],
                                                           dtype=torch.float, device=config.device)
                current_batch_y_train = torch.tensor(curr_y_train[config.batch_size * i:
                                                                  config.batch_size * (i + 1), :],
                                                     dtype=torch.float, device=config.device)

                curr_batch_y_pred = self.Transformer_autoencoder(current_batch_encoder_input,
                                                                 current_batch_decoder_input)
                loss = self.criterion(curr_batch_y_pred, current_batch_y_train)
                loss_ += loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, loss), end='\r', flush=True)

            duration = time.time() - start_time
            train_acc, val_acc, loss_val = self.info_for_tensorboard(dic_data, loss_.item() / num_batches, epochs=epoch)

            print('Epoch {:d}; Train Loss {:.6f}; Val Loss {:.6f}; '
                  'Train Accuracy {:.6f}; Val Accuracy {:.6f}; Duration {:.3f} seconds.'
                  .format(epoch, loss_ / num_batches, loss_val, train_acc, val_acc, duration))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                self.save(epoch)

        self.writer.close()

    def test_or_validate(self, dic_data, checkpoint_num_list):
        encoder_inputs = dic_data["testing_input"]
        decoder_inputs = dic_data["testing_input"]
        y = dic_data["testing_type"]

        self.Transformer_autoencoder.eval()
        print('### Test or Validation ###')

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)

            preds = []
            for i in tqdm(range(encoder_inputs.shape[0])):
                encoder_inter = [encoder_inputs[i]]
                decoder_inter = [decoder_inputs[i]]
                out = self.Transformer_autoencoder(torch.tensor(encoder_inter, dtype=torch.float, device=config.device),
                                                   torch.tensor(decoder_inter, dtype=torch.float, device=config.device))
                out = torch.argmax(out, dim=-1)
                preds.append(out.detach().cpu().numpy())

            preds = np.reshape(preds, (-1, config.output_size))
            sum = 0
            for i in range(y.shape[0]):
                if preds[i] == (y[i]):
                    sum += 1
            accuracy = sum / y.shape[0]
            print('Test accuracy: {:.4f}'.format(accuracy))

    def save_matrix(self, checkpoint_num_list):
        self.Transformer_autoencoder.eval()
        print('### Save Weight Matrices ###')

        isExist = os.path.exists(config.Transformer_weight_save_dir)
        if not isExist:
            os.makedirs(config.Transformer_weight_save_dir)
            print("The new directory is created")

        for checkpoint_num in checkpoint_num_list:
            checkpoint_file = os.path.join(config.Transformer_dir, 'model-%d.ckpt' % checkpoint_num)
            self.load(checkpoint_file)
            for params in self.Transformer_autoencoder.state_dict():
                weight = self.Transformer_autoencoder.state_dict()[params].cpu().numpy()
                checkpoint_file = os.path.join(config.Transformer_weight_save_dir,
                                               'model-%d-%s.csv' % (checkpoint_num, params.replace(".", "_")))
                pd.DataFrame(weight).to_csv(checkpoint_file, header=False, index=False)
