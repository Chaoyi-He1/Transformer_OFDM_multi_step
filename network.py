import numpy as np
import config
from torch import nn
import torch


class Transformer_Encoder_Block(nn.Module):
    def __init__(self, act_mode):
        super(Transformer_Encoder_Block, self).__init__()
        self.act_mode = act_mode
        self.attention_layer = nn.MultiheadAttention(embed_dim=config.embedded_dim, num_heads=config.num_heads,
                                                     dropout=config.encoder_drop_rate, batch_first=True)
        if config.encoder_norm_mode == 'layer':
            self.norm_1 = nn.LayerNorm(normalized_shape=config.embedded_dim)
        elif config.encoder_norm_mode == 'batch':
            self.norm_1 = nn.BatchNorm1d(num_features=config.input_num_symbol)
        self.MLP_1 = nn.Linear(in_features=config.embedded_dim, out_features=config.encoder_dense_dim)
        self.dropout = nn.Dropout(config.encoder_drop_rate)
        self.MLP_2 = nn.Linear(in_features=config.encoder_dense_dim, out_features=config.embedded_dim)
        if config.encoder_norm_mode == 'layer':
            self.norm_2 = nn.LayerNorm(normalized_shape=config.embedded_dim)
        elif config.encoder_norm_mode == 'batch':
            self.norm_2 = nn.BatchNorm1d(num_features=config.input_num_symbol)

    def forward(self, inputs):
        outputs = inputs
        outputs = self.attention_layer(query=outputs, key=outputs, value=outputs, need_weights=False)[0]
        outputs = self.norm_1(outputs + inputs)
        res = outputs
        outputs = self.MLP_1(outputs)
        if self.act_mode == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.act_mode == 'relu':
            outputs = torch.relu(outputs)
        elif self.act_mode == 'tanh':
            outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.MLP_2(outputs)
        if self.act_mode == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.act_mode == 'relu':
            outputs = torch.relu(outputs)
        elif self.act_mode == 'tanh':
            outputs = torch.tanh(outputs)
        outputs = self.norm_2(res + outputs)
        return outputs


class Transformer_Decoder_Block(nn.Module):
    def __init__(self, act_mode):
        super(Transformer_Decoder_Block, self).__init__()
        self.act_mode = act_mode
        self.attention_layer_1 = nn.MultiheadAttention(embed_dim=config.embedded_dim, num_heads=config.num_heads,
                                                       dropout=config.decoder_drop_rate, batch_first=True)
        if config.decoder_norm_mode == 'batch':
            self.norm_1 = nn.BatchNorm1d(num_features=config.input_num_symbol)
        elif config.decoder_norm_mode == 'layer':
            self.norm_1 = nn.LayerNorm(normalized_shape=config.embedded_dim)
        self.attention_layer_2 = nn.MultiheadAttention(embed_dim=config.embedded_dim, num_heads=config.num_heads,
                                                       dropout=config.decoder_drop_rate, batch_first=True)
        if config.decoder_norm_mode == 'batch':
            self.norm_2 = nn.BatchNorm1d(num_features=config.input_num_symbol)
        elif config.decoder_norm_mode == 'layer':
            self.norm_2 = nn.LayerNorm(normalized_shape=config.embedded_dim)
        self.MLP_1 = nn.Linear(in_features=config.embedded_dim, out_features=config.decoder_dense_dim)
        self.dropout = nn.Dropout(config.decoder_drop_rate)
        self.MLP_2 = nn.Linear(in_features=config.decoder_dense_dim, out_features=config.embedded_dim)
        if config.decoder_norm_mode == 'batch':
            self.norm_3 = nn.BatchNorm1d(num_features=config.input_num_symbol)
        elif config.decoder_norm_mode == 'layer':
            self.norm_3 = nn.LayerNorm(normalized_shape=config.embedded_dim)

    def decoder_mask(self, inputs):
        num_seq = inputs.size()[1]
        i = torch.arange(0, num_seq, device=config.device)[:, None]
        j = torch.arange(0, num_seq, device=config.device)
        mask = i >= j
        return ~mask

    def forward(self, inputs, encoder_outputs):
        mask_decoder = self.decoder_mask(inputs)
        outputs_atten_1 = self.attention_layer_1(query=inputs, key=inputs, value=inputs, attn_mask=mask_decoder,
                                                 need_weights=False)[0]
        outputs_atten_1 = self.norm_1(outputs_atten_1 + inputs)
        outputs_atten_2 = self.attention_layer_2(query=outputs_atten_1, key=encoder_outputs, value=encoder_outputs,
                                                 attn_mask=mask_decoder, need_weights=False)[0]
        outputs_atten_2 = self.norm_2(outputs_atten_2 + outputs_atten_1)
        outputs = self.MLP_1(outputs_atten_2)
        if self.act_mode == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.act_mode == 'relu':
            outputs = torch.relu(outputs)
        elif self.act_mode == 'tanh':
            outputs = torch.tanh(outputs)
        outputs = self.dropout(outputs)
        outputs = self.MLP_2(outputs)
        if self.act_mode == 'sigmoid':
            outputs = torch.sigmoid(outputs)
        elif self.act_mode == 'relu':
            outputs = torch.relu(outputs)
        elif self.act_mode == 'tanh':
            outputs = torch.tanh(outputs)
        outputs = self.norm_3(outputs + outputs_atten_2)
        return outputs


class Transformer_net(nn.Module):
    def __init__(self, num_encoder_block=16, num_decoder_block=16, act_mode='relu'):
        super(Transformer_net, self).__init__()
        self.num_encoder_block = num_encoder_block
        self.num_decoder_block = num_decoder_block
        self.act_mode = act_mode
        self.encoder_module = nn.ModuleList()
        self.decoder_module = nn.ModuleList()
        self.MLP_input = nn.Linear(in_features=config.input_size, out_features=config.embedded_dim)
        for _ in range(config.num_encoder_block):
            self.encoder_module.append(Transformer_Encoder_Block(act_mode=config.coder_act))
        for _ in range(config.num_decoder_block):
            self.decoder_module.append(Transformer_Decoder_Block(act_mode=config.coder_act))
        self.bi_LSTM = nn.LSTM(input_size=config.embedded_dim, hidden_size=config.embedded_dim, batch_first=True,
                               num_layers=config.LSTM_num_layers, dropout=config.LSTM_drop_rate,
                               bidirectional=config.bidirectional_LSTM)
        self.MLP_1 = nn.Linear(in_features=int(config.embedded_dim * 2), out_features=config.decoder_dense_dim)
        self.dropout_1 = nn.Dropout(config.decoder_drop_rate)
        self.MLP_2 = nn.Linear(in_features=config.decoder_dense_dim, out_features=config.decoder_dense_dim)
        self.dropout_2 = nn.Dropout(config.decoder_drop_rate)
        self.MLP_3 = nn.Linear(in_features=config.decoder_dense_dim, out_features=config.output_size)

    def forward(self, encoder_inputs, decoder_inputs):
        encoder_outputs = encoder_inputs
        decoder_outputs = decoder_inputs
        encoder_outputs = self.MLP_input(encoder_outputs)
        for i in range(config.num_encoder_block):
            encoder_outputs = self.encoder_module[i](encoder_outputs)
        for i in range(config.num_decoder_block):
            decoder_outputs = self.decoder_module[i](decoder_outputs, encoder_outputs)
        decoder_outputs = self.bi_LSTM(decoder_outputs)[0]

        decoder_outputs = self.MLP_1(decoder_outputs)
        if self.act_mode == 'sigmoid':
            decoder_outputs = torch.sigmoid(decoder_outputs)
        elif self.act_mode == 'relu':
            decoder_outputs = torch.relu(decoder_outputs)
        elif self.act_mode == 'tanh':
            decoder_outputs = torch.tanh(decoder_outputs)
        decoder_outputs = self.dropout_1(decoder_outputs)

        decoder_outputs = self.MLP_2(decoder_outputs)
        if self.act_mode == 'sigmoid':
            decoder_outputs = torch.sigmoid(decoder_outputs)
        elif self.act_mode == 'relu':
            decoder_outputs = torch.relu(decoder_outputs)
        elif self.act_mode == 'tanh':
            decoder_outputs = torch.tanh(decoder_outputs)
        decoder_outputs = self.dropout_2(decoder_outputs)

        decoder_outputs = self.MLP_3(decoder_outputs)
        decoder_outputs = torch.sigmoid(decoder_outputs)

        return decoder_outputs
