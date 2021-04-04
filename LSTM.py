from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN NETWORK


class LSTM(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm = nn.LSTM(
            input_size=args.input_dim, hidden_size=args.perceptrons_per_layer, num_layers=args.num_layers, bidirectional=False, batch_first=True)
        self.linear = nn.Linear(args.perceptrons_per_layer, args.output_dim)

    def forward(self, x):
        x = x.reshape(1, x.shape[0], x.shape[1])

        h0 = torch.zeros(self.args.num_layers, x.size(
            0), self.args.perceptrons_per_layer).to(self.args.device)

        c0 = torch.zeros(self.args.num_layers, x.size(
            0), self.args.perceptrons_per_layer).to(self.args.device)

        output, _ = self.lstm(x, (h0, c0))
        output = self.linear(output)
        return output


if __name__ == '__main__':

    from attrdict import AttrDict
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = AttrDict()
    args_dict = {
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 10000,
        'input_dim': 17,
        'output_dim': 4,
        'num_layers': 7,
        'num_conv_layers': 6,  # num of convolution layers will be 1 less than this
        'conv_channels': [1, 2, 4, 8, 16, 1],  # length same as number above
        'perceptrons_per_layer': 25,
        'perceptrons_in_conv_layers': 35,
        'res_channel': 64,
        'num_residual_layers': 6,
        'load_models': False,
        'model_path': "/content/model100.pt",
        'activation': nn.ReLU,
        'norm': nn.BatchNorm1d,
        'loss_function': nn.MSELoss,
        'save_path': "models/MLP",
        'use_wandb': True,
        'dropout': False,
        'symbol': "MFC",
        'decay': True,
        'test': False,
        'model_type': "LSTM",
        'device': device
    }
    args.update(args_dict)

    x = torch.rand(args.batch_size, args.input_dim)
    model = LSTM(args)
    y = model(x)

    print(y)
