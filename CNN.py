from attrdict import AttrDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# CNN NETWORK


class CNN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.conv = []
        self.linear = []
        for i in range(args.num_layers):
            if i == 0:
                self.linear.append(
                    nn.Linear(args.input_dim, args.perceptrons_per_layer))
                self.linear.append(args.norm(args.perceptrons_per_layer))
                self.linear.append(args.activation())
                if args.dropout:
                    self.linear.append(nn.Dropout(args.dropout))
            else:
                self.linear.append(
                    nn.Linear(args.perceptrons_per_layer, args.perceptrons_per_layer))
                self.linear.append(args.norm(args.perceptrons_per_layer))
                self.linear.append(args.activation())
                if args.dropout:
                    self.linear.append(nn.Dropout(args.dropout))

        for i in range(args.num_conv_layers - 1):
            self.conv.append(
                nn.Conv1d(args.conv_channels[i], args.conv_channels[i+1], kernel_size=2, padding=1))
            self.conv.append(args.norm(args.conv_channels[i+1]))
            self.conv.append(args.activation())

        self.linear_layers = nn.Sequential(*self.linear)
        self.conv_layers = nn.Sequential(*self.conv)

        fake_data = torch.rand(args.batch_size, args.input_dim)
        fake1 = self.linear_layers(fake_data)
        fake1 = fake1.reshape(fake1.shape[0], 1, fake1.shape[1])
        fake2 = self.conv_layers(fake1)
        self._linear_dim = np.prod(fake2[0].shape)

        self.final = nn.Sequential(
            nn.Linear(self._linear_dim, args.output_dim), args.norm(args.output_dim), args.activation())

    def forward(self, x):
        x = self.linear_layers(x)
        x = x.reshape(x.shape[0], 1, x.shape[1])
        x = self.conv_layers(x)
        x = x.view(-1, self._linear_dim)
        x = self.final(x)
        return x


if __name__ == '__main__':

    from attrdict import AttrDict
    import os
    args = AttrDict()
    args_dict = {
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 10000,
        'input_dim': 17,
        'output_dim': 4,
        'num_layers': 2,
        'num_conv_layers': 5,  # num of convolution layers will be 1 less than this
        'conv_channels': [1, 2, 4, 8, 16],  # length same as number above
        'perceptrons_per_layer': 25,
        'perceptrons_in_conv_layers': 15,
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
        'test': False
    }
    args.update(args_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.rand(args.batch_size, args.input_dim)

    model = CNN(args)
    y = model(x)

    print(y)
