from attrdict import AttrDict
import torch
import torch.nn as nn
import numpy as np


class MLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.layers = []
        for i in range(args.num_layers):
            if i == 0:
                self.layers.append(
                    nn.Linear(args.input_dim, args.perceptrons_per_layer))
                self.layers.append(args.norm(args.perceptrons_per_layer))
                self.layers.append(args.activation())
                if args.dropout:
                    self.layers.append(nn.Dropout(args.dropout))
            elif i == args.num_layers - 1:
                self.layers.append(
                    nn.Linear(args.perceptrons_per_layer, args.output_dim))
                self.layers.append(args.norm(args.output_dim))
                self.layers.append(args.activation())
                if args.dropout:
                    self.layers.append(nn.Dropout(args.dropout))
            else:
                self.layers.append(
                    nn.Linear(args.perceptrons_per_layer, args.perceptrons_per_layer))
                self.layers.append(args.norm(args.perceptrons_per_layer))
                self.layers.append(args.activation())
                if args.dropout:
                    self.layers.append(nn.Dropout(args.dropout))

        self.linear = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':

    from attrdict import AttrDict
    args = AttrDict()
    args_dict = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'num_epochs': 500,
        'input_dim': 17,
        'output_dim': 4,
        'num_layers': 5,
        'perceptrons_per_layer': 15,
        'res_channel': 64,
        'num_residual_layers': 6,
        'load_models': False,
        'model_path': "/content/model100.pt",
        'activation': nn.ReLU,
        'norm': nn.BatchNorm1d,
        'loss_function': nn.MSELoss,
        'save_path': "/content/GAN_Style_Transfer/Models",
        'use_wandb': False,
        'dropout': False,
        'symbol': "MFC",
        'decay': False
    }
    args.update(args_dict)
    x = torch.rand(args.batch_size, args.input_dim)
    model = MLP(args)
    print(model(x)[0][0])
