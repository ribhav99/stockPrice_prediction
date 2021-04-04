import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from datetime import datetime
from dataloader import get_data_loader, MyDataset


def test(args, device):
    full_data = get_data_loader(args)

    if args.model_type == "CNN":
        from CNN import CNN
        model = CNN(args).to(device)
    elif args.model_type == "MLP":
        from MLP import MLP
        model = MLP(args).to(device)
    elif args.model_type == "LSTM":
        from LSTM import LSTM
        model = LSTM(args).to(device)

    optimiser = optim.Adam(
        model.parameters(), lr=args.learning_rate)

    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state['model'])
    optimiser.load_state_dict(state['optimiser'])

    total_difference = 0
    n = 0

    for batch_num, data in enumerate(full_data):
        x, y = data[0].float().to(device), data[1].float().to(device)
        num_of_predictions = x.shape[0]
        pred = model(x)
        pred = pred.reshape(y.shape)
        total_difference += sum((abs(pred - y)/y) * 100)
        n += num_of_predictions

    return total_difference/n


if __name__ == '__main__':
    from attrdict import AttrDict
    args = AttrDict()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args_dict = {
        'learning_rate': 0.0001,
        'batch_size': 16,
        'num_epochs': 5000,
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
        'model_path': "models/LSTM/model1.pt",
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

    print(test(args, device))
