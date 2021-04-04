import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from datetime import datetime
from dataloader import get_data_loader, MyDataset


def train(args, device, wandb=None):

    # FLAGS:
    # 0: All
    # 1: High
    # 2: Low
    # 3: Close
    # 4: Adj Close
    full_data = get_data_loader(args, flag=0)

    if args.model_type == "CNN":
        from CNN import CNN
        model = CNN(args).to(device)
    elif args.model_type == "MLP":
        from MLP import MLP
        model = MLP(args).to(device)
    else:
        from LSTM import LSTM
        model = LSTM(args).to(device)

    optimiser = optim.Adam(
        model.parameters(), lr=args.learning_rate)

    loss_fn = args.loss_function()

    if args.load_models:
        print("Loading Models...")
        state = torch.load(args.model_path)
        model.load_state_dict(state['model'])
        optimiser.load_state_dict(state['optimiser'])
        print("Successfully Loaded Models...")

    if args.use_wandb:
        wandb.watch(model, log='all')

    model.train()

    for epoch in trange(1, args.num_epochs + 1):
        total_loss = 0.0
        total_data = 0

        # if args.decay and epoch > 3499:
        #     lr = args.learning_rate - \
        #         ((args.learning_rate / 6500) * (10000 - epoch))
        if args.decay and epoch > 800:
            if epoch > 3000:
                lr = 0.0000005
            elif epoch > 2000:
                lr = 0.000005
            else:
                lr = 0.00003
            for l in range(len(optimiser.param_groups)):
                optimiser.param_groups[l]['lr'] = lr

        for batch_num, data in enumerate(full_data):
            x, y = data[0].float().to(device), data[1].float().to(device)
            total_data += x.shape[0]
            optimiser.zero_grad()
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()

        avg_loss = total_loss / total_data
        if args.use_wandb:
            wandb.log({"Avgerage loss": avg_loss, 'epoch': epoch})

    if args.use_wandb:
        time = datetime.now()
        torch.save({"model": model.state_dict(
        ), "optimiser": optimiser.state_dict()}, 'model{}.pt'.format(time))
        wandb.save('model{}.pt'.format(time))
    else:
        time = datetime.now()
        torch.save({"model": model.state_dict(
        ), "optimiser": optimiser.state_dict()},
            args.save_path + '/model{}.pt'.format(time))


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
        'num_layers': 15,
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
        'use_wandb': False,
        'dropout': False,
        'symbol': "MFC",
        'decay': True,
        'test': False,
        'model_type': "LSTM",
        'device': device
    }
    args.update(args_dict)

    if args.use_wandb:
        # wandb id goes here
        import wandb
        os.system("wandb login 9aca6be0e5d4bb3515f3ad4416f05f2de105257e")
        wandb.init(project="algorithmic_trading")
    else:
        wandb = None

    train(args, device, wandb)
