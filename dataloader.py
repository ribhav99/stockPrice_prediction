from torch.utils.data import Dataset, DataLoader
from train_test_split import my_train_test_split
from torchvision import transforms


class MyDataset(Dataset):

    def __init__(self, args, flag=0):
        self.x_train, self.x_test, self.y_train, self.y_test, _ = my_train_test_split(
            symbol=args.symbol, flag=flag)
        self.args = args

    def __len__(self):
        if not self.args.test:
            return len(self.x_train)
        return len(self.x_test)

    def __getitem__(self, idx):
        if not self.args.test:
            return self.x_train[idx], self.y_train[idx]
        return self.x_test[idx], self.y_test[idx]


def get_data_loader(args, flag=0):
    data_set = MyDataset(args, flag)
    dataloader = DataLoader(data_set, batch_size=args.batch_size, shuffle=True)
    return dataloader


if __name__ == '__main__':
    from attrdict import AttrDict
    import torch.nn as nn
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
        'decay': False,
        'test': False
    }
    args.update(args_dict)

    full_data = get_data_loader(args)
    for batch_num, data in enumerate(full_data):
        x, y = data[0], data[1]
        print(type(x[0]))
        break
