import torch
import polars as pl
import glob
from util.utility import printf
import numpy as np


class loadPRONOSTIA(torch.utils.data.Dataset):
    def __init__(self, batch_size=64, data_path="./data/phm-ieee-2012-data-challenge-dataset-master/combined_sets/b1/train/*.csv"):
        # load data
        files = glob.glob(data_path)

        dataframes = []

        # load all files into memory
        for file in files:
            raw_df = pl.read_csv(file, has_header=False, separator=",")
            dataframes.append(raw_df)

        df = pl.concat(dataframes)
        
        # init x,y
        x = torch.from_numpy(df.to_numpy()[:,:-1]).float()                 # up to last element
        y = torch.from_numpy(df.to_numpy()[:,-1]).float().unsqueeze(1)     # last element is health label

        # STFT - 64x64
        x = torch.stft(x, 127, window=torch.kaiser_window(127), return_complex=True, win_length=127, hop_length=40)
        x = torch.absolute(x) # magnitude
        x = 20 * torch.log10(x) # to dB

        self.x = x
        self.y = y

        self.dataset_size = self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.dataset_size
    

def getPRONOSTIA(train_path, test_path, batch_size=64, train_test_split=0.5):
    # load dataset to memory
    train_set = loadPRONOSTIA(batch_size=batch_size, data_path=train_path)
    test_set = loadPRONOSTIA(batch_size=128, data_path=test_path)

    # # split data into train/test sets
    # train_size = int(complete_dataset.dataset_size * train_test_split)
    # test_size = complete_dataset.dataset_size - train_size
    # train_set, test_set = torch.utils.data.random_split(complete_dataset, [train_size, test_size])


    # create data loaders
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    printf("- Train size: %d (%d)\n- Test size: %d (%d)\n", len(loader_train.dataset), len(loader_train.dataset) // batch_size, len(loader_test.dataset), len(loader_test.dataset) // batch_size)

    return loader_train, loader_test