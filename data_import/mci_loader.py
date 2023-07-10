import torch
import polars as pl
import glob
from util.utility import printf
import numpy as np


class loadMCI(torch.utils.data.Dataset):
    def __init__(self, data_path="mci_dataset/*.csv", batch_size=64):
        # load data
        dirs = ["H/*.csv", "IR/*.csv", "OR/*.csv", "W/*.csv"]
        dataframes = []

        for i in range(len(dirs)):
            files = glob.glob(data_path + dirs[i])

            # load all files into memory
            for file in files:
                raw_df = pl.read_csv(file, has_header=False, separator=",")
                dataframes.append(raw_df)

        
        df = pl.concat(dataframes)
        data = torch.from_numpy(df.to_numpy())
        
        # init x,y
        self.x = data[:,1:].float()
        self.y = data[:,0].long()

        self.dataset_size = self.x.shape[0]


    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.dataset_size
    

def getMCI(train_path, batch_size=64, train_test_split=0.5):
    # load dataset to memory
    complete_dataset = loadMCI(batch_size=batch_size, data_path=train_path)
    # test_set = loadMCI(batch_size=batch_size, data_path=train_path)

    # split data into train/test sets
    train_size = int(complete_dataset.dataset_size * train_test_split)
    test_size = complete_dataset.dataset_size - train_size
    train_set, test_set = torch.utils.data.random_split(complete_dataset, [train_size, test_size])


    # create data loaders
    loader_train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    loader_test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    printf("- Train size: %d (%d)\n- Test size: %d (%d)\n", len(loader_train.dataset), len(loader_train.dataset) // batch_size, len(loader_test.dataset), len(loader_test.dataset) // batch_size)

    return loader_train, loader_test