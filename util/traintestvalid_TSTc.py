import torch
import numpy as np
from util.utility import printf
from sklearn.metrics import confusion_matrix
import numpy as np


def confMatrix(y_t, y_p):
    cf_matrix = confusion_matrix(y_t, y_p)
    cf_matrix = np.round(np.divide(cf_matrix.T, np.sum(cf_matrix, axis=1)).T * 100, 2)
    print(cf_matrix)
    return cf_matrix

def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, current_epoch, log=True):

    model.train()

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):
        # prediction
        y_pred = model(X)

        # compute loss
        loss = loss_fn(y_pred, y)
        # loss = loss_fn(y_pred.squeeze(), X[:,:-1])
        
        # gradients
        optimizer.zero_grad()
        loss.backward()
        
        # gradient descent
        optimizer.step()

        # log
        if (batch+1) % 10 == 0 and log:
            printf("Epoch(%d, %d/%d) --- Loss(MSE): %.8f\n", current_epoch, batch+1, int(size/dataloader.batch_size), loss.item())

        # if (batch+1) % 100 == 0 and batch > 0:
        #     lr_scheduler.step()
        #     printf("--- Learning Rate: %.2e\n", lr_scheduler.get_last_lr()[0])


def test_loop(dataloader, model, loss_fn, current_epoch):

    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    accuracy_counter = 0

    y_t = []
    y_p = []

    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)
            y_pred = torch.argmax(y_pred, dim=1)

            for i in range(y_pred.shape[0]):
                if(y_pred[i] == y[i]):
                    accuracy_counter += 1
                
                y_t.append(y[i])
                y_p.append(y_pred[i])

    test_accuracy = accuracy_counter / size * 100
    test_loss /= num_batches

    printf("***Testing - Epoch %d - Accuracy: %.2f %% - Loss(MSE): %.8f***\n", current_epoch, test_accuracy, test_loss)
    
    cm = confMatrix(y_t, y_p)

    return test_accuracy, test_loss, cm


import polars as pl
import matplotlib.pyplot as plt
from util.moving_average import movingAverage
import glob

def t_rms(x, dim=1):
    out = torch.sqrt((x**2).sum(dim=dim) / x.shape[dim])
    return out

def valid_loop(model, loss_fn, current_epoch, val_data_path):

    printf("Starting validation...\n")
    model.eval()

    total_accuracy = 0
    total_loss = 0
    y_t = []
    y_p = []

    val_files = glob.glob(val_data_path)

    for file in val_files:
        data = pl.read_csv(file, has_header=False).to_numpy()
        X = torch.from_numpy(data[:,1:]).float()
        y = torch.from_numpy(data[:,0]).long()
        # y = X[:,-1].flip(dims=(0,))
        y_pred = torch.empty(y.shape)
        d_size = X.shape[0]

        with torch.no_grad():
            # for i in range(d_size):
            y_pred = model(X.contiguous())
                
            valid_loss = loss_fn(y_pred, y)

            y_pred = torch.argmax(y_pred, dim=1)

            accuracy_counter = 0
            for i in range(y_pred.shape[0]):
                if(y_pred[i] == y[i]):
                    accuracy_counter += 1

                y_t.append(y[i])
                y_p.append(y_pred[i])

            valid_accuracy = accuracy_counter / y_pred.shape[0] * 100
            total_accuracy += valid_accuracy
            total_loss += valid_loss

        printf("Validation - %s - Epoch %d - Accuracy: %.2f %% - Loss(MSE): %.8f\n", file, current_epoch, valid_accuracy, valid_loss)

    total_loss /= len(val_files)
    total_accuracy /= len(val_files)

    cm = confMatrix(y_t, y_p)

    printf("Total validation accuracy: %.2f %%\n", total_accuracy)
    return total_accuracy, total_loss, cm