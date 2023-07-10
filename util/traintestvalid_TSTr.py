import torch
import numpy as np
from util.utility import printf


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

    with torch.no_grad():
        for X, y in dataloader:
            y_pred = model(X)
            test_loss += loss_fn(y_pred, y)

    test_loss /= num_batches
    printf("***Testing - Epoch %d - Loss(MSE): %.8f***\n", current_epoch, test_loss)

    return test_loss


import polars as pl
import matplotlib.pyplot as plt
from util.moving_average import movingAverage
import glob

def t_rms(x, dim=1):
    out = torch.sqrt((x**2).sum(dim=dim) / x.shape[dim])
    return out

def valid_loop(model, loss_fn, current_epoch, val_data_path, plot_result=True):

    printf("Starting validation...\n")
    model.eval()

    val_files = glob.glob(val_data_path)

    fig, ax = plt.subplots(nrows=1, ncols=len(val_files))

    for i in range(len(val_files)):
        raw_data = pl.read_csv(val_files[i], has_header=False).to_numpy()
        X = torch.from_numpy(raw_data[:,:-1]).float()
        y = torch.from_numpy(raw_data[:,-1]).float()
        # y = X[:,-1].flip(dims=(0,))
        y = y.unsqueeze(1)
        y_pred = torch.empty(y.shape)
        d_size = X.shape[0]
        test_loss = 0

        with torch.no_grad():
            # for i in range(d_size):
            y_pred = model(X.contiguous())
                
            test_loss = loss_fn(y_pred, y)

        printf("Validation - %s - Epoch %d - Loss(MSE): %.8f\n", val_files[i], current_epoch, test_loss)

        # linear fit
        y_pred_average = movingAverage(y_pred.squeeze().detach().numpy(), 25)
        # x_range = np.arange(0, y_pred.shape[0], 1)
        # l_param = np.polyfit(x_range, y_pred_average, 1)

        # rul_real = y.shape[0]
        # rul_pred = -l_param[1]/l_param[0]
        # rul_error = (rul_pred / rul_real - 1) * 100
        # printf("-- RUL --\treal: %.2f\tpred: %.2f\terror: %.2f%%\n", rul_real, rul_pred, rul_error)

        # rul factor
        label_factor = 1

        with open("results.csv", 'a') as file:
            np.savetxt("./TST_results/results_b" + str(i) + ".csv", torch.concat([y, y_pred, torch.from_numpy(y_pred_average).unsqueeze(1)], dim=1).detach().numpy(), delimiter=',')

        # plot result
        if(plot_result):
            ax[i].plot(y_pred*label_factor, label="y_pred")
            ax[i].plot(y*label_factor, label="y_real")
            ax[i].plot(y_pred_average*label_factor, label="y_pred_avg")
            # ax[i].hlines(0.5, 0, y.shape[0], colors="#ff0000", linestyles="--")
            # ax[i].plot(l_param[0] * x_range + l_param[1], color="#9300ff", label="fit")
            # ax[i].title(val_files[i])
    
    fig.set_dpi(120)
    fig.set_size_inches(w=2*len(val_files), h=2)
    # plt.legend()
    plt.show()
    # plt.waitforbuttonpress(0)
    # plt.close()