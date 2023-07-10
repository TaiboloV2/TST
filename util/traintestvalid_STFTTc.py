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
        if (batch+1) % 1 == 0 and log:
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
from HI.moving_average import movingAverage
import glob

def t_rms(x, dim=1):
    out = torch.sqrt((x**2).sum(dim=dim) / x.shape[dim])
    return out

def valid_loop(model, loss_fn, current_epoch, val_data_path, plot_result=True):

    printf("Starting validation...\n")
    model.eval()

    val_files = glob.glob(val_data_path)

    fig, ax = plt.subplots(nrows=1, ncols=len(val_files))
    fpt_list = []

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
                
            test_loss = loss_fn(y_pred.max(dim=1).values, y.squeeze())

        printf("Validation - %s - Epoch %d - Loss(MSE): %.8f\n", val_files[i], current_epoch, test_loss)

        # # linear fit
        # y_pred_average = movingAverage(y_pred.squeeze().detach().numpy(), 25)
        # x_range = np.arange(0, y_pred.shape[0], 1)
        # l_param = np.polyfit(x_range, y_pred_average, 1)

        # rul_real = y.shape[0]
        # rul_pred = -l_param[1]/l_param[0]
        # rul_error = (rul_pred / rul_real - 1) * 100
        # printf("-- RUL --\treal: %.2f\tpred: %.2f\terror: %.2f%%\n", rul_real, rul_pred, rul_error)

        # plot result
        if(plot_result):

            rmse = torch.sqrt(torch.sum(X[:,:-1]**2, dim=1) / X.shape[0])
            ax[i].plot(rmse, label="RMSE")

            pred_labels = torch.argmax(y_pred, dim=1)
            pred_labels_indeces = torch.where(pred_labels > 0)[0]
            # print(pred_labels_indeces)
            y = torch.where(pred_labels_indeces.diff() == 1)[0]
            if(y.shape[0] > 0):
                # fpt = pred_labels_indeces[int(y[0])]
                fpt = pred_labels_indeces[0]
                fpt_list.append(fpt.item())

            if(i == 2):
                for u in range(2334):
                    with open("fpt0.csv", 'a') as file0:
                        with open("fpt1.csv", 'a') as file1:
                            if(pred_labels[u].item() == 0):
                                file0.write(str(u) + " " + str(0) + "\n")
                            else:
                                file1.write(str(u) + " " + str(0) + "\n")
                print(u)

            ax[i].scatter(torch.arange(0, y_pred.shape[0], 1), pred_labels, c=torch.argmax(y_pred, dim=1), label="y_pred")
            ax[i].plot(y_pred[:,1])
            ax[i].hlines(0.5, 0, y_pred.shape[0], colors="#00ff00", linestyles="--")
            ax[i].vlines(fpt, 0, 1, colors="#ff0000", linestyles="--")
            # ax[i].plot(y_pred_average, label="y_pred_avg")
            # ax[i].hlines(0.5, 0, y.shape[0], colors="#ff0000", linestyles="--")
            # ax[i].plot(l_param[0] * x_range + l_param[1], color="#9300ff", label="fit")
            # ax[i].title(val_files[i])

    print("FPTs:", fpt_list)
    
    fig.set_dpi(120)
    fig.set_size_inches(w=2*len(val_files), h=2)
    # plt.legend()
    plt.show()
    # plt.waitforbuttonpress(0)
    # plt.close()