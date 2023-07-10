from data_import.mci_loader import getMCI
from util.utility import load_model, save_model, printf, test_val_result_plot
from models.TSTc import TSTc, ModelConfig
from util.traintestvalid_TSTc import train_loop, test_loop, valid_loop
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------------------------------------------------------------------------------------------------
# set seed
torch.manual_seed(42069)
is_load_model = 0
is_save_model = 1
plot_validation_result = True
model_path = "model.pt"
dataset_path = "./datasets/MCI/lab/"
# dataset_path = "./datasets/CWRU/"
val_data_path=dataset_path + "/valid/*.csv"

convergence_limit = 3

# init model, optimizer, loss
model_config = ModelConfig()

if is_load_model:
    epoch, model, optimizer, loss = load_model(model_path, TSTc, model_config)
else:
    model = TSTc(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=0.001)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)

loss_fn = torch.nn.CrossEntropyLoss()

# model size
print("num parameters:", sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# load dataset
loader_train, loader_test = getMCI(train_path=dataset_path, batch_size=model_config.batch_size, train_test_split=0.5)
printf("Data loaded!\n")


# ---------------------------------------------------------------------------------------------------------------------
# train model

best_accuracy = 0
best_val_accuracy = 0
best_cm = None
best_valid_cm = None
best_epoch = 0
best_valid_epoch = 0
num_epochs = 100

test_loss_array = []
test_accuracy_array = []
val_loss_array = []
val_accuracy_array = []

# initial test
test_accuracy, test_loss, _ = test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn, current_epoch=0)
val_accuracy, val_loss, _ = valid_loop(model=model, loss_fn=loss_fn, current_epoch=0, val_data_path=val_data_path)

# log test results
test_loss_array.append(test_loss)
test_accuracy_array.append(test_accuracy/100)
val_loss_array.append(val_loss)
val_accuracy_array.append(val_accuracy/100)

test_val_result_plot(test_loss_array, test_accuracy_array, val_loss_array, val_accuracy_array, num_epochs)


printf("--- Training: %d epochs, learning rate: %.2e!\n", num_epochs, lr_scheduler.get_last_lr()[-1])
for epoch in range(num_epochs):
    # train
    train_loop(dataloader=loader_train, model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, current_epoch=epoch+1, log=True)
    
    # test
    test_accuracy, test_loss, cm = test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn, current_epoch=epoch+1)

    # log test results
    test_loss_array.append(test_loss)
    test_accuracy_array.append(test_accuracy/100)

    # update best epoch
    if(test_accuracy > best_accuracy):
        best_accuracy = test_accuracy
        best_cm = cm
        best_epoch = epoch + 1

    #     convergence_timeout = 0
    #     print("new best!")
    #     save_model(epoch, model, optimizer, loss_fn, model_path)

    # else:
    #     convergence_timeout += 1
    #     print("timeout:", convergence_timeout)
    #     if convergence_timeout == convergence_limit:
    #         break

    # validate
    val_accuracy, val_loss, cm = valid_loop(model=model, loss_fn=loss_fn, current_epoch=epoch+1, val_data_path=val_data_path)

    # log validation results
    val_loss_array.append(val_loss)
    val_accuracy_array.append(val_accuracy/100)

    if(val_accuracy > best_val_accuracy):
        best_val_accuracy = val_accuracy
        best_valid_cm = cm
        best_valid_epoch = epoch + 1
        
        convergence_timeout = 0
        print("new best!")
        save_model(epoch, model, optimizer, loss_fn, model_path)

    else:
        convergence_timeout += 1
        print("timeout:", convergence_timeout)
        if convergence_timeout == convergence_limit:
            break

    # plot results
    test_val_result_plot(test_loss_array, test_accuracy_array, val_loss_array, val_accuracy_array, num_epochs)

    # change learning rate
    lr_scheduler.step()
    printf("--- Learning Rate: %.2e\n", lr_scheduler.get_last_lr()[0])

    # # save model
    # if(is_save_model):
    #     save_model(epoch, model, optimizer, loss_fn, model_path)

printf("Best test accuracy(%d): %.2f %%\n", best_epoch, best_accuracy)
print(best_cm)
printf("Best valid accuracy(%d): %.2f %%\n", best_valid_epoch, best_val_accuracy)
print(best_valid_cm)

# ---------------------------------------------------------------------------------------------------------------------
plt.show()