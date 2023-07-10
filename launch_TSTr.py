from data_import.data_loader import getPRONOSTIA
from util.utility import load_model, save_model, printf
from models.TSTr import STFTTr, ModelConfig
from util.traintestvalid_TSTr import train_loop, test_loop, valid_loop
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------------------------------------------------
# set seed
torch.manual_seed(42069)
is_load_model = False
is_save_model = False
plot_validation_result = True
model_path = "model.pt"
dataset_path = "./datasets/FEMTO_XJTU/"

bearing_id = "b1" # BIAS IN EMBEDDING IS TURNED OFF
convergence_limit = 3


# init model, optimizer, loss
model_config = ModelConfig()

if is_load_model:
    epoch, model, optimizer, loss = load_model(model_path, STFTTr, model_config)
else:
    model = STFTTr(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-2)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
# lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer)

loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.TripletMarginLoss()

# model size
print("num parameters:", sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


# load dataset
train_dir = dataset_path + bearing_id + "/train/*.csv"
test_dir = dataset_path + bearing_id + "/test/*.csv"
loader_train, loader_test = getPRONOSTIA(train_path=train_dir, test_path=test_dir, batch_size=model_config.batch_size, train_test_split=0.5)
printf("Data loaded!\n")


# ---------------------------------------------------------------------------------------------------------------------
# train model
best_loss = 1e6
best_epoch = 0
convergence_timeout = 0

if(not is_load_model):
    num_epochs = 100
    printf("--- Training: %d epochs, learning rate: %.2e!\n", num_epochs, lr_scheduler.get_last_lr()[-1])
    for epoch in range(num_epochs):
        # train
        train_loop(dataloader=loader_train, model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler, current_epoch=epoch+1, log=True)
        
        # test
        test_loss = test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn, current_epoch=epoch+1)
        if test_loss < best_loss:
            best_loss = test_loss.item()
            best_epoch = epoch + 1
            convergence_timeout = 0
            print("new best!")
            save_model(epoch, model, optimizer, loss_fn, model_path)

        else:
            convergence_timeout += 1
            print("timeout:", convergence_timeout)
            if convergence_timeout == convergence_limit:
                printf("best loss: %.5f\n", best_loss)
                print("best epoch:", best_epoch)
                break
        
        # change learning rate
        lr_scheduler.step()
        printf("--- Learning Rate: %.2e\n", lr_scheduler.get_last_lr()[0])

        # validate
        if(((epoch+1) % 1) == 0):
            # test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn, current_epoch=epoch+1)
            val_data_path = "./data/" + bearing_id + "/val/*.csv"
            valid_loop(model=model, loss_fn=loss_fn, current_epoch=epoch+1, val_data_path=val_data_path, plot_result=plot_validation_result)


        # save model
        if(is_save_model):
            save_model(epoch, model, optimizer, loss_fn, model_path)

# ---------------------------------------------------------------------------------------------------------------------

epoch, model, optimizer, loss = load_model(model_path, STFTTr, model_config)
# test_loop(dataloader=loader_test, model=model, loss_fn=loss_fn, current_epoch=epoch+1)
val_data_path = "./data/" + bearing_id + "/val/*.csv"
valid_loop(model=model, loss_fn=loss_fn, current_epoch=epoch+1, val_data_path=val_data_path)