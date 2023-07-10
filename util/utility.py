import torch
import matplotlib.pyplot as plt

def save_model(epoch, model, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, path)
    printf("Model saved to %s!\n", path)
    
def load_model(path, model_class, model_config):
    model = model_class(model_config)
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print("loaded:", path)

    return epoch, model, optimizer, loss

import sys
def printf(format, *args):
    sys.stdout.write(format % args)

def test_val_result_plot(test_loss_array, test_accuracy_array, val_loss_array, val_accuracy_array, num_epochs):
    return
    plt.clf()
    plt.plot(test_loss_array, color="red", linestyle="--", label="Test Loss")
    plt.plot(test_accuracy_array, color="red", label="Test Accuracy")
    plt.plot(val_loss_array, color="green", linestyle="--", label="Validation Loss")
    plt.plot(val_accuracy_array, color="green", label="Validation Accuracy")
    plt.grid()
    plt.hlines([i/10 for i in range(0, 11, 1)], 0, num_epochs, colors="#b2b2b2")
    plt.legend()
    plt.draw()
    plt.pause(0.001)
