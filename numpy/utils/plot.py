import matplotlib.pyplot as plt

def plot_loss_curve(losses, save_path=None):
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    if save_path:
        plt.savefig(save_path)
        print(f"Training loss curve saved to {save_path}")
    else:
        plt.show()

def plot_accuracy_curve(accuracies, save_path=None):
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Curve')
    if save_path:
        plt.savefig(save_path)
        print(f"Training accuracy curve saved to {save_path}")
    else:
        plt.show()