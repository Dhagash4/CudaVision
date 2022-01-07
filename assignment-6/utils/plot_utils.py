import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import seaborn as sns


def smooth(f, K=5):
    """ Smoothing a function using a low-pass filter (mean) of size K """
    kernel = np.ones(K) / K
    f = np.concatenate([f[:int(K//2)], f, f[int(-K//2):]]
                       )  # to account for boundaries
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[K//2: -K//2]  # removing boundary-fixes
    return smooth_f


def plot_stats(train_loss, val_loss, loss_iters, valid_acc):

    plt.style.use('seaborn')

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(24, 5)

    smooth_loss = smooth(loss_iters, 31)
    ax[0].plot(loss_iters, c="blue", label="Loss", linewidth=3, alpha=0.5)
    ax[0].plot(smooth_loss, c="red",
               label="Smoothed Loss", linewidth=3, alpha=1)
    ax[0].legend(loc="best")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("CE Loss")
    ax[0].set_title("Training Progress")

    epochs = np.arange(len(train_loss)) + 1
    ax[1].plot(epochs, train_loss, c="red",
               label="Train Loss", linewidth=3)
    ax[1].plot(epochs, val_loss, c="blue", label="Valid Loss", linewidth=3)
    ax[1].legend(loc="best")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("CE Loss")
    ax[1].set_title("Loss Curves")

    epochs = np.arange(len(val_loss)) + 1

    ax[2].plot(epochs, valid_acc, c="red", label="Valid accuracy", linewidth=3)
    ax[2].legend(loc="best")
    ax[2].set_xlabel("Epochs")
    ax[2].set_ylabel("Accuracy (%)")
    ax[2].set_title(
        f"Valdiation Accuracy (max={round(np.max(valid_acc),2)}% @ epoch {np.argmax(valid_acc)+1})")

    plt.show()

    return


def plot_confusion_matrix(cm, label_dict, title='Confusion matrix', size=(15, 8), cmap=plt.cm.Blues):
    (h, w) = size
    ax = plt.subplot()

    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(cm, annot=True, fmt='g', ax=ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    x_axis = [v for v in label_dict.values()]
    y_axis = [v for v in label_dict.values()]
    ax.set_xticks([k for k in label_dict.keys()])
    ax.set_yticks([k for k in label_dict.keys()])
    ax.set_xticklabels(x_axis, rotation=90)
    ax.set_yticklabels(y_axis, rotation=0)

    plt.gcf().set_size_inches(h, w)

    plt.show()

    return
