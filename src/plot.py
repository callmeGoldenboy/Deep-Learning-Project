import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np
import seaborn as sns
import pickle
import os
import math

sns.set()

def plot_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        features_idx = x_train[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = np.transpose(features_idx[img_num,::],(0,1,2))
        ax.set_title(str(i))
        plt.imshow(im)
    plt.show()

def plot_loss_and_accuracy(cnn=None, name="figure", milestone=None, save=False):
    """Plot the loss and accuracy of the model

    Args:
        cnn (Sequential.history, optional): A dict containing the data to plot. Defaults to None.
        name (str, optional): The name of the plot to be saved. Defaults to "figure".
        milestone (string, optional): If this argument is given, the history will be loaded from the files. Defaults to None.
        save (bool, optional): Whether to save the figure or just show it. Defaults to False.
    """
    if milestone != None:
        cnn = pickle.load(open("dumps/" + milestone, "rb"))
    train_acc = cnn["accuracy"]
    train_loss = cnn["loss"]
    val_acc = cnn["val_accuracy"]
    val_loss = cnn["val_loss"]
    n = len(cnn["accuracy"])
    x = list(range(1,len(cnn["accuracy"]) + 1))
    xticks = range(1,n + 1, math.ceil((round(n / 9))))
    yticks_acc = [round(i*0.1,1) for i in range(0,11)]
    yticks_loss = [round(i*0.5,1) for i in range(0,9)]
    fig,(ax1,ax2) = plt.subplots(2,1)
    ax1.plot(x,train_acc,'r')
    ax1.plot(x,val_acc,'g')
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks_acc)
    ax1.set_ylabel("Accuracy")
    ax1.legend(['train','validation'])
    ax2.plot(x,train_loss,'r')
    ax2.plot(x,val_loss,'g')
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks_loss)
    ax2.set_xlabel("Num of Epochs")
    ax2.set_ylabel("Loss")
    ax2.legend(['train','validation'])
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Accuracy and Loss")
    ax1.grid()
    ax2.grid()
    if save:
        plt.savefig("src/results/"+name)
    else:
        plt.show()

def box_plot_time():
    names = [f_n for f_n in os.listdir("dumps") if ('m' in f_n and not os.path.isdir(os.path.join("dumps",f_n)))]
    cnn_times = [(pickle.load(open("dumps/" + f_n, "rb")))['times'][1:] for f_n in names]
    pairs = [(names[i], cnn_times[i]) for i in range(0,len(names))]
    
    m_3 = [p[1] for p in pairs if 'mileston3' in p[0]]
    pairs = [p for p in pairs if 'mileston3' not in p[0]]
    m_3 = np.array(m_3).ravel()
    pairs.append(('m3-all',m_3))

    pairs.sort(key= (lambda p: np.median(p[1])) )

    names = [ p[0] for p in pairs]
    cnn_times = [ p[1] for p in pairs ]

    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].set_ylabel("execution time (s)")

    axes[0].boxplot(cnn_times[0:7],showfliers=False)
    axes[0].set_xticks(np.arange(1,8))
    axes[0].set_xticklabels(names[0:7],rotation=75, ha="center")

    axes[1].boxplot(cnn_times[7:12],showfliers=False)
    axes[1].set_xticks(np.arange(1,6))
    axes[1].set_xticklabels(names[7:12],rotation=75, ha="center")
    
    axes[2].boxplot(cnn_times[12:],showfliers=False)
    axes[2].set_xticks(np.arange(1,6))
    axes[2].set_xticklabels(names[12:],rotation=75, ha="center")
    
    fig.tight_layout()

    
    
    #plt.suptitle("Execution time (s)\n")
    #plt.figure(figsize=(20,10))
    #plt.yscale('log')
    #plt.boxplot(cnn_times,showfliers=False)
    #plt.xticks(np.arange(0,len(names)),names,rotation=45)
    plt.savefig("src/results/boxplot")


#tmp()
plot_loss_and_accuracy(milestone="mileston3-opt-2-ratio-0.4", name="milestone4-flip-vertical", save=False)
#print(history['loss'])
#plot_loss_and_accuracy(history)

#[plot_loss_and_accuracy(milestone=f_n, save=True, name=f_n+".png") for f_n in os.listdir("dumps") if ('milestone' in f_n and not os.path.isdir(os.path.join("dumps",f_n)))]
#box_plot_time()
