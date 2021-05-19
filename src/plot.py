import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np

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


def plot_loss_and_accuracy(cnn,name="figure"):
    #plt.figure(0)
    plt.figure(figsize=(20,20))
    train_acc = cnn.history["accuracy"]
    train_loss = cnn.history["loss"]
    val_acc = cnn.history["val_accuracy"]
    val_loss = cnn.history["val_loss"]
    x = list(range(1,len(cnn.history["accuracy"]) + 1))
    xticks = list(range(1,int(len(cnn.history["accuracy"])) + 1,int(len(cnn.history["accuracy"]) / 5)))
    print(x)
    fig,(ax1,ax2) = plt.subplots(1,2)
    ax1.plot(x,train_acc,'r')
    ax1.plot(x,val_acc,'g')
    ax1.set_xticks(xticks)
    ax1.set_xlabel("Num of Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.legend(['train','validation'])
    #loss plots 
    ax2.plot(x,train_loss,'r')
    ax2.plot(x,val_loss,'g')
    ax2.set_xticks(xticks)
    ax2.set_xlabel("Num of Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.legend(['train','validation'])
    fig.subplots_adjust(wspace=0.5)
    #fig.suptitle("Accuracy and Loss for Training and Validation")
    
   
    plt.savefig(name)


#plot_dataset()