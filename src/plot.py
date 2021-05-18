import matplotlib.pyplot as plt
from keras.datasets import cifar10

def plot_dataset():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = 10
    fig = plt.figure(figsize=(8,3))
    for i in range(num_classes):
        ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
        idx = np.where(y_train[:]==i)[0]
        features_idx = x_train[idx,::]
        img_num = np.random.randint(features_idx.shape[0])
        im = np.transpose(features_idx[img_num,::],(1,2,0))
        ax.set_title(str(i))
        plt.imshow(im)
    plt.show()


def plot_loss_and_accuracy():
    plt.figure(0)
    plt.plot(cnn.history['acc'],'r')
    plt.plot(cnn.history['val_acc'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy vs Validation Accuracy")
    plt.legend(['train','validation'])


    plt.figure(1)
    plt.plot(cnn.history['loss'],'r')
    plt.plot(cnn.history['val_loss'],'g')
    plt.xticks(np.arange(0, 101, 2.0))
    plt.rcParams['figure.figsize'] = (8, 6)
    plt.xlabel("Num of Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss vs Validation Loss")
    plt.legend(['train','validation'])


    plt.show()