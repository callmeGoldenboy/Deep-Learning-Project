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
    #plt.figure(0)
    if milestone != None:
        cnn = pickle.load(open("dumps/" + milestone, "rb"))
    plt.figure(figsize=(20,20))
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
    #ax1.set_autoscale_on(False)
    ax1.plot(x,train_acc,'r')
    ax1.plot(x,val_acc,'g')
    ax1.set_xticks(xticks)
    ax1.set_yticks(yticks_acc)
    #ax1.set_xlabel("Num of Epochs")
    ax1.set_ylabel("Accuracy")
    #ax1.set_title("Accuracy")
    ax1.legend(['train','validation'])
    #loss plots 
    
    #ax2.set_autoscale_on(False)
    ax2.plot(x,train_loss,'r')
    ax2.plot(x,val_loss,'g')
    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks_loss)
    ax2.set_xlabel("Num of Epochs")
    ax2.set_ylabel("Loss")
    #ax2.set_title("Loss")
    ax2.legend(['train','validation'])
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle("Accuracy and Loss")
    #plt.grid()
    ax1.grid()
    ax2.grid()
    if save:
        plt.savefig("src/results/"+name)
    else:
        plt.show()

def get_datasets():
    #get train,validation and test data 
    from keras.datasets import cifar10
    from sklearn.model_selection import train_test_split
    from sklearn.utils.multiclass import unique_labels
    from tensorflow.keras.utils import to_categorical


    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=.3)
    y_train_one_hot = to_categorical(y_train)
    y_val_one_hot = to_categorical(y_val)
    y_test_one_hot = to_categorical(y_test)
    return x_train,x_val,x_test,y_train_one_hot,y_val_one_hot,y_test_one_hot

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

def random_crop(image, crop_size=(24,24)):
    from skimage.transform import resize
    height, width = image.shape[:2] # get original shape
    rand_arr = np.random.random(size=2) # get two rands
    x, y = (math.floor((height - crop_size[0]) * rand_arr[0]), math.floor((width - crop_size[1]) * rand_arr[1]))
    image_crop = image[x:x+crop_size[0], y:y+crop_size[1], 0:3]
    image_crop = resize(image_crop, image.shape)
    return image_crop


def tmp():
     #investigate the effects of data augmentation
    from keras.preprocessing.image import ImageDataGenerator
    x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()
    
    #rotation
    generators = {
            "rot": {"train_gen": ImageDataGenerator(rescale=1/255, preprocessing_function=random_crop),
            "val_gen": ImageDataGenerator(rescale=1/255), 
                },
    }
    for k,v in generators.items():
        fig,(ax1,ax2) = plt.subplots(1,2)
        train_gen = v["train_gen"]
        train_gen.fit(x_train[0:1])

        iterator_train = train_gen.flow(x_train[0:1] ,y_train[0:1], batch_size=1)
        x = iterator_train.next()
        image = x[0][0]
        print(image.shape)
        ax1.imshow(x_train[0])
        ax2.imshow(image)
        ax1.set_title("Before")
        ax2.set_title("After")
        plt.savefig("croppingtest")
    
#tmp()
#plot_loss_and_accuracy(milestone="milestone4-flip-vertical", name="milestone4-flip-vertical", save=True)

#plot_dataset()
#history = pickle.load(open('dumps/milestone1',"rb"))
#print(history['loss'])
#plot_loss_and_accuracy(history)

#[plot_loss_and_accuracy(milestone=f_n, save=True, name=f_n+".png") for f_n in os.listdir("dumps") if ('milestone' in f_n and not os.path.isdir(os.path.join("dumps",f_n)))]
box_plot_time()
