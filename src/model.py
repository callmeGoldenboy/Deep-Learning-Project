"""
    README
AlexNet contains 8 layers
Input
1. Convolutional layer w. kernel size 11x11
2. Convolutional layer w. kernel size 5x5
3. Convolutional layer w. kernel size 3x3
4. Convolutional layer w. kernel size 3x3
5. Convolutional layer w. kernel size 3x3
6. Fully connected Layer 
7. Fully connected Layer
8. Fully connected Layer
"""
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, Callback, EarlyStopping
from plot import plot_loss_and_accuracy
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize
import math
import time
import numpy as np 
import pickle
import os

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class AlexNet:
    
    def __init__(self,params):
        self.net = Sequential()
        self.filters = [96,256,384,384,256]
        self.kernel_sizes = [(11,11),(5,5),(3,3),(3,3),(3,3)]
        self.strides = [(4,4),(1,1),(1,1),(1,1),(1,1)]
        for key,val in params.items():
            setattr(self,key,val)
            
        self.create_layers()
        self.net.compile(loss = categorical_crossentropy, optimizer= 'adam', metrics=['accuracy'])
        self.set_annelaer()
        
        
    def set_annelaer(self):
        if self.annealer:
            self.lrr= ReduceLROnPlateau(monitor='val_acc',factor=.01,patience=3,min_lr=1e-5)
        else:
            self.lrr = None
        
    def create_layers(self):
        #first convolutional layer 
        self.net.add(Conv2D(filters=self.filters[0],input_shape=(32,32,3), kernel_size=self.kernel_sizes[0], strides=self.strides[0], padding='same'))
        if self.batch_norm:
            self.net.add(BatchNormalization())
        self.net.add(Activation('relu'))
        self.net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        #convolutional layers 2-5
        for i in range(1,len(self.filters)):
            self.net.add(Conv2D(filters=self.filters[i], kernel_size=self.kernel_sizes[i], strides=self.strides[i], padding='same'))
            if self.batch_norm:
                self.net.add(BatchNormalization())
            self.net.add(Activation('relu'))
            if i == 1 or  i == 4:
                self.net.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))  
                
        #add fully connected layers
        self.net.add(Flatten())
        #first fully connected layer
        self.net.add(Dense(4096, input_shape=(32,32,3,)))
        if self.batch_norm:
            self.net.add(BatchNormalization())
        self.net.add(Activation('relu'))
        if self.dropout:
            self.net.add(Dropout(self.dropout_ratio))
            
        #2-3 fully connected layers
        output_spaces = [4096,1000]
        for i,dim in enumerate(output_spaces):
            self.net.add(Dense(dim))
            if self.batch_norm:
                self.net.add(BatchNormalization())
            self.net.add(Activation('relu'))
            if self.dropout and i == 0 and self.dropout_option == 2:
                print("fully connected layer 2 has dropout")
                self.net.add(Dropout(self.dropout_ratio))
                
        #output layer
        self.net.add(Dense(10))
        if self.batch_norm:
            self.net.add(BatchNormalization())
        self.net.add(Activation('softmax'))

    def forward_pass(self):
        pass
    
    
def get_datasets():
    #get train,validation and test data 
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=.3)
    y_train_one_hot = to_categorical(y_train)
    y_val_one_hot = to_categorical(y_val)
    y_test_one_hot = to_categorical(y_test)
    return x_train,x_val,x_test,y_train_one_hot,y_val_one_hot,y_test_one_hot
    
def augment():
    #Milestone 4?
    pass

def init_model(params={}):
    new_params = {
        "batch_norm":False,
        "data_augmentation": False,
        "dropout": False,
        "annealer": False,
        "batch_size": 100,
        "epoch":5,
        "learn_rate":.001
    } 
    for k,v in params.items():
        new_params[k] = v 

    print("Model initated w. params:", new_params)
    model = AlexNet(new_params)

    return model

def milestone1(epoch=5,load_model=None):
    params = {"epoch":epoch}
    model = init_model(params=params)
    x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()
    if not load_model:
        time_callback = TimeHistory()
        early_stopping_monitor = EarlyStopping(restore_best_weights=True, patience=epoch)
        train_result = model.net.fit(x=x_train, y=y_train, batch_size=model.batch_size, 
                                epochs = model.epoch, steps_per_epoch = x_train.shape[0]//model.batch_size,
                                validation_data = (x_val, y_val), validation_steps = x_val.shape[0]//model.batch_size, callbacks = [e for e in [model.lrr, time_callback,early_stopping_monitor] if e], verbose=1)
            
        pickle_dump('dumps/milestone1',train_result,time_callback.times, model=model)
    else:
        model = load_model(load_model)
        loss,acc = model.evaluate(x_test,y_test,batch_size=100)
        print(f"Milestone 1: accuracy: {acc}, loss: {loss}")
        

def milestone2(epoch=5,load_model=None):
    #Investigate the effects of batch normalization
    params = { "batch_norm":True, "epoch":epoch}
    model = init_model(params=params)
    x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()
    if not load_model:
        time_callback = TimeHistory()
        early_stopping_monitor = EarlyStopping(restore_best_weights=True, patience=epoch)
        train_result = model.net.fit(x=x_train, y=y_train, batch_size=model.batch_size, 
                                epochs = model.epoch, steps_per_epoch = x_train.shape[0]//model.batch_size,
                                validation_data = (x_val, y_val), validation_steps = x_val.shape[0]//model.batch_size, callbacks = [e for e in [model.lrr, time_callback, early_stopping_monitor] if e], verbose=1)
        
        pickle_dump('dumps/milestone2',train_result,time_callback.times, model=model)
    else:
        model = load_model(load_model)
        loss,acc = model.evaluate(x_test,y_test,batch_size=100)
        print(f"Milestone 2: accuracy: {acc}, loss: {loss}")


def milestone3(epoch=5, load_model=None):
    if not load_model:
        #Investigate the effects of applying different degrees of dropout
        params =  {"epoch":epoch}
        x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()
        dropout_options = [1,2]
        dropout_ratios = [0.3,0.4,0.5,0.6]
        for option in dropout_options:
            params["options"] = option
            for ratio in dropout_ratios:
                time_callback = TimeHistory()   
                early_stopping_monitor = EarlyStopping(restore_best_weights=True, patience=epoch)
                params["dropout_ratio"] = ratio
                model = init_model(params=params)
                train_results = model.net.fit(x=x_train, y=y_train, batch_size=model.batch_size, 
                                epochs = model.epoch, steps_per_epoch = x_train.shape[0]//model.batch_size,
                                validation_data = (x_val, y_val), validation_steps = x_val.shape[0]//model.batch_size, callbacks = [e for e in [model.lrr, time_callback, early_stopping_monitor] if e], verbose=1)
                pickle_dump("dumps/"+("mileston3-opt-"+str(option)+"-ratio-"+str(ratio)), train_results, time_callback.times,model=model)
    else:
        model = load_model(load_model)
        loss,acc = model.evaluate(x_test,y_test,batch_size=100)
        print(f"Milestone 3: accuracy: {acc}, loss: {loss}")


def random_crop(image, crop_size=(31,31)):
    # crop_size=(16,16) is too small since the native res is (32,32)
    #(24,24) gave small acc too
    #(28,28) also gave small acc
    #(32,32) gave an accof 42% after 5 epochs on the
    height, width = image.shape[:2] # get original shape
    rand_arr = np.random.random(size=2) # get two rands
    x, y = (math.floor((height - crop_size[0]) * rand_arr[0]), math.floor((width - crop_size[1]) * rand_arr[1]))
    image_crop = image[x:x+crop_size[0], y:y+crop_size[1], 0:3]
    image_crop = resize(image_crop, image.shape)
    return image_crop

def milestone4(epoch=5, load_model=None):
    x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()

    if not load_model:
        #investigate the effects of data augmentation
        params =  {"epoch":epoch, "data_augmentation":True}
        time_callback = TimeHistory()

        generators = {
            "rot": ImageDataGenerator(rescale=1/255, rotation_range=45),
            "flip": ImageDataGenerator(rescale=1/255, horizontal_flip=True, vertical_flip=True),
            #"crop": ImageDataGenerator(rescale=1/255, preprocessing_function=random_crop),
        }

        val_gen = ImageDataGenerator(rescale=1/255)
        val_gen.fit(x_val)
        iterator_val = val_gen.flow(x_val, y_val, batch_size=100)
        
        for k,v in generators.items():
            train_gen = v
            train_gen.fit(x_train)
            early_stopping_monitor = EarlyStopping(restore_best_weights=True, patience=epoch)
            model = init_model(params=params)
            iterator_train = train_gen.flow(x_train ,y_train, batch_size=model.batch_size)
            train_result = model.net.fit(iterator_train, batch_size=model.batch_size, 
                                    epochs = model.epoch, steps_per_epoch = x_train.shape[0]//model.batch_size,
                                    validation_data = iterator_val, validation_steps = x_val.shape[0]//model.batch_size,
                                    callbacks = [e for e in [model.lrr, time_callback, early_stopping_monitor] if e], verbose=1)
            pickle_dump('dumps/milestone4'+"-"+k,train_result,time_callback.times, model=model)
    else:
        test_gen = ImageDataGenerator(rescale=1/255)
        test_gen.fit(x_test)
        iterator_test = test_gen.flow(x_test ,y_test, batch_size=100)
        model = load_model(load_model)
        loss,acc = model.evaluate(iterator_test,batch_size=100)
        print(f"Milestone 4: accuracy: {acc}, loss: {loss}")
    


def milestone5(epoch=5):
    #test the model on the ciphar100
    pass
    
def pickle_dump(path,res,times,model=None):
    if model:
        model.net.save(path+"-model")
    res.history['times'] = times
    with open(path, 'wb') as file_pi:
        pickle.dump(res.history, file_pi)

def pickle_load(path):
    return pickle.load(open(path,"rb"))

def load_model(path):
    assert os.path.exists(path + "-model"), "Could not find " + path + "-model"
    return keras.models.load_model(path+"-model")

if __name__ == "__main__":
    milestone4(epoch=100)