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
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D,Flatten,Dropout
from keras.layers.normalization import BatchNormalization
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau
from plot import plot_loss_and_accuracy
import numpy as np 

class AlexNet:
    
    def __init__(self,params):
        self.net = Sequential()
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
            self.net.add(Dropout(0.4))
            
        #2-3 fully connected layers
        output_spaces = [4096,1000]
        for dim in output_spaces:
            self.net.add(Dense(dim))
            if self.batch_norm:
                self.net.add(BatchNormalization())
            self.net.add(Activation('relu'))
            if self.dropout:
                self.net.add(Dropout(0.4))
                
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


def main():
    filters = [96,256,384,384,256]
    kernel_sizes = [(11,11),(5,5),(3,3),(3,3),(3,3)]
    strides = [(4,4),(1,1),(1,1),(1,1),(1,1)]
    params = {
        "batch_norm":False,
        "data_augmentation": False,
        "dropout": False,
        "filters": filters,
        "kernel_sizes": kernel_sizes,
        "strides": strides,
        "annealer": False,
        "batch_size": 100,
        "epochs":1,
        "learn_rate":.001
        }
    x_train,x_val,x_test,y_train,y_val,y_test = get_datasets()
    model = AlexNet(params)
    model.net.fit(x=x_train, y=y_train, batch_size=model.batch_size, 
                            epochs = model.epochs, steps_per_epoch = x_train.shape[0]//model.batch_size,
                            validation_data = (x_val, y_val), validation_steps = 250, callbacks = model.lrr, verbose=1)
    
    plot_loss_and_accuracy(model.net)
    
if __name__ == "__main__":
    main()