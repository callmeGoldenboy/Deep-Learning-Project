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
from keras.layers import Dense, Conv2D, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np 

class AlexNet:
    
    def __init__(self,params):
        self.net = Sequential()
        for key,val in params:
            setattr(self,key,val)
        
    def add_layers(self):
        #first convolutional layer 
        self.net.add(Conv2D(filters=self.filters[0],input_shape=(32,32,3), kernel_size=self.kernel_sizes[0], strides=self.strides[0], padding='same'))
        if self.batch_norm:
            self.net.add(BatchNormalization())
        self.net.add(Activation('relu'))
        self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
        #convolutional layers 2-5
        for i in range(1,len(self.filters)):
            self.net.add(Conv2D(filters=self.filters[i], kernel_size=self.kernel_sizes[i], strides=self.strides[i], padding='same'))
            if self.batch_norm:
                self.net.add(BatchNormalization())
            self.net.add(Activation('relu'))
            if i == 1 or  i == 4:
                self.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))  

    def forward_pass(self):
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
        "strides": strides
        }
    model = AlexNet(params)
    
if __name__ == "__main__":
    main()