'''
Load and prep mnist data set
'''
import numpy as np
import tensorflow

class ImageLoader():
    def __init__(self):
        pass

    def loadMnistData(self):
        (x_train, self.y_train), (x_test, self.y_test) = tensorflow.keras.datasets.mnist.load_data()
        self.x_train = self._normalizeData(data=x_train)
        self.x_test = self._normalizeData(data=x_test)

    def loadFashionMnistData(self):
        (x_train, self.y_train), (x_test, self.y_test) = tensorflow.keras.datasets.fashion_mnist.load_data()
        self.x_train = self._normalizeData(data=x_train)
        self.x_test = self._normalizeData(data=x_test)

    def loadCifarTenData(self):
        (x_train, self.y_train), (x_test, self.y_test) = tensorflow.keras.datasets.cifar10.load_data()
        #x_train = tensorflow.image.resize(x_train, (28, 28))
        self.x_train = self._normalizeData(data=x_train)
        self.x_test = self._normalizeData(data=x_test)
        print('shape')
        print(len(self.x_train.shape))
        #self.x_train =x_train/255.

    def _normalizeData(self, data: np.ndarray):
        data = data/255.
        data = np.expand_dims(data, axis = 3) if len(data.shape)==3 else data
        return data
    
    def getTrainData(self):
        return self.x_train, self.y_train
    
    def getTestData(self):
        return self.x_test, self.y_test
