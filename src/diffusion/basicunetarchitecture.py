import tensorflow as tf

class BasicUnetArchitecture():

    def __init__(self, inputShape: tuple):
        self.numFilters = [32, 64]
        self.inputShape = inputShape

    def buildModel(self, numEncoderBlocks=2, imageDim = 1):
        encoderInputs = tf.keras.layers.Input(shape=self.inputShape)
        d1 = self.generateEncoderBlock(numFilters = self.numFilters[0])(encoderInputs)
        d2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d1)
        d3 = self.generateEncoderBlock(numFilters = self.numFilters[1])(d2)
        d4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(d3)
        
        b1 = self.generateEncoderBlock(numFilters = self.numFilters[1])(d4)
        
        u4 = self.generateEncoderBlock(numFilters = self.numFilters[1])(b1)
        u5 = tf.keras.layers.UpSampling2D(size = (2,2))(u4)
        u5 = tf.keras.layers.concatenate([u5, d3])

        u6 = self.generateEncoderBlock(numFilters = self.numFilters[0])(u5)
        u7 = tf.keras.layers.UpSampling2D(size = (2,2))(u6)
        u7 = tf.keras.layers.concatenate([u7, d1])

        output = self.generateEncoderBlock(numFilters = imageDim)(u7)
        #output = tf.keras.layers.Conv2D(filters = imageDim, kernel_size = 1, padding = 'same', activation = 'silu')(u6)

        mdl = tf.keras.Model(inputs = encoderInputs, outputs = output)
        print(mdl.summary())
        return encoderInputs, output




    def generateEncoderBlock(self, numFilters: int):
        enc = tf.keras.layers.Conv2D(filters = numFilters, 
                                     kernel_size=5, 
                                     padding='same', 
                                     activation='silu')
        
        return enc
