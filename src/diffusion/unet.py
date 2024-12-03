import tensorflow as tf


class CustomUnet():

    def __init__(self, inputShape: tuple):
        self.inputShape = inputShape

    def encoderBlock(self, numFilters: int, layerInputs):
        #each encoder block  has 2 conv layer with the same number of filters followed by max pool
        enc = tf.keras.layers.Conv2D(filters = numFilters, 
                                     kernel_size=3, 
                                     padding='same', 
                                     activation='relu')(layerInputs)
        enc = tf.keras.layers.Conv2D(filters = numFilters, 
                                     kernel_size=3, 
                                     padding='same', 
                                     activation='relu')(enc)

        return enc

    def decoderBlock(self, numFilters: int, layerInputs, skipConnections):
        #the decoder block will upsample and add skip connections
        #skipConnection is the layer from the encoder that should be connected to this decoder block
        dec = tf.keras.layers.Conv2DTranspose(filters = numFilters, 
                                              kernel_size = 2,
                                              padding='same',
                                              strides=(2,2))(layerInputs)
        #resize the images coming out of skip connection to the same as output of dec

        #skipConnections = tf.keras.layers.Cropping2D((dec.shape[1], dec.shape[2]))(skipConnections)
        #print(skipConnections.shape)
        #skipConnections = tf.image.resize(skipConnections, 
        #                                  size=(tf.Tensor(shape=(dec.shape[0], 
         #                                                        dec.shape[1], 
         #                                                        dec.shape[2], 
         #                                                        dec.shape[3]))))
        #concat the skip connection
        dec = tf.keras.layers.Concatenate()([dec, skipConnections])

        dec = tf.keras.layers.Conv2D(filters = numFilters, 
                                              kernel_size = 3,
                                              padding='same')(dec)
        dec = tf.keras.layers.Conv2D(filters = numFilters, 
                                              kernel_size = 3,
                                              padding='same')(dec)
        
        return dec

        

    def buildModel(self, numEncoderBlocks: int, imageDim: int):
        
        encoderLayerNames = ['x'+str(i) for i in range(numEncoderBlocks)]
        baseNumberOfFilters = 64
        counter = 1
        encoderLayers = {}
        encoderInputs = tf.keras.layers.Input(shape=self.inputShape) #28 x 28
        for layerName in encoderLayerNames:
            
            if counter == 1:
                encoderLayers[layerName] = self.encoderBlock(numFilters=baseNumberOfFilters, layerInputs=encoderInputs)
            else:
                encoderLayers[layerName] = self.encoderBlock(numFilters=baseNumberOfFilters, layerInputs=pool)
            #encoderLayers[layerName] = layerName
            pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(encoderLayers[layerName])
            baseNumberOfFilters *= 2
            counter+=1
        #bottleneck
        x = tf.keras.layers.Conv2D(filters = baseNumberOfFilters, kernel_size = 3, padding='same', activation='relu')(pool)
        x = tf.keras.layers.Conv2D(filters = baseNumberOfFilters, kernel_size = 3, padding='same', activation='relu')(x)
        #decoder with skip connections
        for i in range(len(encoderLayerNames)-1, -1, -1):
            baseNumberOfFilters = int(baseNumberOfFilters/2)
            x = self.decoderBlock(numFilters= baseNumberOfFilters, layerInputs=x, 
                                  skipConnections = encoderLayers[encoderLayerNames[i]])
            
        output = tf.keras.layers.Conv2D(filters = imageDim, kernel_size = 1, padding='same', activation='sigmoid')(x)

        '''    
        #The unet will take an image with noise added and predict the noise
        encoderInputs = tf.keras.layers.Input(shape=self.inputShape) #28 x 28
        #encoder block 1
        x1 = self.encoderBlock(numFilters=64, layerInputs=encoderInputs)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x1)
        print('x1 ', x1.shape)
        x2 = self.encoderBlock(numFilters=128, layerInputs=pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x2)
        print('x2 ', x2.shape)
        x3 = self.encoderBlock(numFilters=256, layerInputs=pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x3)
        #x4 = self.encoderBlock(numFilters=512, layerInputs=x3)
        print('x3 ', x3.shape)
        #bottleneck
        x5 = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(pool3)
        x6 = tf.keras.layers.Conv2D(filters = 512, kernel_size = 3, padding='same', activation='relu')(x5)
        print('x6 ', x6.shape)
        #decoder
        #x7 = self.decoderBlock(numFilters= 512, layerInputs=x6, skipConnections = x4)
        x8 = self.decoderBlock(numFilters= 256, layerInputs=x6, skipConnections = x3)
        x9 = self.decoderBlock(numFilters= 128, layerInputs=x8, skipConnections = x2)
        x10 = self.decoderBlock(numFilters= 64, layerInputs=x9, skipConnections = x1)

        output = tf.keras.layers.Conv2D(filters = 3, kernel_size = 1, padding='same', activation='sigmoid')(x10)
        '''
        mdl = tf.keras.Model(inputs = encoderInputs, outputs = output)
        print(mdl.summary())
        return encoderInputs, output



