'''
Class for generating an encoder and decoder for a conditional vae model
This model will include the label so that a images of a specific class can be generated

'''
import tensorflow
from src.architectures.modelarchitecture import ModelArchitecture

class CvaeArchitecture(ModelArchitecture):
    """
    Conditional VAE will take the one hot encoded labels as an input; other is the same as the traditional vae; 
    #TODO can this be refactored? there is a lot of duplication with the vae
    """
    def __init__(self, inputShape: tuple, numclasses: int, latentDim: int = 2):
        super().__init__()
        self.latentDim = latentDim
        self.inputShape = inputShape
        self.numclasses = numclasses

    def buildEncoder(self):
        encoderInputs = tensorflow.keras.layers.Input(shape=self.inputShape) #28 x 28
        oneHotInput = tensorflow.keras.layers.Input(shape = (self.numclasses))
        
        x = tensorflow.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(encoderInputs) #14 x 14 x 32
        x = tensorflow.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x) #7 x 7 x 64
        x = tensorflow.keras.layers.Flatten()(x)
        #x = tensorflow.keras.layers.Dense(16, activation='relu')(x)

        #concatenate the labels
        x = tensorflow.keras.layers.concatenate([x, oneHotInput])

        
        z_mean = tensorflow.keras.layers.Dense(self.latentDim)(x) #autoencoder has to predict the mean
        z_std = tensorflow.keras.layers.Dense(self.latentDim)(x) #autoencoder has to predict the std dev
        #mean and std have to pass through a custom layer so that they can be trained through backprop
        z = CustomLayer()([z_mean, z_std])
        outputs = [z_mean, z_std, z]
        encoder = tensorflow.keras.Model([encoderInputs, oneHotInput], outputs)
        
        encoder.summary()
        return encoder

    def buildDecoder(self):
        initLayerDim = int(self.inputShape[0]/4) #assumes that there are only two upscaling layers
        latent_inputs = tensorflow.keras.layers.Input(shape=(self.latentDim))
        oneHotInput = tensorflow.keras.layers.Input(shape = (self.numclasses))
        
        x = tensorflow.keras.layers.concatenate([latent_inputs, oneHotInput])
        print(x.shape)

        x = tensorflow.keras.layers.Dense(initLayerDim * initLayerDim * 64, activation = 'relu')(x)
        x = tensorflow.keras.layers.Reshape((initLayerDim,initLayerDim,64))(x)
        x = tensorflow.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same', strides = 2)(x)
        x = tensorflow.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same', strides = 2)(x)
        decoderOutputs = tensorflow.keras.layers.Conv2DTranspose(self.inputShape[2],3, activation = 'sigmoid', padding = 'same', name='decoderOutputs')(x)
        
        decoder = tensorflow.keras.Model([latent_inputs, oneHotInput], decoderOutputs)
        
        decoder.summary()
        return decoder
    
class CustomLayer(tensorflow.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tensorflow.shape(z_mean)[0]
        dim = tensorflow.shape(z_mean)[1]
        epsilon = tensorflow.random.normal(shape=(batch, dim))
        return z_mean + tensorflow.exp(0.5 * z_var) * epsilon