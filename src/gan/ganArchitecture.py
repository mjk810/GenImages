import tensorflow
import matplotlib.pyplot as plt

class GANArchitecture():
    def __init__(self):
        self.latentShape = (100,)
        self.imageShape = (28, 28, 1)


    #TODO add dropout and batch norm layers - see paper (?)
    def createGenerator(self):
        #takes noise and outputs an image
        inputs = tensorflow.keras.layers.Input(shape = self.latentShape)
        x = tensorflow.keras.layers.Dense(7*7*256)(inputs) #want to go from a vector of len 100 to a 28 x 28 image
        x = tensorflow.keras.layers.Reshape((7, 7, 256)) (x)
        x = tensorflow.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(x)
        x = tensorflow.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(x)
        outputs = tensorflow.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(x)
       
        generator = tensorflow.keras.Model(inputs, outputs)
        generator.compile(optimizer = tensorflow.keras.optimizers.Adam(), run_eagerly=True)
        return generator

    
    
    def generatorRandomNoise(self):
        return tensorflow.random.normal([1, 100])
    
    #TODO add dropout and batch norm layers
    def createDiscriminator(self):
        #takes in an image and outputs a prediction of real/fake
        inputs = tensorflow.keras.layers.Input(shape = self.imageShape)
        x = tensorflow.keras.layers.Conv2D(filters=64, kernel_size = (5, 5), strides=(2, 2), padding='same')(inputs)
        x = tensorflow.keras.layers.Conv2D(filters=128, kernel_size = (5, 5), strides=(2, 2), padding='same')(x)
        x = tensorflow.keras.layers.Flatten()(x)
        outputs = tensorflow.keras.layers.Dense(1)(x)
        
        discriminator = tensorflow.keras.Model(inputs, outputs)
        discriminator.compile(optimizer = tensorflow.keras.optimizers.Adam(), run_eagerly=True)
        return discriminator
#
'''
gan = GANArchitecture()
gen = gan.getGenerator()

generated_image = gen(gan.generatorRandomNoise(), training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.show()

disc = gan.getDiscriminator()
prediction = disc(generated_image)
print('prediction: ', prediction)
'''