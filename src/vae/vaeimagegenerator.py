
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from src.imageprocessing.imageGenerator import ImageGenerator
#generate image from vae model
class VAEImageGenerator(ImageGenerator):
    def __init__(self, modelPath: str):
        self.mdl = tf.keras.models.load_model(modelPath)
        self.generatedImages = []


    def generateImages(self, numberOfSamples: int):
        sample = self._sampleLatentSpace(numSamples=numberOfSamples)
        for i in range(numberOfSamples):
            print(sample[i])
            prediction = self.mdl.predict([sample[i]])
            imShape = prediction.shape
            self.generatedImages.append(prediction[0].reshape(imShape[1], imShape[2], imShape[3]))
    '''
    def showImages(self, rows: int=10, cols:int=10):
        for i in range(len(self.generatedImages)):
            if i < rows*cols:
                plt.subplot(rows,cols,i+1)
                plt.axis('off')
                plt.imshow(self.generatedImages[i], cmap='gray')
        plt.show()
    '''
    #this is probably the only method that needs to be custom to the model
    def _sampleLatentSpace(self, numSamples: int):
        #sample latent space and generate new images
        #generate pairs of points in the range -1 to 1
        z_sample = [[random.uniform(-3, 3), random.uniform(-3,3)] for _ in range(numSamples)]
        #z_sample = [[np.random.normal(1, 0.1, 1)[0], np.random.normal(1, 0.1,1)[0]] for _ in range(numSamples)]
        return z_sample

