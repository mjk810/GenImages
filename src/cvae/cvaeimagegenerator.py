
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from src.imageprocessing.imageGenerator import ImageGenerator
#generate image from vae model
#TODO refactor! this is duplicate with vae

class CvaeImageGenerator(ImageGenerator):
    def __init__(self, modelPath: str, digit: int):
        self.mdl = tf.keras.models.load_model(modelPath)
        self.digit = digit
        self.generatedImages = []


    def generateImages(self, numberOfSamples: int):
        sample = self._sampleLatentSpace(numSamples=numberOfSamples)
        labels = self.createLabels(numberOfSamples=numberOfSamples)
        for i in range(numberOfSamples):
            print(sample[i])
            print(labels[i])
            prediction = self.mdl.predict([tf.convert_to_tensor([sample[i]]), 
                                           tf.convert_to_tensor([labels[i]])])
            imShape = prediction.shape
            self.generatedImages.append(prediction[0].reshape(imShape[1], imShape[2], imShape[3]))
    
    def createLabels(self, numberOfSamples: int) -> list[int]: 
        return [[0 if i != self.digit else 1 for i in range(10) ] for j in range(numberOfSamples)]

        
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

