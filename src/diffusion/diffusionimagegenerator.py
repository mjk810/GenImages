'''
Generate images from a trained diffusion model
'''
from src.imageprocessing.imageGenerator import ImageGenerator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class DiffusionImageGenerator(ImageGenerator):

    def __init__(self, modelPath: str, imageShape: tuple, nSteps: int):
        super().__init__(modelPath=modelPath)
        self.imageShape: tuple = imageShape
        self.nSteps: int = nSteps
        

    def generateNoise(self):
        #generate an array of noise

        pass
    
    def generateImages(self, numberOfSamples: int):
        for _ in range(numberOfSamples):

            noise = np.random.rand(self.imageShape[0], self.imageShape[1], self.imageShape[2])
            for i in range(self.nSteps):
                pred = self.mdl.predict(tf.expand_dims(noise, axis = 0))
                #plt.imshow(pred[0], cmap='gray')
                #plt.show()
                mix_factor = 1/(self.nSteps - i)
                print('mix factor: ', mix_factor)
                p = (pred[0] * mix_factor)
                n = noise * (1 - mix_factor)
                noise = n+p
                #print(noise)
                #print(noise.shape)
                #self.generatedImages.append(pred[0]) #use to store the intermediate image steps
                #if i %10 == 0:
                #    plt.imshow(pred[0], cmap='gray')
                #    plt.show()
            self.generatedImages.append(pred[0]) #store the final images for each of the samples requested
            
        