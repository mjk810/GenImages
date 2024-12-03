
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod

class ImageGenerator(ABC):
    def __init__(self, modelPath: str):
        self.mdl = tf.keras.models.load_model(modelPath)
        self.generatedImages = []

    @abstractmethod
    def generateImages(self, numberOfSamples: int):
        pass

    def showImages(self, rows: int=10, cols:int=10):
        for i in range(len(self.generatedImages)):
            if i < rows*cols:
                plt.subplot(rows,cols,i+1)
                plt.axis('off')
                plt.imshow(self.generatedImages[i], cmap='gray')
        plt.show()



