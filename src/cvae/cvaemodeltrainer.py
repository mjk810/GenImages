from src.modeltraining.generativemodeltrainer import GenerativeModelTrainer
from src.cvae.cvaearchitecture import CvaeArchitecture
from src.cvae.cvae import ConditionalVAE
import tensorflow as tf

class CvaeModelTrainer(GenerativeModelTrainer):

    def __init__(self, xTrain, yTrain, trainEpochs: int = 30):
        super().__init__()
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.trainEpochs =trainEpochs
        

    def _buildArchitecture(self):
        inShp = (self.xTrain.shape[1], self.xTrain.shape[2], self.xTrain.shape[3])
        arc = CvaeArchitecture(inputShape = inShp, latentDim=2, numclasses=10) #TODO this should not be hardcodee!
        self.encoder = arc.buildEncoder()
        self.decoder = arc.buildDecoder()

    def _trainModel(self):
        #create custom model (vae)
        self.customMdl = ConditionalVAE(encoder = self.encoder, decoder = self.decoder)
        #call custom train step
        self.customMdl.compile(optimizer = tf.keras.optimizers.Adam(), run_eagerly=True)
        #pass xTrain one time; passing validation data causes error
        self.customMdl.fit(self.xTrain, self.yTrain, 
            epochs = self.trainEpochs, 
            batch_size = 256, 
            shuffle=True, 
            verbose=True) 
        
    def getTrainedModel(self):
        return self.customMdl
    
    def _saveTrainedModel(self, filepath: str):
        self.customMdl.saveDecoder(filepath=filepath)
    
        
