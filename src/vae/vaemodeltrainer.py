from src.modeltraining.generativemodeltrainer import GenerativeModelTrainer
from src.vae.vaearchitecture import VAEArchitecture
from src.vae.customvae import CustomVAE
import tensorflow as tf

class VaeModelTrainer(GenerativeModelTrainer):

    def __init__(self, xTrain, trainEpochs: int = 30):
        super().__init__()
        self.xTrain = xTrain
        self.trainEpochs =trainEpochs

    def _buildArchitecture(self):
        inShp = (self.xTrain.shape[1], self.xTrain.shape[2], self.xTrain.shape[3])
        arc = VAEArchitecture(inputShape = inShp, latentDim=2)
        self.encoder = arc.buildEncoder()
        self.decoder = arc.buildDecoder()

    def _trainModel(self):
        #create custom model (vae)
        self.customMdl = CustomVAE(encoder = self.encoder, decoder = self.decoder)
        #call custom train step
        self.customMdl.compile(optimizer = tf.keras.optimizers.Adam())
        #pass xTrain one time; passing validation data causes error
        self.customMdl.fit(self.xTrain, 
            epochs = self.trainEpochs, 
            batch_size = 256, 
            shuffle=True, 
            verbose=True) 
        
    def getTrainedModel(self):
        return self.customMdl
    
    def _saveTrainedModel(self, filepath: str):
        self.customMdl.saveDecoder(filepath=filepath)
    
        
