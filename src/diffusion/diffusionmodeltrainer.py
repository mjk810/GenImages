from src.modeltraining.generativemodeltrainer import GenerativeModelTrainer
from src.diffusion.basicunetarchitecture import BasicUnetArchitecture

import tensorflow as tf
import numpy as np

class DiffusionModelTrainer(GenerativeModelTrainer):

    def __init__(self, xTrain, yTrain=None, trainEpochs: int = 30):
        super().__init__()
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.trainEpochs =trainEpochs

    def _buildArchitecture(self):
        inShp = (self.xTrain.shape[1], self.xTrain.shape[2], self.xTrain.shape[3])
        #arc = CustomUnet(inputShape = inShp)
        arc=BasicUnetArchitecture(inputShape = inShp)
        self.encoderInputs, self.output = arc.buildModel(numEncoderBlocks=2, imageDim = 1)
        #encoderInputs, output = arc.buildModel(numEncoderBlocks=3, imageDim = 3) #for cifar
        

    def _trainModel(self):
        self.customMdl = tf.keras.Model(inputs=self.encoderInputs, outputs=self.output)
        #customMdl = CustomDiffusion(encoderInputs, output)
        self.customMdl.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)) 
        #pass xTrain one time; passing validation data causes error
        dataset = tf.data.Dataset.from_tensor_slices(self.xTrain)
        dataset = dataset.shuffle(buffer_size=1024).batch(128)
        loss_fn = tf.keras.losses.MeanSquaredError()

        train_acc_metric = tf.keras.metrics.MeanSquaredError()
        for epoch in range(self.trainEpochs):
            print('epoch: ', epoch)
            for step, (x_batch_train) in enumerate(dataset):
                batch_size = x_batch_train.shape[0]
                #print('batch size: ', batch_size)
                with tf.GradientTape() as tape:
                    amount = np.random.rand(1,batch_size) 
                    noisy = self._addNoise(data=x_batch_train, amount=amount)
                    #make prediction
                    predicted = self.customMdl(noisy)
                    totalLoss = loss_fn(predicted, x_batch_train)
                grads = tape.gradient(totalLoss, self.customMdl.trainable_weights)
                self.customMdl.optimizer.apply_gradients(zip(grads, self.customMdl.trainable_weights))
                train_acc_metric.update_state(predicted, x_batch_train)

            train_acc = train_acc_metric.result()
            print('Training loss over epoch ', str(train_acc))
            train_acc_metric.reset_states()
    
    def _addNoise(self, data: np.ndarray, amount: list[float]):
    
        noise = np.random.rand(data.shape[0],data.shape[1], data.shape[2], data.shape[3])
        amount = amount.reshape(-1, 1,1,1)
        return data*(1-amount) + noise*amount

    def _saveTrainedModel(self, filepath: str):
        self.customMdl.save(filepath=filepath)