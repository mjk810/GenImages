'''
Subclass model to implement a custom training step in order to 
calculate the kl divergence loss using the mean and std from the encoder
output and the reconstruction loss using the decoder output
'''
import tensorflow

class ConditionalVAE(tensorflow.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tensorflow.keras.metrics.Mean(name="total_loss")
        self.kl_loss_tracker = tensorflow.keras.metrics.Mean(name="klLoss")
        self.reconstruction_loss_tracker = tensorflow.keras.metrics.Mean(name="reconstructionLoss")

        
    def saveDecoder(self, filepath: str):
        self.decoder.save(filepath)

    def train_step(self, data):
        #print(data[1].numpy())
        #a custom training step
        with tensorflow.GradientTape() as tape:
            z_mean, z_std, z = self.encoder([data[0], data[1]])
            decoded = self.decoder([z, data[1]])

            klLoss = self.kllossFunction(z_mean, z_std)
            reconstructionLoss = self.reconstructionLoss(data[0], decoded)
            totalLoss = klLoss + reconstructionLoss
        
        grads = tape.gradient(totalLoss, self.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        #will show the loss values in the console for each batch
        self.total_loss_tracker.update_state(totalLoss)
        self.kl_loss_tracker.update_state(klLoss)
        self.reconstruction_loss_tracker.update_state(reconstructionLoss)
        
        return {"loss": self.total_loss_tracker.result(), "klLoss": self.kl_loss_tracker.result(),
                "reconstructionLoss":self.reconstruction_loss_tracker.result()}
    
    def kllossFunction(self, z_mean, z_std):
        kl_loss = -0.5 * (1 + z_std - tensorflow.square(z_mean) - tensorflow.exp(z_std))
        kl_loss = tensorflow.reduce_mean(tensorflow.reduce_sum(kl_loss, axis=1))
        return kl_loss
    
    def reconstructionLoss(self, y_true, y_pred):
        reconstructionLoss = tensorflow.reduce_mean(
            tensorflow.reduce_sum(
            tensorflow.keras.losses.binary_crossentropy(y_true, y_pred), axis=(1, 2)))
        return reconstructionLoss
    
    
    #copied property below directly from tensorflow example including the comments
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.total_loss_tracker, self.kl_loss_tracker, self.reconstruction_loss_tracker]

     