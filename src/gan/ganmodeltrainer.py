from src.modeltraining.generativemodeltrainer import GenerativeModelTrainer
from src.gan.ganArchitecture import GANArchitecture
import tensorflow

class GANModelTrainer(GenerativeModelTrainer):

    def __init__(self, xTrain, yTrain=None, trainEpochs: int = 30):
        super().__init__()
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.trainEpochs =trainEpochs
        self.noiseDim = 100

    def _buildArchitecture(self):
        architecture = GANArchitecture()
        self.generator = architecture.createGenerator()
        self.discriminator = architecture.createDiscriminator()

    def _trainModel(self):
        #pass xTrain one time; passing validation data causes error
        dataset = tensorflow.data.Dataset.from_tensor_slices(self.xTrain)
        dataset = dataset.shuffle(buffer_size=1024).batch(256)
        
        for epoch in range(self.trainEpochs):
            print('epoch: ', epoch)
            for step, (x_batch_train) in enumerate(dataset):
                noise = tensorflow.random.normal([x_batch_train.shape[0], self.noiseDim])
                with tensorflow.GradientTape() as grad_tape, tensorflow.GradientTape() as disc_tape:
                    gen_images = self.generator(noise)

                    real_discriminator = self.discriminator(x_batch_train)
                    gen_discriminator = self.discriminator(gen_images)
                    real_labels = [1 for _ in range(real_discriminator.shape[0])]
                    fake_labels = [0 for _ in range(gen_images.shape[0])]

                    crossentropyLoss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)
                    gen_loss = crossentropyLoss(gen_images, x_batch_train)
                    real_disc_loss = crossentropyLoss(real_discriminator, real_labels)
                    fake_disc_loss = crossentropyLoss(gen_discriminator, fake_labels)
                    disc_loss = real_disc_loss + fake_disc_loss
                    #disc_loss = tensorflow.add(crossentropyLoss(real_discriminator, real_labels),  
                    #            crossentropyLoss(gen_discriminator, fake_labels))
                    

                gen_grads = grad_tape.gradient(gen_loss, self.generator.trainable_weights)
                disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

                # Run one step of gradient descent by updating
                # the value of the variables to minimize the loss.
                self.generator.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
                self.discriminator.optimizer.apply_gradients(zip(disc_grads), self.discriminator.trainable_weights)
        

    def _saveTrainedModel(self, filepath: str):
        self.generator.save(filepath=filepath)