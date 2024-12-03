import tensorflow

class CustomGAN():
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.noiseDim = 100
    '''
    #taken from tensorflow https://www.tensorflow.org/tutorials/generative/dcgan
    def train_step(self, data):
        #create random noise to pass into generator
        noise = tensorflow.random.normal([self.data.shape[0], self.noiseDim])
        with tensorflow.GradientTape() as grad_tape, tensorflow.GradientTape() as disc_tape:
            gen_images = self.generator(data)

            real_discriminator = self.discriminator(data)
            gen_discriminator = self.discriminator(gen_images)
            real_labels = [1 for _ in range(real_discriminator.shape[0])]
            fake_labels = [0 for _ in range(gen_images.shape[0])]

            crossentropyLoss = tensorflow.keras.losses.BinaryCrossentropy(from_logits=True)
            gen_loss = crossentropyLoss(gen_images, data)
            disc_loss = (crossentropyLoss(real_discriminator, real_labels) + 
                         crossentropyLoss(gen_discriminator, fake_labels))
            

        gen_grads = grad_tape.gradient(gen_loss, self.generator.trainable_weights)
        disc_grads = disc_tape.gradient(disc_loss, self.discriminator.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.generator.optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_weights))
        self.discriminator.optimizer.apply_gradients(zip(disc_grads), self.discriminator.trainable_weights)
        '''

    def discriminatorLoss(self):
        #the loss on real image and the loss on fake images
        pass

    def generatorLoss(self):
        pass
