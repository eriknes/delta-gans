import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt


from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

def load_file(fname):
  X = pd.read_csv(fname)
  X = X.values
  X = X.astype('uint8')
  return X
  

def create_dataset(X, n_braided, nx, ny, n_test = 1000):
  
  n_tot = X.shape[0]
  n_tidal = n_tot - n_braided
  print("Number of braided samples: " + str(n_braided) )
  print("Number of tidal samples: " + str(n_tidal) )
  
  X = X.T
  Y = np.zeros((n_braided+n_tidal))
  Y[0:n_braided] = 0
  Y[n_braided:n_braided+n_tidal] = 1
  
  # Random permutation
  p = np.random.permutation(n_braided)
  X = X[:,p]
  Y = Y[p]
  
  # Reshape X
  X_new = np.zeros((n_braided,nx,ny))
  for i in range(n_braided):
    Xtemp = np.reshape(X[:,i],(101,101))
    X_new[i,:,:] = Xtemp[2:98,2:98]
  
  
  X_train = X_new[0:n_braided-n_test,:,:]
  Y_train = Y[0:n_braided-n_test]
  
  X_test  = X_new[n_braided-n_test:n_braided,:,:]
  Y_test  = Y[n_braided-n_test:n_braided]
  
  print("X_train shape: " + str(X_train.shape))
  print("Y_train shape: " + str(Y_train.shape))
  
  return X_train, Y_train, X_test, Y_test

K.set_image_dim_ordering('th')

# Deterministic output.
# Tired of seeing the same results every time? Remove the line below.
#np.random.seed(1)

# The results are a little better when the dimensionality of the random vector is only 10.
# The dimensionality has been left at 100 for consistency with other GAN implementations.
randomDim = 50

fname = "fluvialStylesData.csv"
X_train = load_file(fname)

n_braided = 26355
nx = 96
ny = 96
n_test = 500
X_train, y_train, X_test, y_test = create_dataset(X_train, n_braided, nx, ny)
X_train = X_train[:, np.newaxis, :, :]

# Load MNIST data
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = (X_train.astype(np.float32) - 127.5)/127.5
#X_train = X_train[:, np.newaxis, :, :]

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
generator = Sequential()
generator.add(Dense(256*12*12, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(Activation('relu'))
generator.add(Reshape((256, 12, 12)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(128, kernel_size=(6,6), padding='same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(6, 6), padding='same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(12, 12), padding='same', activation='sigmoid'))
generator.compile(loss='binary_crossentropy', optimizer=adam)

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(16, 16), strides=(2, 2), padding='same', input_shape=(1, nx, ny), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(128, kernel_size=(8, 8), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Conv2D(512, kernel_size=(4, 4), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)

# Combined network
discriminator.trainable = False
ganInput = Input(shape=(randomDim,))
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

dLosses = []
gLosses = []

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/dcgan_loss_epoch_%d.png' % epoch)

# Create a wall of generated MNIST images
def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, randomDim])
    generatedImages = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/dcgan_generated_image_epoch_%d.png' % epoch)

# Save the generator and discriminator networks (and weights) for later use
def saveModels(epoch):
    generator.save('models/dcgan_generator_epoch_%d.h5' % epoch)
    discriminator.save('models/dcgan_discriminator_epoch_%d.h5' % epoch)

def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] / batchSize
    print 'Epochs:', epochs
    print 'Batch size:', batchSize
    print 'Batches per epoch:', batchCount

    for e in xrange(1, epochs+1):
        print '-'*15, 'Epoch %d' % e, '-'*15
        for _ in tqdm(xrange(batchCount)):
            # Get a random set of input noise and images
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)

        # Store loss of most recent batch from this epoch
        dLosses.append(dloss)
        gLosses.append(gloss)
        print('Generator Loss: '+str(gloss) + ', Discriminator Loss: ' + str(dloss))

        plotGeneratedImages(e)
        # Plot losses from every epoch
        plotLoss(e)

        if e == 1 or e % 5 == 0:
            saveModels(e)
            
 

if __name__ == '__main__':
    train(40, 128)

