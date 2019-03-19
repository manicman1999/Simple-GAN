#Import Keras Things
from keras.layers import Dense, Reshape, Conv2D, UpSampling2D, AveragePooling2D, Activation, Flatten, Input
from keras.optimizers import RMSprop
from keras.models import Sequential, Model

#Import Numpy for dealing with data outside the models
import numpy as np

#Import matplotlib for plotting the images
import matplotlib.pyplot as plt

#Import PIL for importing images
from PIL import Image

#Parameters for training
learning_rate = 0.0003
batch_size = 8

#Helpful function for noise sampling
def noise(batch):
    return np.random.normal(0, 1, size = [batch, 100])

#Helpful function for getting images from an image array
def get_rand(array, batch):
    idx = np.random.randint(0, array.shape[0], batch)
    return array[idx]

#Helpful function for importing images
def import_images(loc, n):
    out = []
    
    for n in range(1, n + 1):
        temp = Image.open("data/"+loc+"/im ("+str(n)+").png").convert('RGB')
        temp = np.array(temp.convert('RGB'), dtype='float32') / 255
        out.append(temp)
            
    return np.array(out)

#Labels
positive_y = np.ones([batch_size, 1])
negative_y = np.zeros([batch_size, 1])

#Generator Model
generator = Sequential()
#100
generator.add(Dense(2048, input_shape = [100]))
generator.add(Reshape(target_shape = [4, 4, 128]))
#4x4x128
generator.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D())
#8x8x64
generator.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D())
#16x16x32
generator.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same'))
generator.add(Activation('relu'))
generator.add(UpSampling2D())
#32x32x16
generator.add(Conv2D(filters = 3, kernel_size = 3, padding = 'same'))
generator.add(Activation('sigmoid'))
#Output 32x32x3

#Discriminator Model
discriminator = Sequential()
#32x32x3
discriminator.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', input_shape = [32, 32, 3]))
discriminator.add(Activation('relu'))
discriminator.add(AveragePooling2D())
#16x16x16
discriminator.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same'))
discriminator.add(Activation('relu'))
discriminator.add(AveragePooling2D())
#8x8x32
discriminator.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same'))
discriminator.add(Activation('relu'))
discriminator.add(AveragePooling2D())
#4x4x64
discriminator.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same'))
discriminator.add(Activation('relu'))
discriminator.add(Flatten())
#2048
discriminator.add(Dense(128))
discriminator.add(Activation('relu'))
#128
discriminator.add(Dense(1))
#Output 1


#Compile Models

#Freeze G parameters
for layer in generator.layers:
    layer.trainable = False

#Get Real Samples
real = Input([32, 32, 3])

#Get Fake Samples
latent = Input([100])
fake = generator(latent)

#Discriminator Outputs
d_real = discriminator(real)
d_fake = discriminator(fake)

#Compile Model Together
DisModel = Model(inputs = [real, latent], outputs = [d_real, d_fake])
DisModel.compile(optimizer = RMSprop(lr = learning_rate), loss = ['mse', 'mse'])


#Freeze D parameters, unfreeze G parameters
for layer in generator.layers:
    layer.trainable = True

for layer in discriminator.layers:
    layer.trainable = False

#Get Input and Generate
latent = Input([100])
fake = generator(latent)
d_fake = discriminator(fake)

#Compile
GenModel = Model(inputs = latent, outputs = d_fake)
GenModel.compile(optimizer = RMSprop(lr = learning_rate), loss = 'mse')


#Import Data
x_train = import_images("Swords", 400)


while(True):
    #Finally, train the models.
    for step in range(1000):

        #Get Real Images
        real_images = get_rand(x_train, batch_size)

        #Sample Noise
        latent_variables = noise(batch_size)

        #Train Discriminator
        (d_loss, d_loss_real, d_loss_fake) = DisModel.train_on_batch([real_images, latent_variables], [positive_y, negative_y])

        #Sample More Noise
        latent_variables = noise(batch_size)

        #Train Generator
        g_loss = GenModel.train_on_batch(latent_variables, positive_y)

        #Print Results
        print("Step " + str(step))
        print("D Loss Real: " + str(d_loss_real))
        print("D Loss Fake: " + str(d_loss_fake))
        print("G Loss Fake: " + str(g_loss))

    #Sample
    samples = generator.predict(noise(10))
    
    for i in range(10):
        plt.figure(i)
        plt.imshow(samples[i])

    plt.show()
        














