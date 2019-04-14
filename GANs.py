import numpy as np
import matplotlib.pyplot as plt
import pickle, gzip
from keras.layers import Input, Activation, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
import keras.backend as K
from keras.optimizers import adam
from tqdm import tqdm
from keras.utils import np_utils

filepath = "/Users/AshleyRamos/PycharmProjects/statMLGAN/"
IMG_SZ = 28 # width and length
NUM_LBLS = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = IMG_SZ * IMG_SZ
#train_data = np.loadtxt(filepath + "mnist_train.csv", delimiter=",")
#test_data = np.loadtxt(filepath + "mnist_test.csv",
#                       delimiter=",")


#####################################################
# PICKLE DATA
#####################################################
#scaling pixel values down to range[.01:.99]
#fac = 255  *0.99 + 0.01
#train_imgs = np.asfarray(train_data[:, 1:]) / fac
#test_imgs = np.asfarray(test_data[:, 1:]) / fac
#train_labels = np.asfarray(train_data[:, :1])
#test_labels = np.asfarray(test_data[:, :1])

#lr = np.arange(NUM_LBLS)
# transform labels into one hot representation
#train_labels_one_hot = (lr==train_labels).astype(np.float)
#test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
#train_labels_one_hot[train_labels_one_hot == 0] = 0.01
#train_labels_one_hot[train_labels_one_hot == 1] = 0.99
#test_labels_one_hot[test_labels_one_hot == 0] = 0.01
#test_labels_one_hot[test_labels_one_hot == 1] = 0.99

#with open(filepath + "pickled_mnist.pkl", "bw") as fh:
#    data = (train_imgs,
#            test_imgs,
#            train_labels,
#            test_labels,
#            train_labels_one_hot,
#            test_labels_one_hot)
#    pickle.dump(data, fh)

with open(filepath + "pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
# example of one-hot representation for caategorical variables
lr = np.arange(10)
for label in range(10):
    one_hot = (lr == label).astype(np.int)
    print("label: ", label, " in one-hot representation: ", one_hot)

# lets see what some of these numbers look like
# You can change the color of the sequential color map with the max being the brightest rep of color
for i in range(10):
    img = train_imgs[i].reshape((IMG_SZ, IMG_SZ))
    plt.imshow(img, cmap="RdPu")
    # plt.savefig(train_data[i,0].astype('str') + ".png") #save the plot to a jpg
    plt.show()


#################################
#GENERATOR
#################################

"""Generator class will learn from the GAN"""


class Generator:
    """this generator will have a set number of channels for each of
    the 4 hidden layers
    We will ensure that the output is a 784 size array since we are
    trying to get an image result"""
    def __init__(self, dim):
        self.first_layer = 256
        self.second_layer = 512
        self.third_layer = 1024
        self.fourth_layer = 784
        self.input_dim = dim

    def create_model(self):
        model = Sequential()
        model.add(Dense(units=self.first_layer,
                        input_dim=self.input_dim,
                        activation='relu'))
        model.add(Dense(units=self.second_layer,
                        activation='relu'))
        model.add(Dense(units=self.third_layer,
                        activation='relu'))
        model.add(Dense(units=self.fourth_layer,
                        activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=adam(lr=0.0002, beta_1=0.5),
                      metrics=['accuracy'])

        return model

    def train_model(self, X, Y, epochs, batch_size):

        """batch size is number of instances taken
        before a weight update in the network is performed

        epochs is the number of iterations"""
        self.model.fit(X,Y,
                       epochs=epochs,
                       batch_size=batch_size)

    def predict(self, dataToPredict):

        return self.model.predict(dataToPredict)

    def eval_model(self, x_val,y_val):
        scores = self.model.evaluate(x_val,y_val)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))


"""This class is the discriminator class which 
will include all of the necessary methods to call and train the discriminator"""


class Discriminator:

    def __init__(self, xx):

        self.trainData = xx
        #self.train_model(np.asarray(self.trainData).
        #                reshape(len(xx), 28, 28, 1))

    #@staticmethod
    def create_model(self):

        # Model Type: Sequential, Provides one output defined by the classes
        # In our case we only care for True or False output (0,1)
        discriminator = Sequential()
        # Convolution Kernel to produce a tensor
        discriminator.add(Conv2D(32, kernel_size=(3, 3),
                                 strides=(1, 1),
                                 input_shape=(28, 28, 1)))
        # normalizes the matrix after convolution layer
        # each dimension is kept in same scale
        discriminator.add(BatchNormalization(axis=-1))
        # max(0,x) - makes negative number equal zero
        # reduces training time and vanishing gradients
        discriminator.add(Activation('relu'))
        discriminator.add(Conv2D(32, kernel_size=(3, 3)))
        discriminator.add(BatchNormalization(axis=-1))
        discriminator.add(Activation('relu'))
        discriminator.add(MaxPooling2D(pool_size=(2, 2)))

        discriminator.add(Conv2D(64, (3, 3)))
        discriminator.add(BatchNormalization(axis=-1))
        discriminator.add(Activation('relu'))
        discriminator.add(Conv2D(64, (3, 3)))
        discriminator.add(BatchNormalization(axis=-1))
        discriminator.add(Activation('relu'))
        # downsamples the input.
        # learns on the feautres. Reduces overfitting
        discriminator.add(MaxPooling2D(pool_size=(2, 2)))

        # Need to flatten the output of the layers
        # so they can be input to dense layer
        discriminator.add(Flatten())

        # Dense layers are fully connected layers
        # serve for classification
        discriminator.add(Dense(512))
        discriminator.add(BatchNormalization())
        discriminator.add(Activation('relu'))
        discriminator.add(Dropout(0.2))
        discriminator.add(Dense(1))

        # Provides a 0 or 1 output - F or T
        discriminator.add(Activation('sigmoid'))

        # for two classes we define a binary crossentropy loss
        # Adam improves SGD. User backpropagation to update weights
        discriminator.compile(loss='binary_crossentropy' , optimizer=adam(lr=0.0002, beta_1=0.5))

        return discriminator

    def train_model(self, x_train):
        # Generate noisy images

        noise = np.random.normal(0, 1, [200, 28, 28, 1])  # 210 noisy images
        K.cast(noise, dtype="float32")
        print("noise shape:", noise.shape)
        print(np.asarray(x_train).shape)
        # Train on digits and noise
        X_train = np.concatenate((noise,
                                  np.asarray(x_train))[:],
                                 axis=0)
        # Assign 0 for noise and 1 for digits
        trainLabels = np.append(np.zeros(200), np.ones(len(x_train)))

        # Begin Training!
        self.model.fit(X_train, trainLabels, epochs=10)

    def test_model(self, x_test):

        # Test model
        noise = np.random.normal(0, 1, [210, 784])
        minitest = np.concatenate((noise[200:], x_test[0:32, :]), axis=0)
        results = self.model.predict_classes(minitest)
        actual = np.concatenate((-1*np.ones(10), x_test[1][0:32]), axis=0)
        print("Predicted results (0/1): ", results.T)
        print("Actual Numbers (-1 = noise): ", actual)


class GAN:
    def __init__(self, disc, gen, X_train):
        self.discriminator = disc
        self.generator = gen
        self.trainSet = X_train
        self.create_GAN()

    def create_GAN(self):

        self.discriminator.trainable = False
        gan_input = Input(shape=np.asarray(self.trainSet).shape)
        x = self.generator.create_model(gan_input)
        gan_output = self.discriminator.create_model(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')

        return gan

    def plot_generated_images(self, epoch, generator, examples=100,
                              dim=(10,10), figsize=(10,10)):

        noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('gan_generated_image %d.png' % epoch)

    def train_GAN(self, epochs=1, batch_size=128):

        # Loading the data
        batch_count = X_train.shape[0] / batch_size

        """Here, we are creating the GAN based on the neural net constructed
        by the input discriminator and generator"""
        gan = self.create_GAN()

        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            """TQDM outputs a nice progress bar"""
            for _ in tqdm(range(batch_size)):
                # generate  random noise as an input  to  initialize the  generator
                noise = np.random.normal(0, 1, [batch_size, 100])

                # Generate fake MNIST images from noised input
                generated_images = self.generator.predict(noise)

                # Get a random set of  real images
                image_batch = X_train[np.random.randint(low=0, high=X_train.shape[0], size=batch_size)]

                # Construct different batches of  real and fake data
                real_and_fake = np.concatenate([image_batch, generated_images])

                # Labels for generated and real data
                # The real data gets a score of 0.9
                y_dis = np.zeros(2 * batch_size)
                y_dis[:batch_size] = 0.9

                # Pre train discriminator on  fake and real data  before starting the gan.
                # Runs a single gradient update on a single batch of data.
                self.discriminator.trainable = True
                self.discriminator.train_on_batch(real_and_fake, y_dis)

                # Tricking the noised input of the Generator as real data
                noise = np.random.normal(0, 1, [batch_size, 784])
                y_gen = np.ones(batch_size)

                # During the training of gan,
                # the weights of discriminator should be fixed.
                # We can enforce that by setting the trainable flag
                self.discriminator.trainable = False

                # training  the GAN by alternating the training of the Discriminator
                # and training the chained GAN model with Discriminator’s weights constant.
                gan.train_on_batch(noise, y_gen)

            if e == 1 or e % 20 == 0:
                self.plot_generated_images(e, self.generator)


# Input Parameters
n_input = image_pixels # number of features

TRAIN = int(np.floor(0.75 * len(train_imgs)))

train_imgs[:].reshape(60000, n_input)

X_train = train_imgs[:TRAIN]
Y_train = train_labels_one_hot[:TRAIN]
X_validation = train_imgs[TRAIN:]
Y_Validation = train_labels_one_hot[TRAIN:]

print("Validation Size: ", len(X_validation))
epochs = 300
batch_size = 100
gen = Generator()

#noise = np.random.normal(0,1, [batch_size, n_input])
#predictions = gen.predict(noise)
#img = predictions[0].reshape((IMG_SZ, IMG_SZ))
#plt.imshow(img, cmap="RdPu")
# plt.savefig(train_data[i,0].astype('str') + ".png") #save the plot to a jpg
plt.show()

disc = Discriminator(X_train)
gan = GAN(disc, gen, X_train)
gan.train_GAN(epochs, batch_size)





