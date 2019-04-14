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

class Generator:

    def __init__(self, dim):
        self.first_layer = 256
        self.second_layer = 512
        self.third_layer = 1024
        self.fourth_layer = 784
        self.input_dim = dim
        self.model = self.create_model()


    def adam_optimizer(self):
        return adam(lr=0.0002, beta_1=0.5)

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
                      optimizer='adam',
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

    @staticmethod
    def plot_generated_images(epoch, generator, examples=100,
                              dim=(10,10), figsize=(10,10)):

        noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100,28,28)
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        #plt.savefig('gan_generated_image %d.png' %epoch)


# Input Parameters
n_input = image_pixels # number of features

TRAIN = int(np.floor(0.75 * len(train_imgs)))

train_imgs[:].reshape(60000, n_input)

X_train = train_imgs[:TRAIN]
Y_train = train_labels_one_hot[:TRAIN]
X_validation = train_imgs[TRAIN:]
Y_Validation = train_labels_one_hot[TRAIN:]

print(len(X_validation))
epochs = 100
batch_size = 100
gen = Generator(n_input)

noise= np.random.normal(0,1, [batch_size, n_input])
predictions = gen.predict(noise)
img = predictions[0].reshape((IMG_SZ, IMG_SZ))
plt.imshow(img, cmap="RdPu")
# plt.savefig(train_data[i,0].astype('str') + ".png") #save the plot to a jpg
plt.show()

