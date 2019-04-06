
import numpy as np
import matplotlib.pyplot as plt

filepath = "C:\\Users\\josev\\OneDrive\\Documents\\GRAD\\ECE 6254 Stat Machine Learning\\Project\\MNIST Dataset\\"
IMG_SZ = 28 # width and length
NUM_LBLS = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = IMG_SZ * IMG_SZ
train_data = np.loadtxt(filepath + "mnist_train.csv", delimiter=",")
test_data = np.loadtxt(filepath + "mnist_test.csv",
                       delimiter=",")
#scaling pixel values down to range[.01:.99]
fac = 255  *0.99 + 0.01
train_imgs = np.asfarray(train_data[:, 1:]) / fac
test_imgs = np.asfarray(test_data[:, 1:]) / fac
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])

lr = np.arange(NUM_LBLS)
# transform labels into one hot representation
train_labels_one_hot = (lr==train_labels).astype(np.float)
test_labels_one_hot = (lr==test_labels).astype(np.float)
# we don't want zeroes and ones in the labels neither:
train_labels_one_hot[train_labels_one_hot == 0] = 0.01
train_labels_one_hot[train_labels_one_hot == 1] = 0.99
test_labels_one_hot[test_labels_one_hot == 0] = 0.01
test_labels_one_hot[test_labels_one_hot == 1] = 0.99

print(train_labels_one_hot[0])

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