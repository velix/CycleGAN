import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(mnist.train.images[1]))

print(len(mnist.train.images))
print(len(mnist.validation.images))
print(len(mnist.test.images))
save_3D_images = np.zeros(shape=(70000,32,32,3))
index = 0
# for i in range(10):
for i in range(len(mnist.train.images)):
    image = mnist.train.images[i]
    image = tf.reshape(image, [-1, 28, 28, 1])  
    pad_image = tf.pad(image, [[0, 0], [2,2], [2,2], [0,0]])
    
    with tf.Session():
        pad_image = pad_image.eval()
    pad_image = np.asarray(pad_image)    
    pad_image = pad_image.reshape(32,32)
    pad_image_3D = np.zeros(shape=(3,32,32))
    pad_image_3D[0] = pad_image
    pad_image_3D[1] = pad_image
    pad_image_3D[2] = pad_image
    pad_image_3D = np.transpose(pad_image_3D, (1, 2, 0))
    save_3D_images[i] = pad_image_3D
    index = i
    if (i%1000==0):
        print(i, " train")


for i in range(len(mnist.validation.images)):
    image = mnist.validation.images[i]
    image = tf.reshape(image, [-1, 28, 28, 1])  
    pad_image = tf.pad(image, [[0, 0], [2,2], [2,2], [0,0]])
    with tf.Session():
        pad_image = pad_image.eval()
    pad_image = np.asarray(pad_image)    
    pad_image = pad_image.reshape(32,32)
    pad_image_3D = np.zeros(shape=(3,32,32))
    pad_image_3D[0] = pad_image
    pad_image_3D[1] = pad_image
    pad_image_3D[2] = pad_image
    pad_image_3D = np.transpose(pad_image_3D, (1, 2, 0))
    index += 1    
    save_3D_images[index] = pad_image_3D
    if (i%1000==0):
        print(i, " train")

for i in range(len(mnist.test.images)):
    image = mnist.test.images[i]
    image = tf.reshape(image, [-1, 28, 28, 1])  
    pad_image = tf.pad(image, [[0, 0], [2,2], [2,2], [0,0]])
    with tf.Session():
        pad_image = pad_image.eval()
    pad_image = np.asarray(pad_image)    
    pad_image = pad_image.reshape(32,32)
    pad_image_3D = np.zeros(shape=(3,32,32))
    pad_image_3D[0] = pad_image
    pad_image_3D[1] = pad_image
    pad_image_3D[2] = pad_image
    pad_image_3D = np.transpose(pad_image_3D, (1, 2, 0))
    index += 1    
    save_3D_images[index] = pad_image_3D
    if (i%1000==0):
        print(i, " train")

np.save('MNIST_3D_32_32', save_3D_images)
# images_3D = np.load('MNIST_3D_32_32.npy')
# print(images_3D.shape) 

#for i in range(10)
    



'''
test_image = mnist.train.images[1]
print(type(test_image))
print(test_image.shape)

test_image = tf.reshape(test_image, [-1, 28, 28, 1])
output = tf.pad(test_image, [[0, 0], [2,2], [2,2], [0,0]])

with tf.Session():
    output = output.eval()

a = np.asarray(output)

a = a.reshape(32,32)

test = np.zeros(shape=(3,32,32))
print(test[0].shape)
test[0] = a
test[1] = a
test[2] = a

plt.imshow(x.reshape(32,32,3), cmap=plt.cm.Greys)
plt.show()

np.save('outfile_name', a)
a = np.load('outfile_name.npy') 
'''