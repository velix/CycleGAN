from PIL import Image as PILImage
from scipy.misc import toimage
import os

import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def get_mnist_batch(batch_size=256, change_colors=False):
    # Select random batch (WxHxC)
    idx = np.random.choice(x_train.shape[0], batch_size)
    batch_raw = x_train[idx, :, :, 0].reshape((batch_size, 28, 28, 1))
    
    # Resize
    # batch_resized = np.asarray([scipy.ndimage.zoom(image, (2.3, 2.3, 1), order=1) for image in batch_raw])
    
    # Extend to RGB
    batch_rgb = np.concatenate([batch_raw, batch_raw, batch_raw], axis=3)
    
    # Make binary
    batch_binary = (batch_rgb > 0.5)
    
    batch = np.zeros((batch_size, 28, 28, 3))
    
    for i in range(batch_size):
        # Take a random crop of the Lena image (background)
        x_c = np.random.randint(0, lena.size[0] - 28)
        y_c = np.random.randint(0, lena.size[1] - 28)
        image = lena.crop((x_c, y_c, x_c + 28, y_c + 28))
        image = np.asarray(image) / 255.0

        if change_colors:
            # Change color distribution
            for j in range(3):
                image[:, :, j] = (image[:, :, j] + np.random.uniform(0, 1)) / 2.0

        # Invert the colors at the location of the number
        image[batch_binary[i]] = 1 - image[batch_binary[i]]

        batch[i] = image

    # Map the whole batch to [-1, 1]
    # batch = batch / 0.5 - 1

    return idx, batch


# Read MNIST data
mnist_train = input_data.read_data_sets("mnist").train

x_train = mnist_train.images
x_labels = mnist_train.labels
x_train = x_train.reshape(-1, 28, 28, 1).astype(np.float32)

# Read Lena image
lena = PILImage.open('./resources/Lenna.png')

# plt.imshow(lena)
# plt.axis('off')
# plt.show()

count = 400
indexes, examples = get_mnist_batch(count)

print(np.shape(examples))

# Map back to normal range
# examples = (examples + 1) * 0.5

# plt.figure(figsize=(15, 3))
# for i in range(count):
#     plt.subplot(2, count // 2, i+1)
#     plt.imshow(examples[i])
#     plt.axis('off')

# plt.tight_layout()
# plt.show()

for i in range(count):
    label = 'img_{}_{}'.format(i, x_labels[indexes[i]])

    filename = '{}.jpg'.format(label)

    if not os.path.exists(os.path.join('input', 'colorful_mnist')):
        os.mkdir(os.path.join('input', 'colorful_mnist'))

    toimage(examples[i, :, :, :], cmin=.0).save(os.path.join('input', 'colorful_mnist', filename))
