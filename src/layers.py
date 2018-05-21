import tensorflow as tf

def lrelu(x, leak=0.2, name='lrelu', alt_relu_impl=False):

    with tf.variable_scope(name):
        # if statemen can be removed
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak*x)

# Applies Instance Norm
def instance_norm(x):

    with tf.variable_scope("instance_norm"):
        epsilon = 1e-5
        # Calculate the mean and variance of x. [1, 2] means height and width
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

        # Gets an existing variable with these parameters or create a new one.
        scale = tf.get_variable('scale', [x.get_shape()[-1]],
                                initializer=tf.truncated_normal_initializer(
                                    mean=1.0, stddev=0.02))

        offset = tf.get_variable('offset', [x.get_shape()[-1]],
                                 initializer=tf.constant_initializer(0.0))

        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out


'''
VALID do not padding, size of the image will be changed
OBS!!! we could change the truncted to the default,
    weights_initializer=initializers.xavier_initializer()

inputconv: input tensor. [batch_size, img_width, img_height, img_layerA]
output_dim: number of output filters
kernel: kernel size for each filter
stride: stride
'''
def general_conv2d(inputconv, output_dim=64, kernel=7, stride=1,
                   stddev=0.02, padding="VALID", name="conv2d",
                   do_norm=True, do_relu=True, lrelu_slope=0):

    with tf.variable_scope(name):
        # This performs the convolution
        # There is no normalization done here
        # and no activation is applies to the output
        conv = tf.contrib.layers.conv2d(
                        inputconv, output_dim, kernel, stride, padding,
                        activation_fn=None,
                        weights_initializer=tf.truncated_normal_initializer(
                                                stddev=stddev),
                        biases_initializer=tf.constant_initializer(0.0))

        # If True, apply normalization
        # either batch_norm or
        # instance_norm, definde by us
        # tf.contrib.layers.instance_norm provides instance
        # norm implementation
        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(lrelu_slope == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                # Applies Leaky ReLu
                # tf.nn.leaky_relu provides implementation, for tf>=1.4
                conv = lrelu(conv, lrelu_slope, "lrelu")

        return conv

# Apples inverse convolution
# inputconv : tensor input
# output_dim : dimension of the output
def general_deconv2d(inputconv, output_dim=64, kernel=7,
                     stride=1, stddev=0.02, padding="VALID",
                     name="deconv2d", do_norm=True, do_relu=True,
                     lrelu_slope=0):

    kernel_dim = _make_list(kernel)
    stride_dim = _make_list(stride)

    with tf.variable_scope(name):
        # Applies inverse convolution
        # stride and kernel arguements are lists of two values
        # specifying the width and height. See docs before changing
        # Applies no activation function
        conv = tf.contrib.layers.conv2d_transpose(
                inputconv, output_dim, kernel_dim,
                stride_dim, padding, activation_fn=None,
                weights_initializer=tf.truncated_normal_initializer(
                                                stddev=stddev),
                biases_initializer=tf.constant_initializer(0.0))

        if do_norm:
            conv = instance_norm(conv)
            # conv = tf.contrib.layers.batch_norm(conv, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope="batch_norm")

        if do_relu:
            if(lrelu_slope == 0):
                conv = tf.nn.relu(conv, "relu")
            else:
                conv = lrelu(conv, lrelu_slope, "lrelu")

        return conv


def _make_list(arg):
    if type(arg) == list:
        return arg
    else:
        out = [arg, arg]

    return out
