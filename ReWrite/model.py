'''
import tensorflow as tf


# dim is output dimension
def build_resnet_block(inputres, dim, name="resnet"):
    # variable "resnet"
    with tf.variable_scope(name):
        # add the reflect padding with second and third dimension
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        # input is the reflected padding image, dim is the output dimension
        # can change to one variable
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
        # padding with the reflect again, (we lose dim in general_conv2d)
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
        out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2", do_relu=False)
        
        return tf.nn.relu(out_res + inputres) # ?? 
    

    
# fl = first layer
def build_generator_c_start(inputgen, fl_conv, conv_size, fl_stride=1, stride=2, g_name = "generator", name="c"):
    ks = conv_size
    f = fl_conv
    with tf.variable_scope(g_name):
        paddings = [[0, 0], [ks, ks], [ks, ks], [0, 0]]
        # padding should be changed if we use smaller images
        input_g = tf.pad(inputgen, paddings, "REFLECT")
        temp_o = None
        # ? is the output dims with different filters?
        for i in range(1, c_layers+1):
            name_c = name+str(i)
            print(i)
            if i==1:
                temp_o = general_conv2d(input_g, ngf, f, f, fl_stride, fl_stride, 0.02, name=name_c)
                input_g, ngf = temp_o, ngf*2
            else:
                temp_o = general_conv2d(input_g, ngf, f, f, stride, stride, 0.02, name=name_c)
                input_g, ngf = temp_o, ngf*2
    return input_g

def build_n_resnet_blocks(input_c, res_num, g_name = "generator", name='r'):
    r_output = input_c
    with tf.variable_scope(g_name):
        for i in range(1, res_num+1)
            res_name = name+str(i)
            r_output = build_resnet_block(r_output, ngf*4, name=res_name)        
    return r_output

def build_generator_c_end(input_r, g_name="generator"):
    with tf.variable_scope(g_name):

    pass

def build_generator_resnet_nblocks(inputgen, f_size, conv_size, c_layers_start, res_layers, c_layers_end, name="generator"):
    build_generator_c_start

    
'''