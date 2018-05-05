import tensorflow as tf
import numpy as np
import os
import random
import sys

from layers import *
from old_model import *

img_height = 32
img_width = 32
img_layerA = 3
img_layerB = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"
batch_size = 1
pool_size = 50
max_images = 100
save_training_images = True

class CycleGAN:
    def input_setup(self):
        ''' 
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''
        # it has type of variable
        filenames_A = tf.train.match_filenames_once("./input/trainingSampleA/*.jpg")    
        self.queue_length_A = tf.size(filenames_A)
        # print(filenames_A)
        # print(type(filenames_A))
        filenames_B = tf.train.match_filenames_once("./input/trainingSampleB/*.png")    
        self.queue_length_B = tf.size(filenames_B)
        
        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        paddings = [[2,2], [2,2], [0, 0]]

        image = tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[28,28])
        image = tf.reshape(image, [28, 28, 1])
        # add the depth from 1 to 3
        image = tf.concat([image, image, image], axis=2)
        images = tf.pad(image, paddings, mode='CONSTANT', name=None)
        # with tf.Session():
          #  images = images.eval()
            
        # plt.imshow(images.reshape(32, 32, 3))
        # plt.show()
        self.image_A = tf.subtract(tf.div(images,16),1)
        # convert black background to white
        # don't work
        # self.image_A = 1 - self.image_A
        imageB = tf.reshape(tf.image.resize_images(tf.image.decode_png(image_file_B),[32,32]), [32, 32, 3])
        
        self.image_B = tf.subtract(tf.div(imageB, 16),1)
        # print("here")

    def input_read(self, sess):
        
        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        '''

        # Loading images into the tensors
        # This class implements a simple mechanism to coordinate the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # run the queue
        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)
        # it's a pool size of 50
        self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layerA))
        self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layerB))
        print(self.fake_images_A)
        print(type(self.fake_images_A))
        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layerA))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layerB))

        for i in range(max_images): 
            # after running the queue then ...
            image_tensor = sess.run(self.image_A)
            # if(image_tensor.size() == img_size*batch_size*img_layerA):
            # save all imageA to numpy array 
            self.A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layerA))

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            # print(image_tensor)
            # print(type(image_tensor), " tensor image B")
            # if(image_tensor.size() == img_size*batch_size*img_layerB):
            self.B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layerB))
        # for the exception
        coord.request_stop()
        # Wait for all the threads to terminate.
        coord.join(threads)
    def model_setup(self):

            ''' This function sets up the model to train

            self.input_A/self.input_B -> Set of training images.
            self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
            self.lr -> Learning rate variable
            self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
            '''

            self.input_A = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layerA], name="input_A")
            self.input_B = tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layerB], name="input_B")
            
            self.fake_pool_A = tf.placeholder(tf.float32, [None, img_width, img_height, img_layerA], name="fake_pool_A")
            self.fake_pool_B = tf.placeholder(tf.float32, [None, img_width, img_height, img_layerB], name="fake_pool_B")

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.num_fake_inputs = 0

            self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
            

            # RESNET 9 BLOCKS
            with tf.variable_scope("Model") as scope:
                # build_generator Tensor("Model/g_A/t1:0", shape=(1, 36, 36, 3), dtype=float32)
                self.fake_B = build_generator_resnet_6blocks(self.input_A, name="g_A")
                self.fake_A = build_generator_resnet_6blocks(self.input_B, name="g_B")
                
                self.rec_A = build_gen_discriminator(self.input_A, "d_A")
                self.rec_B = build_gen_discriminator(self.input_B, "d_B")
                
                scope.reuse_variables()

                self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
                self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
                
                self.cyc_A = build_generator_resnet_6blocks(self.fake_B, "g_B")
                self.cyc_B = build_generator_resnet_6blocks(self.fake_A, "g_A")
                
                scope.reuse_variables()

                self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
                self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")
                '''
                '''
            '''
            with tf.variable_scope("Model") as scope:
                self.fake_B = build_generator_resnet_2blocks(self.input_A, name="g_A")
                self.fake_A = build_generator_resnet_2blocks(self.input_B, name="g_B")
                self.rec_A = build_gen_discriminator(self.input_A, "d_A")
                self.rec_B = build_gen_discriminator(self.input_B, "d_B")

                scope.reuse_variables()

                self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
                self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
                self.cyc_A = build_generator_resnet_2blocks(self.fake_B, "g_B")
                self.cyc_B = build_generator_resnet_2blocks(self.fake_A, "g_A")

                scope.reuse_variables()

                self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
                self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")
            '''
    
    def loss_calc(self):
        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''
        
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # Returns all variables created with trainable=True
        # A list of Variable objects.
        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        # for var in self.model_vars: print(var.name)

        # Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)


    def train(self):

        ''' Training Function '''
        # Load Dataset from the dataset folder
        self.input_setup()  
        # Build the network
        self.model_setup()
        # Loss function calculations
        self.loss_calc()
        init = (tf.global_variables_initializer(), tf.local_variables_initializer())
        # Saves and restores variables.
        saver = tf.train.Saver()     
        with tf.Session() as sess:
            # run the variables first
            sess.run(init)
            #Read input to nd array
            self.input_read(sess)
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)
            # Writes Summary protocol buffers to event files
            writer = tf.summary.FileWriter("./output/2")
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)
            print("no problem")
            # Training Loop
            for epoch in range(sess.run(self.global_step),100):                
                print ("In the epoch ", epoch)
                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                # Learning decay
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100
                if(save_training_images):
                    self.save_training_images(sess, epoch)

                for ptr in range(0,max_images):
                    print("In the iteration ",ptr)
                    print("Starting",time.time()*1000.0)
                    
                    # Optimizing the G_A network
                    # fake B is an image
                    _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                    # print("fake B temp ", fake_B_temp)
                    # print("summary str ", summary_str)
                    writer.add_summary(summary_str, epoch*max_images + ptr)                    
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    
                    self.num_fake_inputs+=1
                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            imsave("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*16).astype(np.uint8))
            imsave("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*16).astype(np.uint8))
            imsave("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*16).astype(np.uint8))
            imsave("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*16).astype(np.uint8))
            imsave("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*16).astype(np.uint8))
            imsave("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*16).astype(np.uint8))
    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


def main():
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()