import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import random
import time

from model import (build_generator_resnet_1block, build_gen_discriminator,
                   build_generator_resnet_2blocks)

img_height = 28
img_width = 28
img_layerA = 3
img_layerB = 3
img_size = img_height * img_width

to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"
summary_dir = "./output/2/exp_6"
batch_size = 1
pool_size = 50
max_images = 100
save_training_images = True

EPOCHS = 100


class CycleGAN:

    def input_setup(self):
        '''
        This function basically setup variables for taking image input.
        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''

        filenames_A = tf.train.match_filenames_once("./input/mnist/*.jpg")
        self.queue_length_A = tf.size(filenames_A)
        filenames_B = tf.train.match_filenames_once("./input/colorful_mnist/*.jpg")
        self.queue_length_B = tf.size(filenames_B)

        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        image_A = tf.image.decode_jpeg(image_file_A)
        # image_A = tf.image.per_image_standardization(image_A)
        self.image_A = image_A
        self.image_A = tf.subtract(tf.div(self.image_A, 14), 1)

        image_B = tf.image.decode_jpeg(image_file_B)
        # image_B = tf.image.per_image_standardization(image_B)
        self.image_B = image_B
        self.image_B = tf.subtract(tf.div(self.image_B, 14), 1)

    def input_read(self, sess):
        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        '''

        # Loading images into the tensors
        # This class implements a simple mechanism to coordinate
        # the termination of a set of threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        # run the queue
        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)
        # it's a pool size of 50
        self.fake_images_A = np.zeros((pool_size, 1, img_height, img_width, img_layerA))
        self.fake_images_B = np.zeros((pool_size, 1, img_height, img_width, img_layerB))

        self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layerA))
        self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layerB))

        for i in range(max_images):
            # after running the queue then ...
            image_tensor = sess.run(self.image_A)
            self.A_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layerA))

        print(np.mean(self.A_input[0]))
        print(np.max(self.A_input[0]))
        print(np.min(self.A_input[0]))
        print()

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            self.B_input[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layerB))

        print(np.mean(self.B_input[0]))
        print(np.max(self.B_input[0]))
        print(np.min(self.B_input[0]))
        print()

        # for the exception
        coord.request_stop()
        # Wait for all the threads to terminate.
        coord.join(threads)

    def model_setup(self):
            ''' This function sets up the model to train
            self.input_A/self.input_B -> Set of training images.
            self.fake_A/self.fake_B -> Generated images by corresponding
                                        generator of input_A and input_B
            self.lr -> Learning rate variable
            self.cyc_A/ self.cyc_B -> Images generated after feeding
                                    self.fake_A/self.fake_B to corresponding generator.
                                    This is use to calcualte cyclic loss
            '''

            self.input_A = tf.placeholder(
                                tf.float32,
                                [batch_size, img_width, img_height, img_layerA],
                                name="input_A"
                            )
            self.input_B = tf.placeholder(
                                tf.float32,
                                [batch_size, img_width, img_height, img_layerB],
                                name="input_B"
                            )

            self.fake_pool_A = tf.placeholder(
                                tf.float32,
                                [None, img_width, img_height, img_layerA],
                                name="fake_pool_A"
                            )
            self.fake_pool_B = tf.placeholder(
                                tf.float32,
                                [None, img_width, img_height, img_layerB],
                                name="fake_pool_B"
                            )

            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            self.num_fake_inputs = 0

            self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

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
    
    def loss_calc(self):
        '''
        In this function we are defining the variables for
        loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        '''

        # Generator losses
        diff_from_original_A = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A))
        diff_from_original_B = tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        cyc_loss = diff_from_original_A + diff_from_original_B

        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A, 1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B, 1))

        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        # Discriminator losses
        real_A_recognition_loss = tf.reduce_mean(tf.squared_difference(
                                                    self.rec_A, 1))
        fake_A_recognition_loss = tf.reduce_mean(tf.square(
                                                    self.fake_pool_rec_A))

        d_loss_A = (fake_A_recognition_loss + real_A_recognition_loss)/2.0

        real_B_recognition_loss = tf.reduce_mean(tf.squared_difference(
                                                    self.rec_B, 1))
        fake_B_recognition_loss = tf.reduce_mean(tf.square(
                                                    self.fake_pool_rec_B))

        d_loss_B = (fake_B_recognition_loss + real_B_recognition_loss)/2.0

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

        # Summary variables for tensorboard
        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_A_real_loss_summ = tf.summary.scalar("d_A_real_recognition_loss", real_A_recognition_loss)
        self.d_A_fake_loss_summ = tf.summary.scalar("d_A_fake_recognition_loss", fake_A_recognition_loss)

        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)
        self.d_B_real_loss_summ = tf.summary.scalar("d_B_real_recognition_loss", real_B_recognition_loss)
        self.d_B_fake_loss_summ = tf.summary.scalar("d_B_fake_recognition_loss", fake_B_recognition_loss)

    def train(self):
        '''
        Training Function
        '''

        # Load Dataset from the dataset folder
        self.input_setup()
        # Build the network
        self.model_setup()
        # Loss function calculations
        self.loss_calc()
        init = (tf.global_variables_initializer(),
                tf.local_variables_initializer())

        # Saves and restores variables.
        saver = tf.train.Saver()
        with tf.Session() as sess:

            # run the variables first
            sess.run(init)

            # Read input to nd array
            self.input_read(sess)
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            # Writes Summaries to event files
            writer = tf.summary.FileWriter(summary_dir)
            writer.add_graph(sess.graph)

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), EPOCHS):
                print("In the epoch ", epoch)
                saver.save(sess, os.path.join(check_dir, "cyclegan"), global_step=epoch)

                # Learning rate decay
                if(epoch < 100):
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100
                if(save_training_images):
                    self.save_training_images(sess, epoch)

                for ptr in range(0, max_images):
                    print("In the iteration ", ptr)
                    iteration_start = time.time()*1000.0

                    # Optimizing the G_A network
                    _, fake_B_temp, summary_str = sess.run(
                        [self.g_A_trainer, self.fake_B, self.g_A_loss_summ],
                        feed_dict={
                                self.input_A: self.A_input[ptr],
                                self.input_B: self.B_input[ptr],
                                self.lr: curr_lr
                            })
                    # print("fake B temp ", fake_B_temp)
                    # print("summary str ", summary_str)
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str, fake_recogn_loss, real_recogn_loss = sess.run(
                        [self.d_B_trainer, self.d_B_loss_summ,
                         self.d_B_fake_loss_summ, self.d_B_real_loss_summ],
                        feed_dict={
                                self.input_A: self.A_input[ptr],
                                self.input_B: self.B_input[ptr],
                                self.lr: curr_lr,
                                self.fake_pool_B: fake_B_temp1
                            })
                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    writer.add_summary(fake_recogn_loss, epoch*max_images + ptr)
                    writer.add_summary(real_recogn_loss, epoch*max_images + ptr)

                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run(
                        [self.g_B_trainer, self.fake_A, self.g_B_loss_summ],
                        feed_dict={
                                self.input_A: self.A_input[ptr],
                                self.input_B: self.B_input[ptr],
                                self.lr: curr_lr
                            })
                    writer.add_summary(summary_str, epoch*max_images + ptr)

                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str, fake_recogn_loss, real_recogn_loss = sess.run(
                        [self.d_A_trainer, self.d_A_loss_summ,
                         self.d_A_fake_loss_summ, self.d_A_real_loss_summ],
                        feed_dict={
                                self.input_A: self.A_input[ptr],
                                self.input_B: self.B_input[ptr],
                                self.lr: curr_lr,
                                self.fake_pool_A: fake_A_temp1
                            })

                    writer.add_summary(summary_str, epoch*max_images + ptr)
                    writer.add_summary(fake_recogn_loss, epoch*max_images + ptr)
                    writer.add_summary(real_recogn_loss, epoch*max_images + ptr)

                    self.num_fake_inputs += 1

                    # if ptr % 99 == 0:
                    #     input_A_summary = tf.summary.image("input_A", self.A_input[ptr], max_outputs=1)
                    #     input_B_summary = tf.summary.image("input_B", self.B_input[ptr], max_outputs=1)
                    #     fake_A_summary = tf.summary.image("fake_A", fake_A_temp1, max_outputs=1)
                    #     fake_B_summary = tf.summary.image("fake_B", fake_B_temp1, max_outputs=1)

                    #     writer.add_summary(input_A_summary, epoch*max_images + ptr)
                    #     writer.add_summary(input_B_summary, epoch*max_images + ptr)
                    #     writer.add_summary(fake_A_summary, epoch*max_images + ptr)
                    #     writer.add_summary(fake_B_summary, epoch*max_images + ptr)

                    iteration_end = time.time()*1000.0
                    print('\ttime: {}'.format(iteration_end-iteration_start))

                sess.run(tf.assign(self.global_step, epoch + 1))

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
        '''
        This function saves the generated image to
            corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full
            and then randomly selects an
        already stored image and replace it with new one.
        '''

        if(num_fakes < pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else:
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else:
                return fake


def main():
    model = CycleGAN()
    if to_train:
        model.train()
    elif to_test:
        model.test()


if __name__ == '__main__':
    main()
