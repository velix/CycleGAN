import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import os
import random
import time

from model import build_generator_resnet_2blocks, build_gen_discriminator

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
summary_dir = "./output/2/exp_23"
batch_size = 1
pool_size = 500
max_images = 1000
save_training_images = False

EPOCHS = 51


class CycleGAN:

    def input_setup(self):
        '''
        This function basically setup variables for taking image input.
        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''

        filenames_A = tf.train.match_filenames_once("./input/mnist/*.jpg")
        self.queue_length_A = tf.size(filenames_A)
        filenames_B = tf.train.match_filenames_once("./input/SVHN/format2/*.jpg")
        self.queue_length_B = tf.size(filenames_B)

        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        image = tf.image.decode_jpeg(image_file_A)
        # image = tf.image.per_image_standardization(image)
        image = tf.image.resize_image_with_crop_or_pad(image, img_height, img_width)
        image = self._normalize_to_minus_plus_one(image)
        self.image_A = image

        image = tf.image.decode_jpeg(image_file_B)
        # image = tf.image.resize_image_with_crop_or_pad(image, img_height, img_width)
        # image = tf.image.per_image_standardization(image)
        image = self._normalize_to_minus_plus_one(image)
        self.image_B = image

    def _normalize_to_minus_plus_one(self, image_tensor):
        return tf.multiply(
                        2.0,
                        tf.divide(image_tensor - tf.reduce_min(image_tensor),
                                  tf.reduce_max(image_tensor) -
                                  tf.reduce_min(image_tensor))) - 1

    def input_read(self, sess):
        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_inputs_list/self.B_inputs_list -> Stores all the training images in python list
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

        self.A_inputs_list = np.zeros((max_images, batch_size, img_height, img_width, img_layerA))
        self.B_inputs_list = np.zeros((max_images, batch_size, img_height, img_width, img_layerB))

        for i in range(max_images):
            # after running the queue then ...
            image_tensor = sess.run(self.image_A)
            self.A_inputs_list[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layerA))

        print(np.min(self.A_inputs_list[-1]), np.max(self.A_inputs_list[-1]))

        for i in range(max_images):
            image_tensor = sess.run(self.image_B)
            self.B_inputs_list[i] = image_tensor.reshape((batch_size, img_height, img_width, img_layerB))

        print(np.min(self.B_inputs_list[-1]), np.max(self.B_inputs_list[-1]))
        print()

        # for the exception
        coord.request_stop()
        # Wait for all the threads to terminate.
        coord.join(threads)

    def model_setup(self):
            ''' This function sets up the model to train
            self.input_A_tensor/self.input_B_tensor -> Set of training images.
            self.fake_A/self.fake_B -> Generated images by corresponding
                                        generator of input_A_tensor and input_B_tensor
            self.lr -> Learning rate variable
            self.cyc_A/ self.cyc_B -> Images generated after feeding
                                    self.fake_A/self.fake_B to corresponding generator.
                                    This is use to calcualte cyclic loss
            '''

            self.input_A_tensor = tf.placeholder(
                                tf.float32,
                                [batch_size, img_width, img_height, img_layerA],
                                name="input_A_tensor"
                            )
            self.input_B_tensor = tf.placeholder(
                                tf.float32,
                                [batch_size, img_width, img_height, img_layerB],
                                name="input_B_tensor"
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
                self.fake_B = build_generator_resnet_2blocks(self.input_A_tensor, name="g_A")
                self.fake_A = build_generator_resnet_2blocks(self.input_B_tensor, name="g_B")

                self.rec_A = build_gen_discriminator(self.input_A_tensor, "d_A")
                self.rec_B = build_gen_discriminator(self.input_B_tensor, "d_B")

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

        disc_A_full_loss/disc_B_full_loss -> loss for discriminator A/B
        gen_A_full_loss/gen_B_full_loss -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions
        '''

        # Generator losses
        forward_cycle_cons = tf.reduce_mean(tf.abs(self.input_A_tensor-self.cyc_A))
        backward_cycle_cons = tf.reduce_mean(tf.abs(self.input_B_tensor-self.cyc_B))
        cyc_loss = forward_cycle_cons + backward_cycle_cons

        gan_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A, 1))
        gan_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B, 1))

        gen_A_full_loss = cyc_loss*10 + gan_loss_B
        gen_B_full_loss = cyc_loss*10 + gan_loss_A

        # Discriminator losses
        real_A_recognition_loss = tf.reduce_mean(tf.squared_difference(
                                                    self.rec_A, 1))
        fake_A_recognition_loss = tf.reduce_mean(tf.square(
                                                    self.fake_pool_rec_A))

        disc_A_full_loss = (fake_A_recognition_loss + real_A_recognition_loss)/2.0

        real_B_recognition_loss = tf.reduce_mean(tf.squared_difference(
                                                    self.rec_B, 1))
        fake_B_recognition_loss = tf.reduce_mean(tf.square(
                                                    self.fake_pool_rec_B))

        disc_B_full_loss = (fake_B_recognition_loss + real_B_recognition_loss)/2.0

        adam_opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        sgd_opt = tf.train.GradientDescentOptimizer(1e-3)
        # Returns all variables created with trainable=True
        # A list of Variable objects.
        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

        self.d_A_trainer = adam_opt.minimize(disc_A_full_loss, var_list=d_A_vars)
        self.d_B_trainer = adam_opt.minimize(disc_B_full_loss, var_list=d_B_vars)
        self.g_A_trainer = adam_opt.minimize(gen_A_full_loss, var_list=g_A_vars)
        self.g_B_trainer = adam_opt.minimize(gen_B_full_loss, var_list=g_B_vars)

        # Summary variables for tensorboard
        self.gen_A_full_loss_summ = tf.summary.scalar("g_A_loss", gen_A_full_loss)
        self.gan_loss_A_summ = tf.summary.scalar("gan_loss_A", gan_loss_A)
        self.gen_A_loss_summ = tf.summary.merge([
                                            self.gen_A_full_loss_summ,
                                            self.gan_loss_A_summ])

        self.gen_B_full_loss_summ = tf.summary.scalar("g_B_loss", gen_B_full_loss)
        self.gan_loss_B_summ = tf.summary.scalar("gan_loss_B", gan_loss_B)
        self.gen_B_loss_summ = tf.summary.merge([
                                            self.gen_B_full_loss_summ,
                                            self.gan_loss_B_summ])

        self.cyc_loss_summ = tf.summary.scalar("cycle_loss", cyc_loss)

        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", disc_A_full_loss)
        self.d_A_real_loss_summ = tf.summary.scalar("d_A_real_recognition_loss", real_A_recognition_loss)
        self.d_A_fake_loss_summ = tf.summary.scalar("d_A_fake_recognition_loss", fake_A_recognition_loss)
        self.d_A_losses = tf.summary.merge([self.d_A_loss_summ,
                                            self.d_A_real_loss_summ,
                                            self.d_A_fake_loss_summ])

        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", disc_B_full_loss)
        self.d_B_real_loss_summ = tf.summary.scalar("d_B_real_recognition_loss", real_B_recognition_loss)
        self.d_B_fake_loss_summ = tf.summary.scalar("d_B_fake_recognition_loss", fake_B_recognition_loss)
        self.d_B_losses = tf.summary.merge([self.d_B_loss_summ,
                                            self.d_B_real_loss_summ,
                                            self.d_B_fake_loss_summ])

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
                if(epoch < int(np.round(EPOCHS/2))):
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100
                if(save_training_images):
                    self.save_training_images(sess, epoch)

                for ptr in range(0, max_images):
                    if ptr % 100 == 0:
                        print("In the iteration ", ptr)
                    iteration_start = time.time()*1000.0

                    # Optimizing the G_A network
                    _, fake_B_temp, gen_A_loss_str, cyc_loss = sess.run(
                        [self.g_A_trainer, self.fake_B, self.gen_A_loss_summ, self.cyc_loss_summ],
                        feed_dict={
                                self.input_A_tensor: self.A_inputs_list[ptr],
                                self.input_B_tensor: self.B_inputs_list[ptr],
                                self.lr: curr_lr
                            })

                    writer.add_summary(gen_A_loss_str, epoch*max_images + ptr)
                    writer.add_summary(cyc_loss, epoch*max_images + ptr)

                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)

                    # Optimizing the D_B network
                    _, summary_str = sess.run(
                        [self.d_B_trainer, self.d_B_losses],
                        feed_dict={
                                self.input_A_tensor: self.A_inputs_list[ptr],
                                self.input_B_tensor: self.B_inputs_list[ptr],
                                self.lr: curr_lr,
                                self.fake_pool_B: fake_B_temp1
                            })
                    writer.add_summary(summary_str, epoch*max_images + ptr)

                    # Optimizing the G_B network
                    _, fake_A_temp, gen_B_loss_str, cyc_loss = sess.run(
                        [self.g_B_trainer, self.fake_A, self.gen_B_loss_summ, self.cyc_loss_summ],
                        feed_dict={
                                self.input_A_tensor: self.A_inputs_list[ptr],
                                self.input_B_tensor: self.B_inputs_list[ptr],
                                self.lr: curr_lr
                            })
                    writer.add_summary(gen_B_loss_str, epoch*max_images + ptr)
                    writer.add_summary(cyc_loss, epoch*max_images + ptr)

                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run(
                        [self.d_A_trainer, self.d_A_losses],
                        feed_dict={
                                self.input_A_tensor: self.A_inputs_list[ptr],
                                self.input_B_tensor: self.B_inputs_list[ptr],
                                self.lr: curr_lr,
                                self.fake_pool_A: fake_A_temp1
                            })

                    writer.add_summary(summary_str, epoch*max_images + ptr)

                    self.num_fake_inputs += 1

                    iteration_end = time.time()*1000.0
                    # print('\ttime: {}'.format(iteration_end-iteration_start))

                # self.save_training_images(sess, epoch)
                if epoch == 0 or epoch % 10 == 0:
                    self._store_image_summaries(writer, sess, epoch)

        sess.run(tf.assign(self.global_step, epoch + 1))

    def _store_image_summaries(self, writer, sess, epoch, ptr=99):

        for i in range(ptr-5, ptr):
            input_A = self.A_inputs_list[i]
            input_B = self.B_inputs_list[i]
            fake_A, fake_B, cyc_A, cyc_B = sess.run(
                [self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                feed_dict={self.input_A_tensor: input_A,
                           self.input_B_tensor: input_B})
            if epoch == 0:
                input_A_tensor_summ = tf.summary.image(
                                            'input_A_tensor_i:{}'.format(i),
                                            tf.convert_to_tensor(input_A),
                                            max_outputs=1)

                input_B_tensor_summ = tf.summary.image(
                                            'input_B_tensor_i:{}'.format(i),
                                            tf.convert_to_tensor(input_B),
                                            max_outputs=1)

                writer.add_summary(input_A_tensor_summ.eval(), epoch)
                writer.add_summary(input_B_tensor_summ.eval(), epoch)

            fake_A_summ = tf.summary.image(
                                        'fake_domain_A_i:{}'.format(i),
                                        fake_A,
                                        max_outputs=1)

            fake_B_summ = tf.summary.image(
                                        'fake_domain_B_i:{}'.format(i),
                                        fake_B,
                                        max_outputs=1)

            cyc_A_summ = tf.summary.image(
                                        'cyc_domain_A_i:{}'.format(i),
                                        cyc_A,
                                        max_outputs=1)

            cyc_B_summ = tf.summary.image(
                                        'cyc_domain_B_i:{}'.format(i),
                                        cyc_B,
                                        max_outputs=1)

            images_summ = tf.summary.merge([fake_A_summ.eval(),
                                            fake_B_summ.eval(),
                                            cyc_A_summ.eval(),
                                            cyc_B_summ.eval()
                                            ])

            writer.add_summary(images_summ.eval(), epoch)

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0, 10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([
                self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],
                feed_dict={self.input_A_tensor: self.A_inputs_list[i],
                           self.input_B_tensor: self.B_inputs_list[i]})

            imsave("./output/imgs/fake_in_domain_A_" + str(epoch) + "_" + str(i)+".jpg",
                   ((fake_A_temp[0])).astype(np.uint8))

            imsave("./output/imgs/fake_in_domain_B_" + str(epoch) + "_" + str(i)+".jpg",
                   ((fake_B_temp[0])).astype(np.uint8))

            imsave("./output/imgs/cycA_" + str(epoch) + "_" + str(i)+".jpg",
                   ((cyc_A_temp[0])).astype(np.uint8))

            imsave("./output/imgs/cycB_" + str(epoch) + "_" + str(i)+".jpg",
                   ((cyc_B_temp[0])).astype(np.uint8))

            imsave("./output/imgs/inputA_" + str(epoch) + "_" + str(i)+".jpg",
                   ((self.A_inputs_list[i][0])).astype(np.uint8))

            imsave("./output/imgs/inputB_" + str(epoch) + "_" + str(i)+".jpg",
                   ((self.B_inputs_list[i][0])).astype(np.uint8))

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
