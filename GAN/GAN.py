import tensorflow as tf
import numpy as np
from random import shuffle
from random import randint


class GAN():
    def __init__(self, images, targets = None, z_len = 100, z_sd = 1, init_sd = 0.02,
                 gen_filter_nums = [512, 256, 128, 64, 3],
                 gen_image_sizes = [4, 8, 16, 32, 64],
                 gen_kernel_size = [5,5],
                 gen_stride_size = [2,2],
                 dis_filter_nums = [64, 128, 256, 512],
                 dis_image_sizes = [32, 16, 8, 4],
                 dis_kernel_size = [5,5],
                 dis_stride_size = [2,2]):

        if targets:
            zipped = list(zip(images, targets))
            shuffle(zipped)
            self.images, self.targets = zip(*zipped)
            self.y_len = targets[0].__len__()
        else:
            shuffle(images)
            self.images = images
            self.targets = None
            self.y_len = 0

        self.num_images = images.__len__()

        self.z_len = z_len
        self.z_sd = z_sd

        self.init_sd = init_sd
        if init_sd is not None:
            self.kernel_initializer = tf.random_normal_initializer(stddev=init_sd)
            self.gamma_initializer = tf.random_normal_initializer(1.0, init_sd)
        else:
            self.kernel_initializer = None
            self.gamma_initializer = None

        self.gen_filter_nums = gen_filter_nums
        self.gen_image_sizes = gen_image_sizes
        self.gen_kernel_size = gen_kernel_size
        self.gen_stride_size = gen_stride_size

        self.dis_filter_nums = dis_filter_nums
        self.dis_image_sizes = dis_image_sizes
        self.dis_kernel_size = dis_kernel_size
        self.dis_stride_size = dis_stride_size

        self.num_gen_layers = len(gen_filter_nums)
        self.num_dis_layers = len(dis_filter_nums)

        self.image_dim = [gen_image_sizes[-1], gen_image_sizes[-1], gen_filter_nums[-1]]
        tf.reset_default_graph()

        self.z_input_ph = tf.placeholder(tf.float32, [None, self.z_len], name='z_input')
        self.images_ph = tf.placeholder(tf.float32, [None, self.gen_image_sizes[-1], self.gen_image_sizes[-1], self.gen_filter_nums[-1]], name='input_images')
        if targets:
            self.targets_ph = tf.placeholder(tf.float32, [None, self.y_len], name='targets')
            self.y_input_ph = tf.placeholder(tf.float32, [None, self.y_len], name='y_input')
            self.gen_logits, self.gen_images = self.generate(self.z_input_ph, self.z_sd * self.y_input_ph, is_train=True, reuse=False)
            self.real_logits, self.real_probs = self.discriminate(self.images_ph, self.targets_ph, is_train=True, reuse=False)
            self.fake_logits, self.fake_probs = self.discriminate(self.gen_images, self.y_input_ph, is_train=True, reuse=True)
            self.gen_logits_eval, self.gen_images_eval = self.generate(self.z_input_ph, self.z_sd * self.y_input_ph, is_train=False, reuse=True)
        else:
            self.targets_ph = None
            self.y_input_ph = None
            self.gen_logits, self.gen_images = self.generate(self.z_input_ph, None, is_train=True, reuse=False)
            self.real_logits, self.real_probs = self.discriminate(self.images_ph, None, is_train=True, reuse=False)
            self.fake_logits, self.fake_probs = self.discriminate(self.gen_images, None, is_train=True, reuse=True)
            self.gen_logits_eval, self.gen_images_eval = self.generate(self.z_input_ph, None, is_train=False, reuse=True)

        self.dis_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.real_logits, labels=tf.ones_like(self.real_logits)))
        self.dis_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.zeros_like(self.fake_logits)))
        self.dis_loss = self.dis_loss_fake + self.dis_loss_real

        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.fake_logits, labels=tf.ones_like(self.fake_logits)))

        self.dis_gstep = tf.Variable(0, trainable=False)
        self.gen_gstep = tf.Variable(0, trainable=False)

        self.sess = tf.Session()

    def get_batch(self, index, batch_size, gen_class = None):
        batch_images = self.images[index:(index+batch_size)]
        batch_images = np.concatenate([arr[np.newaxis] for arr in batch_images])
        if self.targets:
            batch_targets = self.targets[index:(index+batch_size)]
            batch_targets = np.concatenate([arr[np.newaxis] for arr in batch_targets])
        else:
            batch_targets = None

        if gen_class:
            z_init = np.random.normal(0, self.z_sd, size=batch_size*(self.z_len))
            z_reshaped = np.reshape(z_init, [batch_size, self.z_len])
            y_init = [0] * (batch_size*self.y_len)
            y_reshaped = np.reshape(y_init, [batch_size, self.y_len])
            y_reshaped[:,gen_class] = 1
        else:
            z_init = np.random.normal(0, self.z_sd, size=batch_size*(self.z_len))
            z_reshaped = np.reshape(z_init, [batch_size, self.z_len])
            if self.targets:
                y_init = [randint(0,self.y_len-1) for _ in range(batch_size)]
                y_reshaped = np.eye(self.y_len)[y_init]
            else:
                y_reshaped = None
        return batch_images, batch_targets, z_reshaped, y_reshaped

    def train(self, num_epoch = 25, batch_size = 64, batch_multiplication = 1, lr_dis_init = 0.0005, lr_gen_init = 0.0005, lr_decay = 0.95, log_percent = 0.1, gen_dis_ratio = 1, verbose = 0, verbose_num_images = 3, verbose_path = 'Results'):
        learning_rate_dis = tf.train.exponential_decay(lr_dis_init, self.dis_gstep*batch_size, self.num_images, lr_decay, staircase = True)
        learning_rate_gen = tf.train.exponential_decay(lr_gen_init, self.dis_gstep*batch_size, self.num_images, lr_decay, staircase = True)
        dis_opt_real = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=0.5).minimize(self.dis_loss_real, var_list=self.dis_vars, global_step=self.dis_gstep)
        dis_opt_fake = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=0.5).minimize(self.dis_loss_fake, var_list=self.dis_vars, global_step=self.dis_gstep)
        dis_opt = tf.train.AdamOptimizer(learning_rate=learning_rate_dis, beta1=0.5).minimize(self.dis_loss, var_list=self.dis_vars, global_step=self.dis_gstep)
        gen_opt = tf.train.AdamOptimizer(learning_rate=learning_rate_gen, beta1=0.5).minimize(self.gen_loss, var_list=self.gen_vars, global_step=self.gen_gstep)

        train_get = self.get_batch(0, int(np.floor(self.num_images * log_percent / 100)))
        train_get_images, train_get_targets, train_get_z, train_get_y = train_get
        train_dict = {self.images_ph: train_get_images,
                      self.z_input_ph: train_get_z}
        if self.targets:
            train_dict[self.targets_ph] = train_get_targets
            train_dict[self.y_input_ph] = train_get_y

        self.sess.run(tf.global_variables_initializer())
        if verbose == 1:
            import matplotlib.pyplot as plt
            import sys
            import os
            import shutil
            import imageio
            if os.path.exists(verbose_path):
                shutil.rmtree(verbose_path)
            os.makedirs(verbose_path)

            if self.targets:
                z_init = np.random.normal(0, self.z_sd, size=verbose_num_images*self.y_len*(self.z_len))
                z_reshaped = np.reshape(z_init, [verbose_num_images*self.y_len, self.z_len])
                z_listed = [z_reshaped[i_,:] for i_ in range(verbose_num_images*self.y_len)]
                y_listed = []
                for verbose_class in range(self.y_len):
                    for iter in range(verbose_num_images):
                        y_init = [0] * self.y_len
                        y_init[verbose_class] = 1
                        y_listed.append(np.asarray(y_init))
            else:
                z_init = np.random.normal(0, self.z_sd, size=verbose_num_images*(self.z_len))
                z_reshaped = np.reshape(z_init, [verbose_num_images, self.z_len])
                z_listed = [z_reshaped[i_,:] for i_ in range(verbose_num_images)]
                y_listed = None
        images_for_gif = []
        for epoch in range(num_epoch):
            if verbose == 1:
                the_image_wow = self.generate_using_z(z_listed, y_listed)
                images_to_show = np.concatenate([arr[np.newaxis] for arr in the_image_wow])
                images_gallery = self.gallery(images_to_show, ncols=verbose_num_images)
                if self.image_dim[2] == 1:
                    plt.imshow(images_gallery[:,:,0])
                    plt.show()
                    plt.imsave(verbose_path + '/Epoch' + str(epoch) + '.png', images_gallery[:,:,0])
                    images_for_gif.append(imageio.imread(verbose_path + '/Epoch' + str(epoch) + '.png'))
                else:
                    plt.imshow(images_gallery)
                    plt.show()
                    plt.imsave(verbose_path + '/Epoch' + str(epoch) + '.png', images_gallery)
                    images_for_gif.append(imageio.imread(verbose_path + '/Epoch' + str(epoch) + '.png'))

            dis_loss_real = self.sess.run(self.dis_loss_real, feed_dict=train_dict)
            dis_loss_fake = self.sess.run(self.dis_loss_fake, feed_dict=train_dict)
            gen_loss = self.sess.run(self.gen_loss, feed_dict=train_dict)
            misclass_ratio = self.compute_misclassified_ratio(train_get)
            sys.stdout.write("\r" + "EPOCH: " + str(epoch)
                             + "    Discriminator Loss Real: " + str(dis_loss_real)
                             + "    Discriminator Loss Fake: " + str(dis_loss_fake)
                             + "    Generator Loss: " + str(gen_loss)
                             + "    MisClassified Ratio: " + str(misclass_ratio)
                             + "\n")
            for iter, i in enumerate(range(0, self.num_images - batch_size, batch_size)):

                batch_images, batch_targets, batch_z, batch_y = self.get_batch(i, batch_size)
                batch_dict = {self.images_ph: batch_images,
                              self.z_input_ph: batch_z}
                if self.targets:
                    batch_dict[self.targets_ph] = batch_targets
                    batch_dict[self.y_input_ph] = batch_y

                self.sess.run(dis_opt_real, feed_dict=batch_dict)
                self.sess.run(dis_opt_fake, feed_dict=batch_dict)
                # self.sess.run(dis_opt, feed_dict=batch_dict)
                sys.stdout.write("\r" + "processing... " + str(np.floor(i*100/self.num_images)) + "%")
                sys.stdout.flush()
                for _ in range(gen_dis_ratio):
                    self.sess.run(gen_opt, feed_dict=batch_dict)
            sys.stdout.write("\r" + "processing... 100%")
            sys.stdout.flush()
            batch_size = int(batch_size * batch_multiplication)

        if verbose == 1:
            imageio.mimsave(verbose_path + str('/Training.gif'), images_for_gif, fps = 3)

    def compute_misclassified_ratio(self, batch_get):
        batch_images, batch_targets, batch_z, batch_y = batch_get
        batch_dict = {self.images_ph: batch_images,
                      self.z_input_ph: batch_z}
        if self.targets:
            batch_dict[self.targets_ph] = batch_targets
            batch_dict[self.y_input_ph] = batch_y

        batch_real_probs = self.sess.run(self.real_probs, feed_dict=batch_dict)
        batch_fake_probs = self.sess.run(self.fake_probs, feed_dict=batch_dict)
        real_sum = np.sum(np.round(1 - batch_real_probs))
        fake_sum = np.sum(np.round(batch_fake_probs))
        total_sum = real_sum + fake_sum
        num_batch = batch_images.shape[0] + batch_z.shape[0]

        return total_sum/num_batch

    def generate_using_z(self, z_list, y_list = None):
        z_stacked = np.concatenate([arr[np.newaxis] for arr in z_list])
        the_dict = {self.z_input_ph:z_stacked}
        if y_list is not None:
            y_stacked = np.concatenate([arr[np.newaxis] for arr in y_list])
            the_dict[self.y_input_ph] = y_stacked

        images_generated = self.sess.run(self.gen_images_eval, feed_dict=the_dict)
        images_list = [images_generated[i,:,:,:] for i in range(z_list.__len__())]
        return images_list

    def gallery(self, array, ncols):
        nindex, height, width, intensity = array.shape
        nrows = nindex//ncols
        assert nindex == nrows*ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        result = (array.reshape(nrows, ncols, height, width, intensity)
            .swapaxes(1,2)
            .reshape(height*nrows, width*ncols, intensity))
        return result

    def leaky_relu(self, x, leakiness=0.2, name = ''):
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name = 'leaky_relu'+name)

    def generate(self, z_input, y_input, is_train = True, reuse = False):
        with tf.variable_scope('generator', reuse = reuse):
            if y_input is not None:
                z_concat = tf.concat((z_input, y_input), axis = 1)
            else:
                z_concat = z_input

            z_projected = tf.layers.dense(z_concat, self.gen_filter_nums[0]*self.gen_image_sizes[0]*self.gen_image_sizes[0], kernel_initializer=self.kernel_initializer)
            z_reshaped = tf.reshape(z_projected, [-1, self.gen_image_sizes[0], self.gen_image_sizes[0], self.gen_filter_nums[0]])
            z_last = self.leaky_relu(tf.layers.batch_normalization(z_reshaped, training=is_train, gamma_initializer=self.gamma_initializer), leakiness=0.2, name='z_last_init')
            for iter, num_filters in enumerate(self.gen_filter_nums[1:-1]):
                z_last = tf.layers.conv2d_transpose(z_last, num_filters, self.gen_kernel_size, strides=self.gen_stride_size, padding='SAME', name='conv_transpose' + str(iter), kernel_initializer=self.kernel_initializer)
                z_last = self.leaky_relu(tf.layers.batch_normalization(z_last, training=is_train, gamma_initializer=self.gamma_initializer), leakiness=0.2, name= str(iter))
            z_last = tf.layers.conv2d_transpose(z_last, self.gen_filter_nums[-1], self.gen_kernel_size, strides=self.gen_stride_size, padding='SAME', name='generated_logits', kernel_initializer=self.kernel_initializer)
            if is_train:
                generated_images = tf.nn.tanh(z_last)
            else:
                generated_images = tf.nn.sigmoid(z_last)

        self.gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return z_last, generated_images

    def discriminate(self, images, y_input, is_train = True, reuse = False):
        with tf.variable_scope('discriminator', reuse = reuse):
            if y_input is not None:
                y_input_fill_first = tf.reshape(y_input, [-1, 1, 1, self.y_len])
                tempo = tf.ones_like(tf.reshape(tf.concat([y_input for _ in range(images.get_shape()[1] * images.get_shape()[2])], axis=1), [-1, images.get_shape()[1], images.get_shape()[2], self.y_len]))
                y_input_fill = y_input_fill_first * tempo
                images = tf.concat((images, y_input_fill), axis=3)
            for iter, num_filters in enumerate(self.dis_filter_nums):
                images = tf.layers.conv2d(images, filters=num_filters, kernel_size=self.dis_kernel_size, strides=self.dis_stride_size, padding='SAME', name='conv2d' + str(iter), kernel_initializer=self.kernel_initializer)
                images = self.leaky_relu(tf.layers.batch_normalization(images, training=is_train, gamma_initializer=self.gamma_initializer), leakiness=0.2, name=str(iter))
            image_flattened = tf.reshape(images, [-1, images.get_shape()[1] * images.get_shape()[2] * images.get_shape()[3]])
            output_logits = tf.layers.dense(image_flattened, 1, name='discriminated_logits', kernel_initializer=self.kernel_initializer)
            output = tf.nn.sigmoid(output_logits, name='discriminated_prob')

        self.dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return output_logits, output

    def save(self, path = 'Models/Model'):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        print('\nModel has been saved!')

    def load(self, path = 'Models/Model'):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        print('\nModel loaded successfully')








