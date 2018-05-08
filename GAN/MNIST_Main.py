import numpy as np
import pandas as pnd
from PIL import Image
import matplotlib.pyplot as plt

import GAN
import MNIST_Visualization as mnv

mnist_train = np.array(pnd.read_csv("Databases/MNIST/mnist_train.csv"))
mnist_test = np.array(pnd.read_csv("Databases/MNIST/mnist_test.csv"))

train_images = mnist_train[:,1:785]/255
train_images = train_images.reshape((-1, 28, 28, 1))

train_targets = np.reshape(mnist_train[:,0], [-1, 1])
targets_one_hot = list(np.eye(10)[np.int32(mnist_train[:,0])])

images_mnist = [train_images[i,:,:,:] for i in range(train_images.shape[0])]
images_mnist = [np.asarray(Image.fromarray(images_mnist[i][:,:,0]).resize([32, 32], Image.ANTIALIAS)) for i in range(images_mnist.__len__())]
images_mnist = [np.reshape(images_mnist[i], [32, 32, 1]) for i in range(images_mnist.__len__())]
mnist_gan = GAN.GAN(images_mnist, targets=targets_one_hot, z_len = 100, z_sd = 1, init_sd=0.02,
                    gen_filter_nums = [256, 128, 64, 1],
                    gen_image_sizes = [4, 8, 16, 32],
                    gen_kernel_size = [4,4],
                    gen_stride_size = [2,2],
                    dis_filter_nums = [8, 16, 64, 128, 256],
                    dis_image_sizes = [16, 8, 4, 2, 1],
                    dis_kernel_size = [4,4],
                    dis_stride_size = [2,2])
mnist_gan.train(num_epoch = 20, verbose = 1, verbose_num_images = 5, verbose_path = 'MNIST_Results/Training',
                batch_size = 16, lr_dis_init = 0.0005, lr_gen_init = 0.0005, lr_decay = 0.97, log_percent = 0.1, gen_dis_ratio = 1)
mnist_gan.save(path='Models/MNIST_cGAN/Model')
mnist_gan.load(path='Models/MNIST_cGAN/Model')
generated_images = mnv.generate_images_using_numbers(mnist_gan, [1,5,4,7,9], verbose=1)
gallery = mnv.images_table(mnist_gan, verbose=1)
plt.imsave('MNIST_Results/generated_numbers_table.png', gallery[:,:,0])
translation_table = mnv.translation_table_for_4_digits(mnist_gan, 1, 2, 3, 4, verbose=1)
plt.imsave('MNIST_Results/generated_translation_table.png', translation_table[:,:,0])
mnv.generate_translation_gif(mnist_gan, 4, 7, path='MNIST_Results/Translation_Gif.gif')
mnv.generate_count_gif(mnist_gan, path='MNIST_Results/Count_Gif.gif')
