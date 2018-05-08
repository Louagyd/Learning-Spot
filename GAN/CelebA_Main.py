import numpy as np
import os
from PIL import Image
import sys
import matplotlib.pyplot as plt

import GAN
import CelebA_Visualization as clv

all_images = []
celeba_dir = 'Databases/img_align_celeba'
data_list = os.listdir(celeba_dir)
num_data = 200000
for iter, image_name in enumerate(data_list[0:num_data]):
    this_image = Image.open(celeba_dir + '/' + image_name)
    img = this_image.resize([64, 64], Image.ANTIALIAS)
    img = (np.asarray(img)/255 * 2) - 1
    all_images.append(img)
    sys.stdout.write("\r" + "reading data... " + str(np.floor(iter*100/num_data)) + "%")
    sys.stdout.flush()

celeba_gan = GAN.GAN(all_images, z_len = 100, z_sd = 1, init_sd = 0.02,
                     gen_filter_nums = [512, 256, 128, 64, 3],
                     gen_image_sizes = [4, 8, 16, 32, 64],
                     gen_kernel_size = [5,5],
                     gen_stride_size = [2,2],
                     dis_filter_nums = [64, 128, 256, 512],
                     dis_image_sizes = [32, 16, 8, 4],
                     dis_kernel_size = [5,5],
                     dis_stride_size = [2,2])

celeba_gan.train(num_epoch=15,batch_size = 32, batch_multiplication=1.1, lr_dis_init = 0.0005, lr_gen_init = 0.0005, lr_decay = 0.95, log_percent = 0.1,
                 verbose = 1, verbose_num_images=10, verbose_path='CelebA_Results/Training')
celeba_gan.save('Models/CelebA_DCGAN/Model')
