import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import shutil

def generate_images_using_numbers(GAN, numbers_list, verbose = 0, z_list=None):
    num_show_image = numbers_list.__len__()
    if z_list is None:
        z_init = np.random.normal(0, GAN.z_sd, size=num_show_image*(GAN.z_len))
        z_reshaped = np.reshape(z_init, [num_show_image, GAN.z_len])
        z_listed = [z_reshaped[i_,:] for i_ in range(num_show_image)]
    else:
        z_listed = z_list

    y_listed = []
    for gen_class in numbers_list:
        y_init = [0] * GAN.y_len
        y_init[gen_class] = 1
        y_init = np.asarray(y_init)
        y_listed.append(y_init)

    the_image_wow = GAN.generate_using_z(z_listed, y_listed)

    if verbose == 1:
        for i_ in range(num_show_image):
            plt.imshow(the_image_wow[i_][:,:,0])
            plt.show()

    return the_image_wow

def gallery(array, ncols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1,2)
        .reshape(height*nrows, width*ncols, intensity))
    return result

def images_table(GAN, table_size = 10, verbose = 0):
    results = np.ndarray(dtype=np.float32, shape=[10*table_size, GAN.image_dim[0], GAN.image_dim[1], GAN.image_dim[2]])
    ct = 0
    for number in range(10):
        for iter in range(table_size):
            the_image_wow = generate_images_using_numbers(GAN, [number])
            results[ct,:,:,:] = the_image_wow[0]
            ct += 1

    gal = gallery(results, ncols=table_size)
    if verbose == 1:
        plt.imshow(gal[:,:,0])
        plt.show()

    return gal

def translation_table_for_4_digits(GAN, d1, d2, d3, d4, table_size = 10, verbose = 0):
    results = np.ndarray(dtype=np.float32, shape=[table_size*table_size, GAN.image_dim[0], GAN.image_dim[1], GAN.image_dim[2]])
    ct = 0
    z_init = [np.random.normal(0, GAN.z_sd, size=GAN.z_len)]
    for i in range(table_size):
        for j in range(table_size):
            y_init = [0] * GAN.y_len
            t1 = i/table_size
            t2 = j/table_size
            y_init[d1] = (1-t1)*(1-t2)
            y_init[d2] = (1-t1)*t2
            y_init[d3] = t1*(1-t2)
            y_init[d4] = t1*t2
            y_init = [np.asarray(y_init)]
            the_image_wow = GAN.generate_using_z(z_init, y_init)
            results[ct,:,:,:] = the_image_wow[0]
            ct += 1
    gal = gallery(results, ncols=table_size)
    if verbose == 1:
        plt.imshow(gal[:,:,0])
        plt.show()

    return gal

def generate_translation_gif(GAN, d1, d2, z=None, num_frames = 50, fps = 5, path = 'MNIST_Results/Translation_Gif.gif'):
    temp = 'MNIST_Results/Temp_Folder'
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)
    if z is None:
        z_init = [np.random.normal(0, GAN.z_sd, size=GAN.z_len)]
    else:
        z_init = [z]

    images = []
    for iter in range(num_frames):
        y_init = [0] * GAN.y_len
        t = iter/num_frames
        y_init[d1] = 1-t
        y_init[d2] = t
        y_init = [np.asarray(y_init)]
        the_image = GAN.generate_using_z(z_init, y_init)
        plt.imsave(temp + '/img' + str(iter) + '.png', the_image[0][:,:,0])
        images.append(imageio.imread(temp + '/img' + str(iter) + '.png'))

    imageio.mimsave(path, images, fps = fps)
    shutil.rmtree(temp)

def generate_count_gif(GAN, num_frames = 200, fps = 20, path = 'MNIST_Results/Count_Gif.gif'):
    temp = 'MNIST_Results/Temp_Folder'
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)
    z_init = [np.random.normal(0, GAN.z_sd, size=GAN.z_len)]
    images = []
    each_frame = num_frames // 10
    for first_digit in range(9):
        for iter in range(each_frame):
            y_init = [0] * GAN.y_len
            t = iter/each_frame
            y_init[first_digit] = 1-t
            y_init[first_digit+1] = t
            y_init = [np.asarray(y_init)]
            the_image = GAN.generate_using_z(z_init, y_init)
            plt.imsave(temp + '/img' + str(iter) + '.png', the_image[0][:,:,0])
            images.append(imageio.imread(temp + '/img' + str(iter) + '.png'))
    imageio.mimsave(path, images, fps = fps)
    shutil.rmtree(temp)
