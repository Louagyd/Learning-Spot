import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.colorchooser import *
from PIL import Image, ImageTk
import os
import CelebA_Visualization_Form as form
import shutil
import imageio

def gallery(array, ncols):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1,2)
        .reshape(height*nrows, width*ncols, intensity))
    return result

def generate_face(GAN, num_faces, z_list=None, verbose = 0, verbose_gallery = None, return_gallery = False):

    if z_list is None:
        z_init = np.random.normal(0, GAN.z_sd, size=num_faces*GAN.z_len)
        z_reshaped = np.reshape(z_init, newshape=[num_faces, GAN.z_len])
        real_z_list = [z_reshaped[i,:] for i in range(num_faces)]
    else:
        real_z_list = z_list

    images_generated = GAN.generate_using_z(real_z_list, None)
    if verbose_gallery is None:
        ncols = num_faces
    else:
        ncols = verbose_gallery[1]

    images_to_show = np.concatenate([arr[np.newaxis] for arr in images_generated])
    images_gallery = gallery(images_to_show, ncols)
    if verbose == 1:
        plt.imshow(images_gallery)
        plt.show()

    if return_gallery:
        return images_gallery
    else:
        return images_generated

def generate_translation_gif(GAN, num_translations = 1, num_frames = 50, fps = 5, z_start = None, z_end = None, path = 'CelebA_Results/Translation_Gif.gif'):
    temp = 'CelebA_Results/Temp_Folder'
    if os.path.exists(temp):
        shutil.rmtree(temp)
    os.makedirs(temp)
    z_s = []
    if z_start is None:
        z_s.append(np.random.normal(0, GAN.z_sd, size=GAN.z_len))
    else:
        z_s.append(z_start)

    if num_translations > 1:
        for iter in range(num_translations-1):
            z_s.append(np.random.normal(0, GAN.z_sd, size=GAN.z_len))

    if z_end is None:
        z_s.append(np.random.normal(0, GAN.z_sd, size=GAN.z_len))
    else:
        z_s.append(z_end)

    images = []
    each_frame = num_frames//num_translations
    for i in range(num_translations):
        for iter in range(each_frame):
            t = iter/each_frame
            this_z = [np.asarray((1-t)*z_s[i] + (t)*z_s[i+1])]
            the_image = GAN.generate_using_z(this_z)
            plt.imsave(temp + '/img' + str(iter) + '.png', the_image[0])
            images.append(imageio.imread(temp + '/img' + str(iter) + '.png'))

    imageio.mimsave(path, images, fps = fps)
    shutil.rmtree(temp)

def show_visualization_form(GAN, num_sliders = None, slider_width = 300):
    form.CelebA_Visualization_Form(GAN, num_sliders, slider_width)



