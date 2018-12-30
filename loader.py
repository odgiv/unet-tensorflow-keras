'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:43
 * @modify date 2017-05-19 03:06:43
 * @desc [description]
'''

from data_generator.image import ImageDataGenerator
import scipy.misc as misc
import numpy as np
import os
import glob
import itertools
from PIL import ImageFile
from PIL import Image as pil_image
import h5py
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Modify this for data normalization
def preprocess(img, mean, std, label, normalize_label=True):
    out_img = img / img.max()  # scale to [0,1]
    out_img = (out_img - np.array(mean).reshape(1, 1, 3)) / \
        np.array(std).reshape(1, 1, 3)

    if len(label.shape) == 4:
        label = label[:, :, :, 0]
    if normalize_label:
        if np.unique(label).size > 2:
            print(
                'WRANING: the label has more than 2 classes. Set normalize_label to False')
        # if the loaded label is binary has only [0,255], then we normalize it
        label = label / label.max()
    return out_img, label.astype(np.int32)


def deprocess(img, mean, std, label):
    out_img = img / img.max()  # scale to [0,1]
    out_img = (out_img * np.array(std).reshape(1, 1, 3)) + \
        np.array(std).reshape(1, 1, 3)
    out_img = out_img * 255.0

    return out_img.astype(np.uint8), label.astype(np.uint8)

# image normalization default: scale to [-1,1]


def imerge(a, b, mean, std, normalize_label):
    for img, label in itertools.zip_longest(a, b):
        # j is the mask: 1) gray-scale and int8
        #img, label = preprocess(img, mean, std, label, normalize_label=normalize_label)
        yield img, label


'''
    Use the Keras data generators to load train and test
    Image and label are in structure:
        train/
            img/
                0/
            gt/
                0/

        test/
            img/
                0/
            gt/
                0/

'''


def dataLoader(path, batch_size, imSize, train_mode=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):

    # augmentation parms for the train generator
    if train_mode:
        train_data_gen_args = dict(
            horizontal_flip=True,
            vertical_flip=True,
        )
    else:
        train_data_gen_args = dict()

    # seed has to been set to synchronize img and mask generators
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
        path+'img',
        class_mode=None,
        target_size=imSize,
        batch_size=batch_size,
        seed=seed,
        shuffle=train_mode)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
        path+'gt',
        class_mode=None,
        target_size=imSize,
        batch_size=batch_size,
        color_mode='grayscale',
        seed=seed,
        shuffle=train_mode)

    samples = train_image_datagen.samples
    generator = imerge(train_image_datagen,
                       train_mask_datagen, mean, std, True)

    return generator, samples


def mergeDatasets(path):
    list_us_array = []
    list_gt_array = []

    min_img_size = (266, 369)
    num_rows_cut_bottom = 33

   # max_num_zero_bottom_rows = 0

    for f in sorted(os.listdir(path)):
        # If f is directory, not a file
        files_directory = os.path.join(path, f)
        if not os.path.isdir(files_directory):
            continue
        print("entering directory: ", files_directory)
        h5f = h5py.File(os.path.join(files_directory, 'us_gt_vol.h5'), 'r')
        us_vol = h5f['us_vol'][:]
        gt_vol = h5f['gt_vol'][:]
        gt_vol = np.transpose(gt_vol, (1, 0, 2))

        cut_at_ax0 = 0
        cut_at_ax1 = 0
        # To check maximum num of consecutive all 0.0 rows from bottom.
        # for i in range(us_vol.shape[-1]):
        #     sli = us_vol[:, :, i]
        #     num_zero_bottom_rows = 0
        #     for j in range(sli.shape[0]-1, 0, -1):
        #         row = sli[j, :]
        #         if np.all(row == 0.0):
        #             num_zero_bottom_rows += 1
        #         else:
        #             break

        #     if max_num_zero_bottom_rows < num_zero_bottom_rows:
        #         max_num_zero_bottom_rows = num_zero_bottom_rows
        # print(max_num_zero_bottom_rows)

        if us_vol.shape[0] > min_img_size[0]:
            cut_at_ax0 = random.randrange(
                0, (us_vol.shape[0] - min_img_size[0]), 1)

        if us_vol.shape[1] > min_img_size[1]:
            cut_at_ax1 = random.randrange(
                0, (us_vol.shape[1] - min_img_size[1]), 1)

        us_vol = us_vol[cut_at_ax0:cut_at_ax0 +
                        min_img_size[0] - num_rows_cut_bottom, cut_at_ax1:cut_at_ax1 + min_img_size[1], :]
        gt_vol = gt_vol[cut_at_ax0:cut_at_ax0 +
                        min_img_size[0] - num_rows_cut_bottom, cut_at_ax1:cut_at_ax1 + min_img_size[1], :]

        list_us_array.append(us_vol)
        list_gt_array.append(gt_vol)

    X = np.dstack(list_us_array)
    Y = np.dstack(list_gt_array)

    return X, Y


def dataLoaderNp(path, batch_size, train_mode=True, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    X, Y = mergeDatasets(path)
    X = np.transpose(X, (2, 0, 1))
    Y = np.transpose(Y, (2, 0, 1))

    X = np.expand_dims(X, -1)
    Y = np.expand_dims(Y, -1)

    train_data_gen_args = dict(
        rotation_range=20,
        zoom_range=[0.7, 1.]
    )
    # seed has to been set to synchronize img and mask generators
    seed = 1
    train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow(
        x=X,
        batch_size=batch_size,
        seed=seed,
        shuffle=train_mode)
    train_mask_datagen = ImageDataGenerator(**train_data_gen_args).flow(
        x=Y,
        batch_size=batch_size,
        seed=seed,
        shuffle=train_mode)

    generator = imerge(train_image_datagen,
                       train_mask_datagen, mean, std, False)

    return generator, X.shape[0]


if __name__ == "__main__":
    path = "C:\\Users\\odgiiv\\tmp\\code\\u-net\\data\\juliana_wo_symImages\\test"

    gen, samples = dataLoaderNp(path, 1, False)

    for _ in range(5):
        x, y = next(gen)
        x = np.uint8(x[0,:,:,0])    
        y = np.uint8(y[0,:,:,0])
        x = pil_image.fromarray(x)
        y = pil_image.fromarray(y*255, 'L')

        x.show()
        y.show()