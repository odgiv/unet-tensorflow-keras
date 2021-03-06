'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:32
 * @modify date 2017-05-25 02:20:32
 * @desc [description]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import numpy as np
import scipy.misc as misc
from model import UNet
from utils import dice_coef, dice_coef_loss
from loader import dataLoader, deprocess, dataLoaderNp
from PIL import Image
from utils import VIS, mean_IU

# configure args
from opts import *
from opts import dataset_mean, dataset_std # set them in opts

vis = VIS(save_path=opt.load_from_checkpoint, is_train=False)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define data loader
img_shape = [233, 369]  #img_shape = [opt.imSize, opt.imSize]
test_generator, test_samples = dataLoaderNp(opt.data_path, 1, train_mode=False)
# define model, the last dimension is the channel
label = tf.placeholder(tf.int32, shape=[None]+img_shape)
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class)
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('cross_entropy'): 
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))

saver = tf.train.Saver() # must be added in the end

''' Main '''
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    # restore from a checkpoint if exists
    try:
        saver.restore(sess, tf.train.latest_checkpoint(opt.load_from_checkpoint))
        print ('--> load from checkpoint '+opt.load_from_checkpoint)
    except Exception as ex:
        print ('Unable to load checkpoint ...', ex)
        sys.exit(0)
    dice_score = 0
    for it in range(0, test_samples):
        x_batch, y_batch = next(test_generator)
        # tensorflow wants a different tensor order
        feed_dict = {   
                        img: x_batch,
                        label: y_batch
                    }
        loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
        pred_map = np.argmax(pred_logits[0], axis=2)
        score = vis.add_sample(pred_map, y_batch[0])

        im, gt = deprocess(x_batch[0], dataset_mean, dataset_std, y_batch[0])

        # im = Image.fromarray(im)
        # im.save(os.path.join(opt.load_from_checkpoint, '{0:}.png'.format(it)))
        
        # gt = Image.fromarray(gt, 'L')
        # gt.save(os.path.join(opt.load_from_checkpoint, '{0:}_{1:.3f}.png'.format(it, score)))
        vis.save_seg(pred_map, name='{0:}_{1:.3f}.png'.format(it, score), im=im, gt=gt)

        print ('[iter %f]: loss=%f, meanIU=%f' % (it, loss, score))

    vis.compute_scores()