from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import os
from opts import dataset_mean, dataset_std  # set them in opts
from opts import *
from utils import VIS, mean_IU
from loader import dataLoader, dataLoaderNp
from utils import dice_coef
from model import UNet
import tensorflow as tf
from tensorflow.python.ops import array_ops
import numpy as np

try:
    from tensorflow.contrib import keras as keras
    print ('load keras from tensorflow package')
except:
    print ('update your tensorflow')


'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:32
 * @modify date 2017-05-19 03:06:32
 * @desc [description]
'''

def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits,gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    labels=tf.one_hot(labels,depth=y_pred.shape[-1])
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_mean(L,axis=-1)
    return L

def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = tf.clip(pt_1, 1e-3, .999)
    pt_0 = tf.clip(pt_0, 1e-3, .999)

    return -tf.sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1))-tf.sum((1-alpha) * tf.pow( pt_0, gamma) * tf.log(1. - pt_0))

def focal_loss(target, output, gamma=2.):
    # tf.cast(target, tf.float32)
    # tf.cast(output, tf.float32)
    output /= keras.backend.sum(output, axis=-1, keepdims=True)
    eps = keras.backend.epsilon()
    output = keras.backend.clip(output, eps, 1. - eps)
    return -keras.backend.sum(keras.backend.pow(1. - output, gamma) * target * keras.backend.log(output), axis=-1)

SEED = 0  # set set to allow reproducing runs
np.random.seed(SEED)
tf.set_random_seed(SEED)

# configure args

# save and compute metrics
vis = VIS(save_path=opt.checkpoint_path, is_train=True)

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


''' Users define data loader (with train and test) '''
img_shape = [233, 369]  # [opt.imSize, opt.imSize]
# train_generator, train_samples = dataLoader(
#     opt.data_path+'/train/', opt.batch_size, img_shape, mean=dataset_mean, std=dataset_std)
# test_generator, test_samples = dataLoader(
#     opt.data_path+'/val/', 1,  img_shape, train_mode=False, mean=dataset_mean, std=dataset_std)

train_generator, test_generator, train_samples, test_samples = dataLoaderNp(
    opt.data_path + 'train/', opt.batch_size, mean=dataset_mean, std=dataset_std)


opt.iter_epoch = int(train_samples)
# define input holders
label = tf.placeholder(tf.int32, shape=[None]+img_shape)
# define model
with tf.name_scope('unet'):
    model = UNet().create_model(
        img_shape=img_shape+[3], num_class=opt.num_class) #3
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('cross_entropy'):
    cross_entropy_loss = focal_loss(target=label, output=pred) #tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))
# define optimizer
global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(opt.learning_rate, global_step,
                                               opt.iter_epoch, opt.lr_decay, staircase=True)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    cross_entropy_loss, global_step=global_step)

# compute dice score for simple evaluation during training
# with tf.name_scope('dice_eval'):
#     dice_evaluator = tf.reduce_mean(dice_coef(label, pred))

''' Tensorboard visualization '''
# cleanup pervious info
if opt.load_from_checkpoint == '':
    cf = os.listdir(opt.checkpoint_path)
    for item in cf:
        if 'event' in item:
            os.remove(os.path.join(opt.checkpoint_path, item))
# define summary for tensorboard
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()
# define saver
train_writer = tf.summary.FileWriter(opt.checkpoint_path, sess.graph)
saver = tf.train.Saver()  # must be added in the end

''' Main '''
tot_iter = opt.iter_epoch * opt.epoch
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change
    if opt.load_from_checkpoint != '':
        try:
            saver.restore(sess, opt.load_from_checkpoint)
            print('--> load from checkpoint '+opt.load_from_checkpoint)
        except:
            print('unable to load checkpoint ...' + str(e))
    # debug
    start = global_step.eval()
    for it in range(start, tot_iter):
        if it % 1000 == 0 or it == start: #opt.iter_epoch

            saver.save(sess, opt.checkpoint_path +
                       'model', global_step=global_step)
            print('save a checkpoint at ' +
                  opt.checkpoint_path+'model-'+str(it))
            print('start testing {} samples...'.format(test_samples))
            for ti in range(test_samples):
                x_batch, y_batch = next(test_generator)
                # tensorflow wants a different tensor order
                feed_dict = {
                    img: x_batch,
                    label: y_batch,
                }
                loss, pred_logits = sess.run(
                    [cross_entropy_loss, pred], feed_dict=feed_dict)
                pred_map_batch = np.argmax(pred_logits, axis=3)
                # import pdb; pdb.set_trace()
                for pred_map, y in zip(pred_map_batch, y_batch):
                    score = vis.add_sample(pred_map, y)
            vis.compute_scores(suffix=it)

        x_batch, y_batch = next(train_generator)
        feed_dict = {img: x_batch,
                     label: y_batch
                     }
        _, loss, summary, lr, pred_logits = sess.run([train_step,
                                                      cross_entropy_loss,
                                                      summary_merged,
                                                      learning_rate,
                                                      pred
                                                      ], feed_dict=feed_dict)
        global_step.assign(it).eval()
        train_writer.add_summary(summary, it)

        pred_map = np.argmax(pred_logits[0], axis=2)
        score, _ = mean_IU(pred_map, y_batch[0])

        if it % 20 == 0:
            print('[iter %d, epoch %.3f]: lr=%f loss=%f, mean_IU=%f' %
                  (it, float(it)/opt.iter_epoch, lr, loss, score))
