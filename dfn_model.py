# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from subnetworks import nn_base, nn_border, nn_smooth
from losses import focal_loss, pw_softmaxwithloss_2d

def cal_iou(y_true, y_pred):
	overlap_map = y_true * y_pred

	gt_count = tf.reduce_sum(tf.cast(y_true, tf.float32), [1, 2], keepdims=True)
	pred_count = tf.reduce_sum(tf.cast(y_pred, tf.float32), [1, 2], keepdims=True)
	overlap_count = tf.reduce_sum(tf.cast(overlap_map, tf.float32), [1, 2], keepdims=True)
	
	iou = tf.div(overlap_count, gt_count + pred_count - overlap_count)
	
	return iou

class DFN(object):
	def __init__(self, X, Y, max_iter, n_classes, depth, init_lr=0.004, power=0.9, momentum=0.9, stddev=0.02, regularization_scale=0.0001, alpha=0.25, gamma=2.0, fl_weight=0.1):
		self.n_classes = n_classes
		self.depth = depth
		self.max_iter = max_iter
		self.init_lr = init_lr
		self.power = power
		self.momentum = momentum
		self.stddev = stddev
		self.regularization_scale = regularization_scale
		self.alpha = alpha
		self.gamma = gamma
		self.fl_weight = fl_weight
			
		self.X = X
		
		self.build_arch()

		# Only build loss / summary when we have labels
		self.Y = Y
		if self.Y is not None:
			self.loss()
			self.evaluation()
			self.global_iter = tf.train.get_global_step()
			self.lr = tf.train.polynomial_decay(self.init_lr, self.global_iter, self.max_iter, end_learning_rate=0.0, power=self.power)
			self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum)
			self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_iter)
			self._summary()
		
		tf.logging.info('Setting up the main structure')
	
	def build_arch(self):
		######### -*- ResNet-101 -*- #########
		with tf.variable_scope("resnet"):
			self.ib_2, self.ib_3, self.ib_4, self.ib_5, self.ib_6, self.global_avg_pool = nn_base(self.X, self.n_classes, depth=self.depth, k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=None)
		
		######### -*- Smooth Network -*- #########
		with tf.variable_scope("smooth"):
			self.b1, self.b2, self.b3, self.b4, self.b5, self.fuse = nn_smooth(self.ib_2, self.ib_3, self.ib_4, self.ib_5, self.ib_6, self.global_avg_pool, self.n_classes,k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=None)
		
		######### -*- Border Network -*- #########
		with tf.variable_scope("border"):
			self.o = nn_border(self.ib_2, self.ib_3, self.ib_4, self.ib_5, self.ib_6, self.n_classes, k=0, initializer=tf.random_normal_initializer(0, self.stddev), regularizer=None)
	
	def loss(self):
		######### -*- Softmax Loss -*- #########
		self.softmax_b1, self.ce1 = pw_softmaxwithloss_2d(self.Y, self.b1)
		self.softmax_b2, self.ce2 = pw_softmaxwithloss_2d(self.Y, self.b2)
		self.softmax_b3, self.ce3 = pw_softmaxwithloss_2d(self.Y, self.b3)
		self.softmax_b4, self.ce4 = pw_softmaxwithloss_2d(self.Y, self.b4)
		self.softmax_b5, self.ce5 = pw_softmaxwithloss_2d(self.Y, self.b5)
		self.softmax_fuse, self.cefuse = pw_softmaxwithloss_2d(self.Y, self.fuse)
		self.total_ce = self.ce1 + self.ce2 + self.ce3 + self.ce4 + self.ce5 + self.cefuse
		
		######### -*- Focal Loss -*- #########
		self.fl = focal_loss(self.Y, self.o, alpha=self.alpha, gamma=self.gamma)
		
		######### -*- Total Loss -*- #########
		self.total_loss = self.total_ce + self.fl_weight * self.fl
	
	def evaluation(self):
		self.prediction = tf.argmax(self.fuse, axis = 3)
		self.ground_truth = tf.argmax(self.Y, axis = 3)
		self.iou = cal_iou(self.ground_truth, self.prediction)
		self.mean_iou = tf.reduce_mean(self.iou)
	
	def _summary(self):
		trainval_summary = []
		trainval_summary.append(tf.summary.scalar('softmax_loss', self.total_ce))
		trainval_summary.append(tf.summary.scalar('focal_loss', self.fl))
		trainval_summary.append(tf.summary.scalar('total_loss', self.total_loss))
		trainval_summary.append(tf.summary.scalar('mean_iou', self.mean_iou))
		trainval_summary.append(tf.summary.image("input_image", tf.cast(255 * self.X[:, :, :, :3], tf.uint8), max_outputs=3))
		trainval_summary.append(tf.summary.image("output", tf.cast(255 * self.Y[:, :, :, :1], tf.uint8), max_outputs=3))
		trainval_summary.append(tf.summary.image("prediction", tf.cast(255 * self.softmax_fuse[:, :, :, :1], tf.uint8), max_outputs=3))
		self.trainval_summary = tf.summary.merge(trainval_summary)
