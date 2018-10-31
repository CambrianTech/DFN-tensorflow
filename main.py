# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from config import cfg
from utils import *
from dfn_model import DFN
import argparse
from scipy import misc
from skimage import color
import shutil

from tensorflow.python.framework import graph_util,dtypes
from tensorflow.python.tools import optimize_for_inference_lib, selective_registration_header_lib


parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='data', help='Training input path')
parser.add_argument('--output_dir', type=str, default='output', help='Output path')
parser.add_argument('--checkpoint', type=str, default='models', help='Training input path')
parser.add_argument("--batch_size", type=int, default=3, help="number of images in batch")
parser.add_argument("--save_freq", type=int, default=2500, help="save_freq")
parser.add_argument('--mode', type=str, default="train", help='train, test, export')
parser.add_argument("--crop_size", type=int, default=512, help="crop size of input and output")
parser.add_argument("--channels", type=int, default=3, help="number of input channels")

args = parser.parse_args()

def train(result, model, logdir, train_sum_freq, val_sum_freq, save_freq, models, fd):
	
	num_val_batch = len(result["val"]) // model.batch_size
	step = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	with tf.Session(config=config, graph=model.graph) as sess:
		
		sess.run(tf.global_variables_initializer())
		train_writer = tf.summary.FileWriter(logdir + "/train", sess.graph)
		val_writer = tf.summary.FileWriter(logdir + "/val", sess.graph)
		
		start = time.time()
		for global_step in range(model.max_iter):
			
			print("Training for iter %d/%d: " % (global_step, model.max_iter))
			fd.write("Training for iter %d/%d: \n" % (global_step, model.max_iter))
			trX, trY = get_batch_of_trainval(result, "train", model.batch_size)
			_, total_loss, softmax_loss, focal_loss, mean_iou = sess.run([model.train_op, model.total_loss, model.total_ce, model.fl, model.mean_iou], feed_dict={model.X: trX, model.Y: trY})
			
			assert not np.isnan(total_loss), "Something wrong! loss is nan..."
			
			print("total loss: {}, softmax loss: {}, focal loss: {}, mean iou: {}".format(total_loss, softmax_loss, focal_loss, mean_iou))
			fd.write("total loss: {}, softmax loss: {}, focal loss: {}, mean iou: {}\n".format(total_loss, softmax_loss, focal_loss, mean_iou))
			
			if global_step % train_sum_freq == 0:
				
				summary_str = sess.run(model.trainval_summary, feed_dict={model.X: trX, model.Y: trY})
				train_writer.add_summary(summary_str, global_step)
			
			if val_sum_freq != 0 and global_step % val_sum_freq == 0:
				
				print("\nValidation phase: ")
				fd.write("\nValidation phase: \n")
				
				val_loss = 0
				val_ce = 0
				val_fl = 0
				val_iou = 0
				
				for i in range(num_val_batch):
					
					valX, valY = get_batch_of_trainval(result, "val", model.batch_size)
					total_loss, softmax_loss, focal_loss, mean_iou, summary_str = sess.run([model.total_loss, model.total_ce, model.fl, model.mean_iou, model.trainval_summary], feed_dict={model.X: valX, model.Y: valY})
					val_writer.add_summary(summary_str, step)
					step += 1
					val_loss += total_loss
					val_ce += softmax_loss
					val_fl += focal_loss
					val_iou += mean_iou
				
				val_loss /= (model.batch_size * num_val_batch)
				val_ce /= (model.batch_size * num_val_batch)
				val_fl /= (model.batch_size * num_val_batch)
				val_iou /= (model.batch_size * num_val_batch)
				
				print("total loss: {}, softmax loss: {}, focal loss: {}, mean iou: {}\n".format(val_loss, val_ce, val_fl, val_iou))
				fd.write("total loss: {}, softmax loss: {}, focal loss: {}, mean iou: {}\n\n".format(val_loss, val_ce, val_fl, val_iou))
			
			if save_freq != 0 and (global_step + 1) % save_freq == 0:
				
				print("Saving model for iter %d..." % global_step)
				fd.write("Saving model for iter %d...\n" % global_step)
				model.saver.save(sess, models + "/model_iter_%04d" % global_step, global_step=global_step)
		
		print("Total time: %d" % (time.time() - start))
		fd.write("Total time: %d" % (time.time() - start))

def test(result, model, models, test_outputs):
	
	num_te_batch = int(math.ceil(float(len(result["test"]) / model.batch_size)))
	idx = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	with tf.Session(config=config, graph=model.graph) as sess:
		
		model.saver.restore(sess, tf.train.latest_checkpoint(models))
		tf.logging.info("Model restored!")
		print("Test phase: ")
		
		test_iou = 0
		start = time.time()
		for i in range(num_te_batch):
			
			mean_iou = 0
			teX, size_list, idx, filenames = get_batch_of_test(result, idx, model.batch_size)

			prediction = sess.run(model.prediction, feed_dict={model.X: teX})
			
			for j in range(len(filenames)):
				output = Image.fromarray(prediction[j] * 255.0).convert("L").resize(size_list[j], Image.NEAREST)
				img = teX[j]
				mask = np.asarray(output)
				img = misc.imresize(img, (mask.shape[0], mask.shape[1]))

				img = overlayMask(mask, img, 60)

				path = test_outputs + "/" + filenames[j]
				misc.imsave(path, img)

				# output.save(test_outputs + "/" + filenames[j])
				print(path + " has been saved.")
		
		print("Total time: %d" % (time.time() - start))
		print("All results have been saved.")

def export(args, model):
	
	idx = 0
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	
	with tf.Session(config=config, graph=model.graph) as sess:

		print("##############################################################")
		print("\nLoading model from checkpoint")
		model.saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
		print("Model restored")

		input_name = "input"
		output_name = "output"
		shutil.rmtree(args.output_dir)

		print("##############################################################\n")
		print("Input Name:", input_name)
		print("Output Name:", output_name)
		print("##############################################################\n")
		tf.saved_model.simple_save(sess, args.output_dir, {input_name: model.X}, {output_name: model.prediction})

		print("Finished: %d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node))
		print("##############################################################\n") 

def main(_):
	
	# get dataset info
	cfg.images = args.input_dir
	cfg.is_training = args.mode == "train"
	cfg.models = args.checkpoint
	cfg.batch_size = args.batch_size
	cfg.save_freq = args.save_freq

	if args.mode == "export":
		args.batch_size = None
	
	result = create_image_lists(cfg.images)

	if cfg.is_training:
		if len(result["train"]) < cfg.batch_size:
			raise ValueError("%d training images found at path '%s'" % (len(result["train"]), cfg.images))

		if len(result["val"]) < cfg.batch_size:
			raise ValueError("%d validation images found at path '%s'" % (len(result["val"]), cfg.images))

	max_iters = len(result["train"]) * cfg.epoch // cfg.batch_size
	
	tf.logging.info('Loading Graph...')
	model = DFN(max_iters, batch_size=cfg.batch_size, init_lr=cfg.init_lr, power=cfg.power, momentum=cfg.momentum, stddev=cfg.stddev, regularization_scale=cfg.regularization_scale, alpha=cfg.alpha, gamma=cfg.gamma, fl_weight=cfg.fl_weight)
	tf.logging.info('Graph loaded.')
	
	if cfg.is_training:
		
		if not tf.gfile.Exists(cfg.logdir):
			
			tf.gfile.MakeDirs(cfg.logdir)
		
		if not tf.gfile.Exists(cfg.models):
			
			tf.gfile.MakeDirs(cfg.models)
		
		if os.path.exists(cfg.log):
			
			os.remove(cfg.log)
		
		fd = open(cfg.log, "a")
		tf.logging.info('Start training...')
		fd.write('Start training...\n')
		train(result, model, cfg.logdir, cfg.train_sum_freq, cfg.val_sum_freq, cfg.save_freq, cfg.models, fd)
		tf.logging.info('Training done.')
		fd.write('Training done.')
		fd.close()
	
	elif args.mode == "test":
		if not tf.gfile.Exists(cfg.test_outputs):
			
			tf.gfile.MakeDirs(cfg.test_outputs)
		
		tf.logging.info('Start testing...')
		test(result, model, cfg.models, cfg.test_outputs)
		tf.logging.info('Testing done.')
	elif args.mode == "export":
		export(args, model)


if __name__ == "__main__":
	
	tf.app.run()
