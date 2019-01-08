# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from dfn_model import DFN
from glob import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input_dir', type=str, default='data', help='Training input path')
parser.add_argument('--output_dir', type=str, default='output', help='Output path')
parser.add_argument("--batch_size", type=int, default=3, help="number of images in batch")
parser.add_argument('--mode', type=str, default="train", help='train, test, export')
parser.add_argument("--crop_size", type=int, default=512, help="crop size of input and output")
parser.add_argument("--classes", type=int, default=2, help="number of output channels / classes excluding none (black)")
parser.add_argument('--models', type=str, default='models', help='path for saving models')
parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use")
parser.add_argument("--layer_depth", type=int, default=21, help="depth of dfn model 21 for resnet 51 and 34 for res 152")
parser.add_argument('--alpha', type=float, default=0.25, help='coefficient for focal loss')
parser.add_argument('--gamma', type=float, default=2.0, help='factor for focal loss')
parser.add_argument('--fl_weight', type=float, default=0.1, help='regularization coefficient for focal loss')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--init_lr', type=float, default=0.004, help='initial learning rate')
parser.add_argument('--power', type=float, default=0.9, help='decay factor of learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor')
parser.add_argument('--stddev', type=float, default=0.02, help='stddev for W initializer')
parser.add_argument('--regularization_scale', type=float, default=0.0001, help='regularization coefficient for W and b')
parser.add_argument('--augment', action="store_true", help="augment images")
parser.add_argument("--num_pyramids", type=int, default=0, help="number of image pyramids")


def model_fn(features, labels, mode, params, config):
	is_predict = mode == tf.estimator.ModeKeys.PREDICT
	is_train = mode == tf.estimator.ModeKeys.TRAIN

	# input_image = features["image"]
	# input_normals = features["normals"]
	# input_elevation = features["elevation"]
	# inputs = tf.concat((input_image, input_normals, input_elevation), axis=-1)
	input_list = [features["image"]]

	for _ in range(params.num_pyramids):
            size = input_list[-1].shape[1:3]
            input_list.append(tf.image.resize_images(input_list[-1], (size[0] // 2, size[1] // 2), tf.image.ResizeMethod.BILINEAR))
            
	inputs = tf.concat(input_list, axis=1)

	model = DFN(X=inputs, Y=labels, n_classes=params.classes+1, depth=params.layer_depth,
				max_iter=params.max_iters, init_lr=params.init_lr, power=params.power,
				momentum=params.momentum, stddev=params.stddev, regularization_scale=params.regularization_scale,
				alpha=params.alpha, gamma=params.gamma, fl_weight=params.fl_weight)

	predictions = {
		"output": model.fuse,
	}

	eval_metric_ops = {}
	
	export_outputs = {
		"output": tf.estimator.export.PredictOutput(model.fuse),
	}

	logging_hook = [tf.train.LoggingTensorHook({
		"loss" : model.total_loss,
		"mean_iou": model.mean_iou
	}, every_n_iter=100)] if is_train else None

	evaluation_hooks = []
	if is_train:
		eval_summary_hook = tf.train.SummarySaverHook(
			save_steps=1,
			output_dir=os.path.join(params.models, "eval_core"),
			summary_op=model.trainval_summary)
		evaluation_hooks.append(eval_summary_hook)

	return tf.estimator.EstimatorSpec(
		mode=mode,
		predictions=predictions,
		loss=model.total_loss if not is_predict else None,
		train_op=model.train_op if is_train else None,
		eval_metric_ops=eval_metric_ops,
		export_outputs=export_outputs,
		training_chief_hooks=None,
		training_hooks=logging_hook,
		scaffold=None,
		evaluation_hooks=evaluation_hooks
	)

def get_input_fn(input_dir, output_dir, shuffle, batch_size, epochs, crop_size, classes, augment):
	def parse_image(input_file_name, output_file_name):
		def _parse(file_name, crop_fraction=None, channels=3):
			image_data = tf.read_file(file_name)

			# Despite its name decode_png can actually decode all image types.
			image = tf.image.convert_image_dtype(tf.image.decode_png(image_data, channels=channels), tf.float32)

			if crop_fraction is not None:
				# Resize first since decode does not seem to give a shape
				image = tf.image.resize_images(image, [2*crop_size, 2*crop_size])
				a = int(crop_fraction[0]*2*crop_size)
				b = -max(1, int(crop_fraction[1]*2*crop_size))
				c = int(crop_fraction[2]*2*crop_size)
				d = -max(1, int(crop_fraction[3]*2*crop_size))
				image = image[a:b, c:d]

			image = tf.image.resize_images(image, [crop_size, crop_size])
			return image

		# Randomly crop out up to 10% of the sides if augmenting
		crop_fraction = np.random.rand(4) * 0.1 if augment else None

		image = _parse(input_file_name, crop_fraction)
		output = _parse(output_file_name, crop_fraction)

		# Random augmentation if wanted
		if augment:
			ops = [
				lambda im: tf.image.random_brightness(im, 0.2),
				lambda im: tf.image.random_contrast(im, 0.8, 1.2),
				lambda im: tf.image.random_hue(im, 0.05),
				lambda im: tf.image.random_saturation(im, 0.8, 1.2),
				lambda im: im + tf.random.normal(im.shape, stddev=0.02)
			]

			np.random.shuffle(ops)

			for augment_op in ops:
				image = tf.clip_by_value(augment_op(image), 0, 1)

		# Make none (black) a class
		is_none = 1 - tf.reduce_sum(output, axis=-1, keepdims=True)
		output = tf.concat((output[:, :, :classes], is_none), axis=-1)

		return {"image": image}, output

	def input_fn():
		file_seed = np.random.randint(10000000)
		input_files = tf.data.Dataset.list_files(input_dir, seed=file_seed)
		output_files = tf.data.Dataset.list_files(output_dir, seed=file_seed)

		dataset = tf.data.Dataset.zip((input_files, output_files))
		if shuffle:
			dataset = dataset.shuffle(buffer_size=100000)
		dataset = dataset.map(parse_image, num_parallel_calls=4)
		dataset = dataset.repeat(epochs)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(2)
		return dataset
	return input_fn

def get_serving_input_receiver_fn(crop_size):
	def serving_input_receiver_fn():
		inputs = {
			"image": tf.placeholder(tf.float32, [None, crop_size, crop_size, 3]),
		}
		return tf.estimator.export.ServingInputReceiver(inputs, inputs)
	return serving_input_receiver_fn

def main():
	args = parser.parse_args()

	# Hackish: calculate the max iters by looking at the input dir
	args.max_iters = args.epochs * len(glob(os.path.join(args.input_dir, "train", "main", "*")))
	
	tf.logging.set_verbosity(tf.logging.INFO)

	run_config = tf.estimator.RunConfig(
		model_dir=args.models,
		train_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus) if args.num_gpus > 1 else None,
		eval_distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=args.num_gpus) if args.num_gpus > 1 else None,
	)

	estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params=args)
	
	if args.mode == "train":
		train_input_fn = get_input_fn(os.path.join(args.input_dir, "train", "main", "*"),
										os.path.join(args.input_dir, "train", "segmentation", "*"),
										shuffle=True, batch_size=args.batch_size, epochs=args.epochs,
										crop_size=args.crop_size, classes=args.classes, augment=args.augment)
		train_spec = tf.estimator.TrainSpec(train_input_fn)

		eval_input_fn = get_input_fn(os.path.join(args.input_dir, "test", "main", "*"),
										os.path.join(args.input_dir, "test", "segmentation", "*"),
										shuffle=False, batch_size=args.batch_size, epochs=args.epochs,
										crop_size=args.crop_size, classes=args.classes, augment=False)
		eval_spec = tf.estimator.EvalSpec(eval_input_fn)

		tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
	elif args.mode == "test":
		eval_input_fn = get_input_fn(os.path.join(args.input_dir, "test", "main", "*"),
										os.path.join(args.input_dir, "test", "segmentation", "*"),
										shuffle=False, batch_size=args.batch_size, epochs=args.epochs,
										crop_size=args.crop_size, classes=args.classes, augment=False)
		estimator.evaluate(eval_input_fn)
	elif args.mode == "export":
		estimator.export_saved_model(args.output_dir, get_serving_input_receiver_fn(args.crop_size))
	else:
		print("Unknown mode", args.mode)

if __name__ == "__main__":
	main()
