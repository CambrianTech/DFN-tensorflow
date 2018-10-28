# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from PIL import Image

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Take in image directory and return a directory containing image directory
# and images split into train, val, test
def create_image_lists(image_dir, batch_size=32):
	
	result = {}
	
	training_images = []
	validation_images = []
	testing_images = []

	included_extensions = ['jpg', 'png']
	
	for category in ["train", "val", "test"]:
		
		category_path = os.path.join(image_dir, category)
		main_path = os.path.join(category_path, "main")
		segmentation_path = os.path.join(category_path, "segmentation")

		main_filenames = [fn for fn in os.listdir(main_path) if any(fn.endswith(ext) for ext in included_extensions)]
		segmentation_filenames = [fn for fn in os.listdir(segmentation_path) if any(fn.endswith(ext) for ext in included_extensions)]
		
		if category != "test":
			assert len(main_filenames) == len(segmentation_filenames), "The number of images in the " + main_path + " is not equal to that in the " + segmentation_path
		
		for main_filename in main_filenames:

			if category == "train":
				
				training_images.append(main_filename)
			
			if category == "val":
				
				validation_images.append(main_filename)
			
			if category == "test":
				
				testing_images.append(main_filename)
			
			else:
				
				pass
	
	result = {
		"root": image_dir,
		"train": training_images,
		"val": validation_images,
		"test": testing_images
	}
	
	return result


# for f in /home/joel/YUGE/datasets/unreal_rugs_dfn/train/segmentation/Screenshot_*; do mv -v "$f" $(echo "$f" | sed -e "s/S\.png/C\.png/g"); done
# for f in /home/joel/YUGE/datasets/unreal_rugs_dfn/train/segmentation/ADE_train_*; do mv -v "$f" $(echo "$f" | sed -e "s/_seg\.png/\.png/g"); done

def get_batch_of_trainval(result, category="train", batch_size=32):
	
	assert category != "test", "category is not allowed to be 'test' here"
	
	image_dir = result["root"]
	filenames = result[category]
	batch_list = random.sample(filenames, batch_size)
	
	main_list = []
	segmentation_list = []
	
	for filename in batch_list:
		
		category_path = os.path.join(image_dir, category)
		main_path = os.path.join(category_path, "main/" + filename)

		segmentation_path = os.path.join(category_path, "segmentation/" + filename)

		# Get other extension. Do not require labels to have same extension as src
		if not os.path.isfile(segmentation_path):
			path_wo_extension = os.path.splitext(segmentation_path)[0]
			extension = os.path.splitext(segmentation_path)[1][1:]
			other_extension = ".jpg" if extension == ".png" else ".png"
			segmentation_path = path_wo_extension + other_extension
		
		img = Image.open(main_path)
		img = img.resize((512, 512), Image.NEAREST)
		img = np.array(img, np.float32)
		img = img[:,:,:3]
		
		assert img.ndim == 3 and img.shape[2] == 3
		
		img = np.expand_dims(img, axis=0)
		label = Image.open(segmentation_path).convert("L").resize((512, 512), Image.NEAREST)
		label = np.array(label, np.bool)
		labels = np.zeros((512, 512, 2), np.float32)
		labels[:, :, 0] = ~label
		labels[:, :, 1] = label
		labels = np.expand_dims(labels, axis=0)
		
		main_list.append(img)
		segmentation_list.append(labels)
	
	X = np.concatenate(main_list, axis=0)
	X -= np.mean(X)
	X /= (np.max(np.fabs(X)) + 1e-12)
	Y = np.concatenate(segmentation_list, axis=0)
	
	return X, Y

def get_batch_of_test(result, start_id, batch_size=32):
	
	image_dir = result["root"]
	filenames = result["test"]
	next_start_id = start_id + batch_size
	
	if next_start_id > len(filenames):
		
		next_start_id = len(filenames)
	
	paddings = start_id + batch_size - next_start_id
	
	main_list = []
	# segmentation_list = []
	size_list = []

	category="test"
	
	for idx in range(start_id, next_start_id):
		
		category_path = os.path.join(image_dir, category)
		main_path = os.path.join(category_path, "main/" + filenames[idx])
		# segmentation_path = os.path.join(category_path, "segmentation/" + filenames[idx])
		
		img = Image.open(main_path)
		# label = Image.open(segmentation_path).convert("L")
		img_size = img.size
		# label_size = label.size
		
		# assert img_size[0:2] == label_size

		img = img.resize((512, 512), Image.NEAREST)
		img = np.array(img, np.float32)
		img = img[:,:,:3]
		
		assert img.ndim == 3 and img.shape[2] == 3
		
		img = np.expand_dims(img, axis=0)
		# label = label.resize((512, 512), Image.NEAREST)
		# label = np.array(label, np.bool)
		# labels = np.zeros((512, 512, 2), np.float32)
		# labels[:, :, 0] = ~label
		# labels[:, :, 1] = label
		# labels = np.expand_dims(labels, axis=0)
		
		main_list.append(img)
		# segmentation_list.append(labels)
		size_list.append(img_size[0:2])
	
	for i in range(paddings):
		
		main_list.append(main_list[-1])
		# segmentation_list.append(segmentation_list[-1])
		size_list.append(size_list[-1])
	
	X = np.concatenate(main_list, axis=0)
	X -= np.mean(X)
	X /= (np.max(np.fabs(X)) + 1e-12)
	# Y = np.concatenate(segmentation_list, axis=0)
	
	return X, size_list, next_start_id, filenames[start_id:next_start_id]
