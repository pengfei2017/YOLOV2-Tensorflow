import cv2
import numpy as np
import tensorflow as tf
from settings import *
from darknet import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

sess = tf.InteractiveSession()
image = tf.placeholder(tf.float32, [1, conf.height, conf.width, 3], name = 'image')
darknet = darknet(image, conf.classes, conf.anchors.shape[0])
model = Model(darknet, conf.classes, np.reshape(conf.anchors, (-1, 2)))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

try:
	saver.restore(sess, os.getcwd() + '/model.ckpt')
	print 'load from past checkpoint'
except:
	print 'not found any checkpoint, exiting..'
	exit(0)
	
tensors = [model.conf, model.xy_min, model.xy_max]
tensors = [tf.check_numerics(t, t.op.name) for t in tensors]

list_inputs = os.listdir('input')
for i in list_inputs:
	print 'processing: ' + i
	image_bgr = cv2.imread('input/' + i)
	image_height, image_width, _ = image_bgr.shape
	scale = [image_width / model.cell_width, image_height / model.cell_height]
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	image_std = np.expand_dims(per_image_standardization(cv2.resize(image_rgb, (conf.width, conf.height))).astype(np.float32), 0)
	config, xy_min, xy_max = sess.run(tensors, feed_dict = {image: image_std})
	print config.shape
	boxes = non_max_suppress(config[0], xy_min[0], xy_max[0], threshold, threshold_iou)
	for _conf, _xy_min, _xy_max in boxes:
		index = np.argmax(_conf)
		if _conf[index] > threshold:
			_xy_min = (_xy_min * scale).astype(np.int)
			_xy_max = (_xy_max * scale).astype(np.int)
			print _xy_min
			print _xy_max
			cv2.rectangle(image_rgb, tuple(_xy_min), tuple(_xy_max), (255, 0, 255), 3)
			cv2.putText(image_rgb, conf.class_name[index] + ' (%.1f%%)' % (_conf[index] * 100), 
						tuple(_xy_min), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	cv2.imwrite('output/' + i, image_rgb)