import darknet
from settings import *
import tensorflow as tf
import numpy as np
import struct
import os

def transpose_weights(weights, num_anchors):
	ksize1, ksize2, channels_in, _ = weights.shape
	weights = weights.reshape([ksize1, ksize2, channels_in, num_anchors, -1])
	coords = weights[:, :, :, :, 0:4]
	iou = np.expand_dims(weights[:, :, :, :, 4], -1)
	classes = weights[:, :, :, :, 5:]
	return np.concatenate([iou, coords, classes], -1).reshape([ksize1, ksize2, channels_in, -1])

def transpose_biases(biases, num_anchors):
	biases = biases.reshape([num_anchors, -1])
	coords = biases[:, 0:4]
	iou = np.expand_dims(biases[:, 4], -1)
	classes = biases[:, 5:]
	return np.concatenate([iou, coords, classes], -1).reshape([-1])

def transpose(sess, layer, num_anchors):
	v = next(iter(filter(lambda v: v.op.name.endswith('kernel'), layer)))
	sess.run(v.assign(transpose_weights(sess.run(v), num_anchors)))
	v = next(iter(filter(lambda v: v.op.name.endswith('bias'), layer)))
	sess.run(v.assign(transpose_biases(sess.run(v), num_anchors)))

with tf.Session() as sess:
	image = tf.placeholder(tf.float32, [1, conf.height, conf.width, 3], name = 'image')
	darknet.darknet(image, conf.classes, conf.anchors.shape[0])
	tf.contrib.framework.get_or_create_global_step()
	tf.global_variables_initializer().run()
	variables = tf.global_variables()
	for i in reversed(xrange(len(variables))):
		if variables[i].op.name.find('darknet_conv') < 0 and variables[i].op.name.find('batch_normalization') < 0:
			del variables[i]
	layers_variables = []
	for i in xrange(0, (len(variables) // 5) * 5, 5):
		layers = [i / 5, variables[i:i + 5]]
		layers_variables.append(layers)
	layers_variables.append([22, variables[-2:]])
	with tf.name_scope('assign'):
		with open(weight_name, 'rb') as fopen:
			major, minor, revision, seen = struct.unpack('4i', fopen.read(16))
			print "major = %d, minor = %d, revision = %d, seen = %d" % (major, minor, revision, seen)
			for i, layer in layers_variables:
				print "processing layer %d" % (i)
				total = 0
				for suffix in ['bias', 'beta', 'gamma', 'moving_mean', 'moving_variance', 'kernel']:
					try:
						v = next(iter(filter(lambda v: v.op.name.endswith(suffix), layer)))
					except:
						continue
					shape = v.get_shape().as_list()
					cnt = np.multiply.reduce(shape)
					total += cnt
					print "%s: %s = %d" % (v.op.name, str(shape), cnt)
					print (os.fstat(fopen.fileno()).st_size - fopen.tell()) / 4
					p = struct.unpack('%df' % cnt, fopen.read(4 * cnt))
					if suffix == 'kernel':
						ksize1, ksize2, channels_in, channels_out = shape
						p = np.reshape(p, [channels_out, channels_in, ksize1, ksize2])
						p = np.transpose(p, [2, 3, 1, 0])
					sess.run(v.assign(p))
				print "%d parameters assigned" % (total)
			remaining = os.fstat(fopen.fileno()).st_size - fopen.tell()
		transpose(sess, layer, conf.anchors.shape[0] / 2)
		saver = tf.train.Saver()
		saver.save(sess, os.getcwd() + '/model.ckpt')
	if remaining > 0:
		print '%d bytes remaining' % (remaining)
		
			