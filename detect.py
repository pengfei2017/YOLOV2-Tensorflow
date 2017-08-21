import cv2
import numpy as np
import tensorflow as tf
from settings import *
import darknet
import os

sess = tf.InteractiveSession()
image = tf.placeholder(tf.float32, [1, conf.height, conf.width, 3], name = 'image')
darknet = darknet.darknet(image, conf.classes, conf.anchors.shape[0])
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
cap = cv2.VideoCapture(1)

while True:
	try:
		ret, image_bgr = cap.read()
		image_height, image_width, _ = image_bgr.shape
		scale = [image_width / model.cell_width, image_height / model.cell_height]
		image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
		image_std = np.expand_dims(per_image_standardization(cv2.resize(image_rgb, (conf.width, conf.height))).astype(np.float32), 0)
		conf, xy_min, xy_max = sess.run(tensors, feed_dict = {image: image_std})
		boxes = non_max_suppress(conf[0], xy_min[0], xy_max[0], threshold, threshold_iou)
		for _conf, _xy_min, _xy_max in boxes:
			index = np.argmax(_conf)
			if _conf[index] > threshold:
				_xy_min = (_xy_min * scale).astype(np.int)
				_xy_max = (_xy_max * scale).astype(np.int)
				cv2.rectangle(image_bgr, tuple(_xy_min), tuple(_xy_max), (255, 0, 255), 3)
				cv2.putText(image_bgr, conf.class_name[index] + ' (%.1f%%)' % (_conf[index] * 100), 
						tuple(_xy_min), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
			
		cv2.imshow('detection', image_bgr)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	except Exception as e:
		print e
		continue
	
cap.release()
cv2.destroyAllWindows()