import tensorflow as tf
import numpy as np

def leaky_relu(inputs, alpha = 0.1):
	return tf.maximum(inputs, alpha * inputs)

def reorganized(net, stride = 2, name = 'reorganized'):
	batch_size, height, width, channels = net.get_shape().as_list()
	_height, _width, _channel = height // stride, width // stride, channels * stride * stride
	with tf.name_scope(name) as name:
		net = tf.reshape(net, [batch_size, _height, stride, _width, stride, channels])
		net = tf.transpose(net, [0, 1, 3, 2, 4, 5])
		net = tf.reshape(net, [batch_size, _height, _width, -1], name = name)
	return net

def darknet(net, classes, len_anchors, training = False, center = True):
	def batch_norm(net):
		net = tf.layers.batch_normalization(net, center = center, scale = True, epsilon = 1e-5, training = training)
		if not center:
			net = tf.nn.bias_add(net, tf.Variable(tf.zeros((int(net.shape[3]))), 'biases'))
		return net
	
	def conv2d(net, filters, name, activation = leaky_relu, k = 3):
		net = tf.layers.conv2d(net, filters, k, activation = activation, use_bias = False, padding = 'SAME', name = name)
		return batch_norm(net)
	
	def max_pooling2d(net, name, k = 2, stride = 2):
		return tf.layers.max_pooling2d(net, k, stride, padding = 'SAME', name = name)
	
	# COC: 608 x 608
	# VOC: 416 x 416
	net = conv2d(net, 32, 'darknet_conv0')
	net = max_pooling2d(net, 'darknet_maxpool0')
	# COC: 304 x 304
	# VOC: 208 x 208
	net = conv2d(net, 64, 'darknet_conv1')
	net = max_pooling2d(net, 'darknet_maxpool1')
	# COC: 152 x 152
	# VOC: 104 x 104
	net = conv2d(net, 128, 'darknet_conv2')
	net = conv2d(net, 64, 'darknet_conv3', k = 1)
	net = conv2d(net, 128, 'darknet_conv4')
	net = max_pooling2d(net, 'darknet_maxpool4')
	# COC: 76 x 76
	# VOC: 52 x 52
	net = conv2d(net, 256, 'darknet_conv5')
	net = conv2d(net, 128, 'darknet_conv6', k = 1)
	net = conv2d(net, 256, 'darknet_conv7')
	net = max_pooling2d(net, 'darknet_maxpool7')
	# COC: 38 x 38
	# VOC: 26 x 26
	net = conv2d(net, 512, 'darknet_conv8')
	net = conv2d(net, 256, 'darknet_conv9', k = 1)
	net = conv2d(net, 512, 'darknet_conv10')
	net = conv2d(net, 256, 'darknet_conv11', k = 1)
	net = conv2d(net, 512, 'darknet_conv12')
	# checkpoint for current net, 9th layer
	passthrough = tf.identity(net, name ='passthrough')
	net = max_pooling2d(net, 'darknet_maxpool12')
	# COC: 19 x 19
	# VOC: 13 x 13
	net = conv2d(net, 1024, 'darknet_conv13')
	net = conv2d(net, 512, 'darknet_conv14', k = 1)
	net = conv2d(net, 1024, 'darknet_conv15')
	net = conv2d(net, 512, 'darknet_conv16', k = 1)
	net = conv2d(net, 1024, 'darknet_conv17')
	net = conv2d(net, 1024, 'darknet_conv18')
	net = conv2d(net, 1024, 'darknet_conv19')
	_net = conv2d(passthrough, 64, 'darknet_conv20', k = 1)
	_net = reorganized(_net)																																																																																																																	
	net = tf.concat([_net, net], 3, 'concat')
	net = conv2d(net, 1024, 'darknet_conv21')
	net = tf.layers.conv2d(net, (len_anchors * (5 + classes)) / 2, 1, activation = None, use_bias = True, padding = 'SAME', name = 'darknet_conv22')
	return net
	
def iou(xymin1, xymax1, xymin2, xymax2):
	areas1 = np.multiply.reduce(xymax1 - xymin1)
	areas2 = np.multiply.reduce(xymax2 - xymin2)
	xymin = np.maximum(xymin1, xymin2) 
	xymax = np.minimum(xymax1, xymax2)
	wh = np.maximum(xymax - xymin, 0)
	areas = np.multiply.reduce(wh)
	return areas / np.maximum(areas1 + areas2 - areas, 1e-10)

def non_max_suppress(conf, xy_min, xy_max, threshold, threshold_iou):
	_, _, classes = conf.shape
	boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in zip(conf.reshape(-1, classes), xy_min.reshape(-1, 2), xy_max.reshape(-1, 2))]
	for c in range(classes):
		boxes.sort(key = lambda box: box[0][c], reverse = True)
		for i in range(len(boxes) - 1):
			box = boxes[i]
			if box[0][c] <= threshold:
				continue
			for _box in boxes[i + 1:]:
				if iou(box[1], box[2], _box[1], _box[2]) >= threshold_iou:
					_box[0][c] = 0
	return boxes

def calc_cell_xy(cell_height, cell_width, dtype = np.float32):
	cell_base = np.zeros([cell_height, cell_width, 2], dtype=dtype)
	for y in range(cell_height):
		for x in range(cell_width):
			cell_base[y, x, :] = [x, y]
	return cell_base

def per_image_standardization(image):
    #stddev = np.std(image)
    #return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))
	return image

class Model:
	def __init__(self, net, classes, anchors):
		_, self.cell_height, self.cell_width, _ = net.get_shape().as_list()
		cells = self.cell_height * self.cell_width
		inputs = tf.reshape(net, [-1, cells, len(anchors), 5 + classes], name = 'inputs')
		inputs_sigmoid = tf.nn.sigmoid(inputs[:, :, :, :3])
		self.iou = inputs_sigmoid[:, :, :, 0]
		self.offset_xy = inputs_sigmoid[:, :, :, 1 : 3]
		self.wh = tf.identity(tf.exp(inputs[:, :, :, 3 : 5]) * np.reshape(anchors, [1, 1, len(anchors), -1]))
		self.prob = tf.nn.softmax(inputs[:, :, :, 5:])
		self.areas = tf.reduce_prod(self.wh, -1)
		_wh = self.wh / 2
		self.offset_xy_min = self.offset_xy - _wh
		self.offset_xy_max = self.offset_xy + _wh
		self.wh01 = self.wh / np.reshape([self.cell_width, self.cell_height], [1, 1, 1, 2])
		self.wh01_sqrt = tf.sqrt(self.wh01)
		self.coords = tf.concat([self.offset_xy, self.wh01_sqrt], -1)
		cell_xy = calc_cell_xy(self.cell_height, self.cell_width).reshape([1, cells, 1, 2])
		self.xy = cell_xy + self.offset_xy
		self.xy_min = cell_xy + self.offset_xy_min
		self.xy_max = cell_xy + self.offset_xy_max
		self.conf = tf.expand_dims(self.iou, -1) * self.prob
		self.inputs = net
		self.classes = classes
		self.anchors = anchors	