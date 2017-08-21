import numpy as np

def load_text(text):
	with open(text, 'r') as fopen:
		return fopen.read().split('\n')

class config:
	def __init__(self, conf = 'COC'):
		self.reject_classes = []
		if conf == 'COC':
			self.width = 608
			self.height = 608
			self.classes = 80
			self.class_name = load_text('80')
			self.anchors = np.array([0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828])
		elif conf == 'VOC':
			self.width = 416
			self.height = 416
			self.classes = 20
			self.class_name = load_text('20')
			self.anchors = np.array([1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071])
		else:
			print 'darknet type not supported, exiting..'
			exit(0)

# support {COC, VOC} only
conf = config(conf = 'COC')
weight_name = 'yolo-coc.weights'

# YOLO hyperparameters
prob = 1
iou_best = 5
iou_normal = 1
coords = 1
threshold = 0.3
threshold_iou = 0.4


