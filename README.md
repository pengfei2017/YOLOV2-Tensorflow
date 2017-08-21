# YOLOV2-Tensorflow
Convert from Darknet to Tensorflow

This code not able to train the model, just load the pretrained model from [here](https://pjreddie.com/darknet/yolo/)

Right it only support YOLO [VOC](https://pjreddie.com/media/files/yolo-voc.weights) and [COC](https://pjreddie.com/media/files/yolo.weights), and put in the same folder

## Please check settings.py before parse
```python
# support {COC, VOC} only
conf = config(conf = 'COC')
# rename the weight name to what you want
weight_name = 'yolo-coc.weights'

# YOLO hyperparameters
prob = 1
iou_best = 5
iou_normal = 1
coords = 1
threshold = 0.3
threshold_iou = 0.4
```

After that, you can
```bash
python python parse_darknet.py
```

Later if you want to detect on webcam
```bash
python live.py
```

If you want to detect on input/ folder
```bash
python image.py
```

## The code is still done yet, but the parser already working
