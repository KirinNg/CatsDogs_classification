import tensorflow as tf
from PIL import Image
import numpy as np
import os

class Config(object):
    def __init__(self):
        self.text_path = "cat&dogs/annotations/test.txt"
        self.train_path = "cat&dogs/annotations/trainval.txt"
        self.image_path = "cat&dogs/images/"
        self.learning_rate = 0.0001
        self.num_train = 3680

class Images(object):
    def __init__(self, config):
        self.config = config
        train_fin = open(self.config.train_path, "r")
        self.data = []
        self.label = []
        for line_train in train_fin.readlines():
            line_train = line_train.strip().split(' ')
            path = self.config.image_path + line_train[0] + ".jpg"
            img = Image.open(path)
            img = img.resize((500,500))
            self.data.append(img)
            self.label.append(line_train[1:])

class Model(object):
    def __init__(self, config):
        self.config = config
        self._make_model()
        self.loss = self._make_loss()
        self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

    def _make_model(self):
        def conv2(input):
            return tf.nn.conv2d(input)
        def pooling(input):
            return tf.nn.avg_pool(input)
        pass

    def _make_loss(self):
        pass

# MAIN 部分
config = Config()
image = Images(config)
model = Model(config)