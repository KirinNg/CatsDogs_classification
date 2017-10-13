import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import io,transform
import matplotlib.pyplot as plt

class Config(object):
    def __init__(self):
        self.text_path = "cat&dogs/annotations/test.txt"
        self.train_path = "cat&dogs/annotations/trainval.txt"
        self.image_path = "cat&dogs/images/"
        self.learning_rate = 0.0001
        self.num_train = 3680
        self.iter = 500

class Images(object):
    def __init__(self, config):
        self.config = config
        train_fin = open(self.config.train_path, "r")
        self.data = np.zeros([3680,500,500,3])
        self.label = np.zeros([3680,3])
        i = 0
        for line_train in train_fin.readlines():
            line_train = line_train.strip().split(' ')
            path = self.config.image_path + line_train[0] + ".jpg"
            img = io.imread(path)
            img = transform.resize(img,(500,500,3))
            #io.imsave("try.jpg",img)
            self.data[i,:,:,:] = img
            self.label[i,0] = line_train[1]
            self.label[i,1] = line_train[2]
            self.label[i,2] = line_train[3]
            i = i+1

class Model(object):
    def __init__(self, config, image):
        self.config = config
        self.image = image
        ########## INPUT ##########
        self.input = tf.placeholder(tf.float32, [None,500,500,3])
        self.label = tf.placeholder(tf.float32, [None,3])
        ###########################
        self._make_model()
        self.loss = self._make_loss()
        self.train_step = tf.train.GradientDescentOptimizer(self.config.learning_rate).minimize(self.loss)

    def _make_model(self):
        def conv(input_data, input_size, output_size):
            w = tf.Variable(tf.random_normal([3,3,input_size,output_size],stddev = 0.1))
            b = tf.Variable(tf.constant(0.1, shape = [output_size]))
            return tf.nn.relu(tf.nn.conv2d(input_data, w, [1,1,1,1], padding = 'SAME') + b)

        def pooling(input_data):
            return tf.nn.avg_pool(input_data, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

        def addLayer(input_data, input_size, output_size):
            W = tf.Variable(tf.random_normal([input_size,output_size],stddev = 0.1))
            basis = tf.Variable(tf.constant(0.1,shape = [output_size]))
            return tf.matmul(input_data, W) + basis

        #500*500*3
        lay1 = conv(self.input, 3, 4)
        lay1_p = pooling(lay1)
        #250*250*4
        lay2 = conv(lay1_p, 4, 16)
        lay2_p = pooling(lay2)
        #125*125*16
        lay3 = conv(lay2_p, 16, 64)
        lay_3 = pooling(lay3)
        #63*63*64
        lay4 = conv(lay_3, 64, 256)
        lay4_p = pooling(lay4)
        #32*32*256
        lay5 = conv(lay4_p, 256, 256)
        lay5_p = pooling(lay5)
        #16*16*256
        lay6 = conv(lay5_p, 256, 256)
        lay6_p = pooling(lay6)
        #8*8*256
        lay7 = conv(lay6_p, 256, 256)
        lay7_p = pooling(lay7)
        #4*4*256
        array7 = tf.reshape(lay7_p, [-1,4096])
        #4096
        #out_put
        self.output = addLayer(array7, 4096, 3)

    def _make_loss(self):
        def logic_loss(y_out, ys):
            return -tf.reduce_sum(ys*tf.log(y_out))
        return logic_loss(self.output, self.label)

    def run_model(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        for i in range(self.config.iter):
            self.sess.run(self.train_step, {self.input:self.image.data, self.label:self.image.label})
            print(self.sess.run(self.loss, {self.input:self.image.data, self.label:self.image.label}))

# MAIN 部分
config = Config()
image = Images(config)
model = Model(config, image)
model.run_model()