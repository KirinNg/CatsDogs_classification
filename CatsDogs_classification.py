import tensorflow as tf
from PIL import Image
import numpy as np
import os
from skimage import io,transform
import skimage
import matplotlib.pyplot as plt

class Config(object):
    def __init__(self):
        self.text_path = "cat&dogs/annotations/test.txt"
        self.train_path = "cat&dogs/annotations/trainval.txt"
        self.image_path = "cat&dogs/images/"
        self.learning_rate = 0.01
        self.num_train = 3680
        self.test_num = 3669
        self.iter = 100000
        self.save_path = "cat&dogs/model.ckpt"
        self.choose = 0
        self.wide = 224

class Images_read(object):
    def __init__(self, config):
        self.config = config
        train_fin = open(self.config.train_path, "r")
        self.data = np.zeros([self.config.num_train,self.config.wide,self.config.wide,3])#
        self.label_0 = np.zeros([self.config.num_train,37])#
        self.label_1 = np.zeros([self.config.num_train,2])
        i = 0
        j = 0
        for line_train in train_fin.readlines():
            if j%1==config.choose:
                line_train = line_train.strip().split(' ')
                path = self.config.image_path + line_train[0] + ".jpg"
                img = io.imread(path)
                #img = transform.resize(img,(self.config.wide,self.config.wide,3))
                #img = skimage.img_as_uint(img)
                #io.imsave(path,img)
                self.data[i,:,:,:] = img
                self.label_0[i,int(line_train[1])-1] = 1
                self.label_1[i,int(line_train[2])-1] = 1
                #self.label[i,2] = line_train[3]
                i += 1
                print(i)
            j += 1
        np.save("Image_data",self.data.astype(int))
        np.save("Image_label_0",self.label_0.astype(int))
        np.save("Image_label_1",self.label_1.astype(int))

class Images_test_read(object):
    def __init__(self, config):
        self.config = config
        train_fin = open(self.config.text_path, "r")
        self.data = np.zeros([3669,self.config.wide,self.config.wide,3])#
        self.label_0 = np.zeros([3669,37])#
        self.label_1 = np.zeros([3669,2])
        i = 0
        j = 0
        for line_train in train_fin.readlines():
            if j%1==config.choose:
                line_train = line_train.strip().split(' ')
                path = self.config.image_path + line_train[0] + ".jpg"
                img = io.imread(path)
                #img = transform.resize(img,(self.config.wide,self.config.wide,3))
                #img = skimage.img_as_uint(img)
                #io.imsave(path,img)
                self.data[i,:,:,:] = img
                self.label_0[i,int(line_train[1])-1] = 1
                self.label_1[i,int(line_train[2])-1] = 1
                #self.label[i,2] = line_train[3]
                i += 1
            j += 1
            print(i)
        np.save("Image_data_text",self.data.astype(int))
        np.save("Image_label_0_text",self.label_0.astype(int))
        np.save("Image_label_1_text",self.label_1.astype(int))

class Images(object):
    def __init__(self):
        self.data = np.load("Image_data.npy").astype(int)
        self.label_0 = np.load("Image_label_0.npy").astype(int)
        self.label_1 = np.load("Image_label_1.npy").astype(int)

class Images_test(object):
    def __init__(self):
        self.data = np.load("Image_data_text.npy").astype(int)
        self.label_0 = np.load("Image_label_0_text.npy").astype(int)
        self.label_1 = np.load("Image_label_1_text.npy").astype(int)

class Model(object):
    def __init__(self, config, image, image_test):
        self.config = config
        self.flag = 0
        #self.image = image
        ########## INPUT ##########
        self.input = tf.placeholder(tf.float32, [None,self.config.wide,self.config.wide,3])
        self.label_0 = tf.placeholder(tf.float32, [None,37])
        self.label_1 = tf.placeholder(tf.float32, [None,2])
        self.keep_prob = tf.placeholder(tf.float32)
        ###########################
        self._make_model()
        self.loss = self._make_loss()
        with tf.device("/gpu:0"):##########
            self.train_step = tf.train.RMSPropOptimizer(self.config.learning_rate).minimize(self.loss)

    def _make_model(self):
        def conv(input_data, input_size, output_size, kernel_size = 3, stride = 1):
            w = tf.Variable(tf.random_normal([kernel_size,kernel_size,input_size,output_size],stddev = 0.0001))
            b = tf.Variable(tf.constant(0.0001, shape = [output_size]))
            return tf.nn.relu(tf.nn.conv2d(input_data, w, [1,stride,stride,1], padding = 'SAME') + b)

        def pooling(input_data):
            return tf.nn.max_pool(input_data, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

        def addLayer(input_data, input_size, output_size):
            W = tf.Variable(tf.random_normal([input_size,output_size],stddev = 0.0001))
            basis = tf.Variable(tf.constant(0.0001,shape = [output_size]))
            return tf.matmul(input_data, W) + basis
        """
        #128*128*3
        lay1 = conv(self.input, 3, 4)
        lay1_p = pooling(lay1)
        #64*64*4
        lay2 = conv(lay1_p, 4, 16)
        lay2_p = pooling(lay2)
        #32*32*16
        lay3 = conv(lay2_p, 16, 64)
        lay3_p = pooling(lay3)
        #16*16*64
        lay4 = conv(lay3_p, 64, 256)
        lay4_p = pooling(lay4)
        #8*8*256
        lay5 = conv(lay4_p, 256, 512)
        lay5_p = pooling(lay5)
        #4*4*256
        lay6 = conv(lay5_p, 512, 1024)
        lay6_p = pooling(lay6)
        #2*2*256
        lay7 = conv(lay6_p, 1024, 1024)
        lay7_p = pooling(lay7)
        #1*1*256
        array7 = tf.reshape(lay7_p, [-1,1024])
        #1024
        #out_put
        array8 = tf.nn.relu(addLayer(array7, 1024, 256))
        self.output = tf.nn.dropout(addLayer(array8, 256, 37), self.keep_prob)
        """
        """
        #Alex_Net 
        #DFD_1 227*227*3
        lay1 = conv(self.input,3,96,11,4)
        lay1_p = pooling(lay1)
        #DFD_2 27*27*96
        lay2 = conv(lay1_p,96,256,5,1)
        lay2_p = pooling(lay2)
        #DFD_3 13*13*256
        lay3 = conv(lay2_p,256,384,3,1)
        #DFD_4 13*13*384
        lay4 = conv(lay3,384,384,3,1)
        #DFD_5 13*13*384
        lay5 = conv(lay4,384,256,3,1)
        lay5_p = pooling(lay5)
        #DFD_6 6*6*256
        array6 = tf.reshape(lay5_p,[-1,9216])
        fc6 = tf.nn.relu(addLayer(array6,9216,4096))
        drop6 = tf.nn.dropout(fc6,self.keep_prob)
        #DFD_7 -1*4096
        fc7 = tf.nn.relu(addLayer(drop6,4096,4096))
        drop7 = tf.nn.dropout(fc7,self.keep_prob)
        #DFD_8 -1*4096
        self.output = addLayer(drop7,4096,39)
        #FINISH
        """
        #VGG_19 224*224*3
        lay1 = conv(self.input, 3, 64)
        lay2 = pooling(conv(lay1, 64, 64))
        #112*112*64
        lay3 = conv(lay2, 64,128)
        lay4 = pooling(conv(lay3, 128, 128))
        #56*56*128
        lay5 = conv(lay4, 128, 256)
        lay6 = conv(lay5, 256, 256)
        lay7 = conv(lay6, 256, 256)
        lay8 = pooling(conv(lay7, 256, 256))
        #28*28*256
        lay9 = conv(lay8, 256, 512)
        lay10 = conv(lay9, 512, 512)
        lay11 = conv(lay10, 512, 512)
        lay12 = pooling(conv(lay11, 512, 512))
        #14*14*512
        lay13 = conv(lay12, 512 ,512)
        lay14 = conv(lay13, 512, 512)
        lay15 = conv(lay14, 512, 512)
        lay16 = pooling(conv(lay15, 512, 512))
        #7*7*512
        array16 = tf.reshape(lay16,[-1,7*7*512])
        lay17 = tf.nn.dropout(addLayer(array16, 7*7*512, 4096),self.keep_prob)
        lay18 = tf.nn.dropout(addLayer(lay17, 4096, 4096),self.keep_prob)
        lay19 = addLayer(lay18, 4096, 1000)
        self.output = addLayer(lay19, 1000, 39)

    def _make_loss(self):
        def loss(y_out, ys):
            return tf.reduce_sum(tf.square(y_out-ys))
        def logic_loss(y_out, ys):
            return -tf.reduce_sum(ys*tf.log(y_out))
        def tfsoft_with_entrance(y_out, ys):
            return tf.nn.softmax_cross_entropy_with_logits(None,ys,y_out)
        return tf.reduce_sum(tfsoft_with_entrance(self.output[:,0:37], self.label_0)) + 16*tf.reduce_sum(tfsoft_with_entrance(self.output[:,37:39], self.label_1))

    def _get_feed(self):
        self.min_batch_data = np.zeros([23,224,224,3])
        self.min_batch_label_0 = np.zeros([23,37])
        self.min_batch_label_1 = np.zeros([23,2])
        for i in range(23):
            self.min_batch_data[i:i+1,:,:,:] = image.data[self.flag*23:self.flag*23+1,:,:,:]
            self.min_batch_label_0[i:i+1,:] = image.label_0[self.flag*23:self.flag*23+1,:]
            self.min_batch_label_1[i:i+1,:] = image.label_1[self.flag*23:self.flag*23+1,:]
            self.flag += 1
            self.flag = self.flag%160
        return {self.input:self.min_batch_data, self.label_1:self.min_batch_label_1, self.label_0:self.min_batch_label_0, self.keep_prob:0.5}

    def run_model(self):
        #config = tf.ConfigProto()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allocator_type = 'BFC'
        config.gpu_options.per_process_gpu_memory_fraction = 0.70
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        ###
        #self.restore_model(self.config.save_path)
        ###
        correct_p = tf.equal(tf.argmax(self.output[:,0:37],1),(tf.argmax(self.label_0,1)))
        accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
        for i in range(self.config.iter):
            for j in range(160):
                if i>100:
                    self.config.learning_rate = 0.001
                if i>500:
                    self.config.learning_rate = 0.0001
                print()
                with tf.device("/gpu:0"):##########
                    feed = self._get_feed()
                    #feed_test = {self.input:image_test.data, self.label_1:image_test.label_1, self.label_0:image_test.label_0, self.keep_prob:1}
                    self.sess.run(self.train_step, feed)
                with tf.device("/gpu:0"):
                    print("i:%d j:%d loss:%.6f accur:%f" %(i,j,self.sess.run(self.loss, feed),self.sess.run(accuracy, feed)))
                    #print("loss:%.6f accur:%f" %(self.sess.run(self.loss, feed),self.sess.run(accuracy, feed)))
            print()
            print("accur_test:%f" %(self.test()/3669))
            if i%10 == 0:
                if i>0:
                    self.save(self.config.save_path)
        #print()
        #self.test()

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def test(self):
        #self.sess = tf.Session()
        #self.sess.run(tf.global_variables_initializer())
        correct_p = tf.equal(tf.argmax(self.output[:,0:37],1),(tf.argmax(self.label_0,1)))
        accuracy = tf.reduce_mean(tf.cast(correct_p, "float"))
        #self.restore_model(self.config.save_path)
        accur = 0
        for i in range(3669):
            feed_test = {self.input:image_test.data[i:i+1,:,:,:], self.label_1:image_test.label_1[i:i+1,:], self.label_0:image_test.label_0[i:i+1,:], self.keep_prob:1}
            accur += self.sess.run(accuracy, feed_test)
        #print("accur_test%.10f" %(accur/3669))
        return accur

# MAIN 部分
config = Config()
#Images_read(config)
#Images_test_read(config)
image = Images()
image_test = Images_test()
model = Model(config, image, image_test)
model.run_model()
model.save(config.save_path)
model.test()