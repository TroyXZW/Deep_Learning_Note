#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    视频资料来源：
    https://www.bilibili.com/video/BV1kW411W7pZ?p=25
"""

# ------------------------------------------- 8-2 tensorflow模型保存和使用 -------------------------------------------

# -------------------------------------------
"""
    生成网络模型
    先定义一个简单的神经网络，用来训练模型，然后将模型保存下来，最后加载保存下来的模型进行检测，查看输出结果。
"""
# 模型训练和保存
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 每个批次100张照片
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 创建一个简单的神经网络，输入层784个神经元，输出层10个神经元
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用梯度下降法
trian_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果保存在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(trian_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))
    # 保存模型
    saver.save(sess, 'Data/TF/net/my_net.ckpt')  # 将训练好的模型及参数保存

# -------------------------------------------

"""
     载入训练好的模型
     上面的输出结果为0.098接近于0.1，原因是我们用的模型数据W、b为tf.zeros()接口初始化的数据，初始化都为0，所以结果都为随机猜的；
     下面的输出的结果为0.917，这个结果就比较接近训练时候模型的输出，这里我们用的模型数据的W、b为saver.restore加载后的。
"""

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  # 初始化为0的参数
    saver.restore(sess, 'Data/TF/net/my_net.ckpt')
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))  # 将已训练好的模型载入继续训练

# ------------------------------------------- 8-3 下载inception v3 google训练好的模型并解压 -------------------------------------------

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
import tarfile
import requests

# 模型下载地址
inception_pretrain_model_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# 模型存放地址
inception_pretrain_model_dir = "Data/TF/inception/inception_model"
if not os.path.exists(inception_pretrain_model_dir):
    os.makedirs(inception_pretrain_model_dir)

# 获取文件名以及文件路径
filename = inception_pretrain_model_url.split('/')[-1]
filepath = os.path.join(inception_pretrain_model_dir, filename)

# 下载模型
if not os.path.exists(filepath):
    print("download:", filename)
    r = requests.get(inception_pretrain_model_url, stream=True)
    with open(filepath, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
print("finish: ", filename)
# 解压文件
tarfile.open(filepath, 'r:gz').extractall(inception_pretrain_model_dir)

# 模型结构存放文件
log_dir = 'Data/TF/inception/inception_log'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# classify_image_graph_def.pb为google训练好的模型
inception_graph_def_file = os.path.join(inception_pretrain_model_dir, 'classify_image_graph_def.pb')
with tf.Session() as sess:
    # 创建一个图来保存google训练好的模型
    with tf.gfile.FastGFile(inception_graph_def_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')
    # 保存图的结构
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    writer.close()

# ------------------------------------------- 8-4 使用inception v3做各种图像分类识别 -------------------------------------------

"""
    获取inception v3模型数据，现在用下载好的模型进行简单的图片分类实验
    已有1000个类别，我们将自己找的图片作为输入，从而判别是1000类中的哪一个
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import os
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt


class NodeLookup(object):
    def __init__(self):
        label_lookup_path = 'Data/TF/inception/inception_model/imagenet_2012_challenge_label_map_proto.pbtxt'
        uid_lookup_path = 'Data/TF/inception/inception_model/imagenet_synset_to_human_label_map.txt'
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        # 加载分类字符串n************对应分类名称的文件
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        # 一行一行读取数据
        for line in proto_as_ascii_lines:
            # 去掉换行符
            line = line.strip('\n')
            # 按照‘\t’分割
            parsed_items = line.split('\t')
            # 获取分类编号
            uid = parsed_items[0]
            # 获取分类名称
            human_string = parsed_items[1]
            # 保存编号字符串n********与分类名称的映射关系
            uid_to_human[uid] = human_string
        # 加载分类字符串n**********对应分类编号1-1000的文件
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        node_id_to_uid = {}
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                # 获取分类编号1-1000
                target_class = int(line.split(':')[1])
            if line.startswith('  target_class_string:'):
                # 获取编号字符串n*********
                target_class_string = line.split(':')[1]
                # 保存分类编号1-1000与编号字符串n*******映射关系
                node_id_to_uid[target_class] = target_class_string[2:-2]
        # 建立分类编号1-1000对应分类名称的映射关系
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            # 获取分类名称
            name = uid_to_human[val]
            node_id_to_name[key] = name
        return node_id_to_name

    # 传入分类编号1-1000返回分类名称
    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


# 创建一个图来放google训练好的模型
with tf.gfile.FastGFile('Data/TF/inception/inception_model/classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with tf.Session() as sess:
    # 拿到softmax的op
    # 'softmax:0'这个名字，可以在网络中找到这个节点，它的名字就'(softmax)',
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    # 遍历目录
    for root, dirs, files in os.walk('Data/TF/inception/images/'):
        # root:Data/TF/inception/images/, dir:root下的子目录, files：root下的文件
        for file in files:
            # 载入图片
            image_data = tf.gfile.FastGFile(os.path.join(root, file), 'rb').read()
            # 运行softmax节点，向其中feed值
            # 可以在网络中找到这个名字，DecodeJpeg/contents，
            # 据此可以发现，根据名字取网络中op时，如果其名字带括号，就用括号内的名字，如果不带括号，就用右上角介绍的名字。
            # 而带个0，是默认情况，如果网络中出现同名节点，这个编号会递增
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式，predictions此时为二维
            predictions = np.squeeze(predictions)  # 把结果转换为1维数据，即概率值

            # 打印图片路径及名称
            print()
            image_path = os.path.join(root, file)
            print(image_path)
            # 显示图片
            img = Image.open(image_path)
            plt.imshow(img)
            plt.axis('off')
            plt.show()

            # 排序
            top_k = predictions.argsort()[-5:][::-1]
            node_lookup = NodeLookup()
            for node_id in top_k:
                # 获取分类名称
                human_string = node_lookup.id_to_string(node_id)
                # 获取该分类的置信度
                score = predictions[node_id]
                print('%s (score=%.5f)' % (human_string, score))
