#!/usr/bin/env python
# -*- coding:utf-8 -*-

# ------------------------------------------- 3-1 tensorflow非线性回归 -------------------------------------------

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层，十个神经元
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
biases_L1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # 1.x相当于形参
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 定义神经网络输出层，一个神经元
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))  # 2.相当于形参
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})  # 3.将实参x与y传入函数

    # 获取预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()

# ------------------------------------------- 3-2 MNIST手写数字分类simple版 -------------------------------------------

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
"""网络为784个输入，10个输出，没用隐层"""
x = tf.placeholder(tf.float32, [None, 784])  # 因为批次是100，所以其实是[100, 784]
y = tf.placeholder(tf.float32, [None, 10])  # 数字是0-9，所以有10个标签

# 创建一个简单的神经网络，无隐藏层
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
# loss = tf.reduce_mean(tf.square(y - prediction))
# softmax交叉熵
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast将bool值转换为float32

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(acc))

"""
    调优：
    batch_size = 100可修改
    增加隐层个数，以及各隐层的节点数
    激活函数换为Relu等
    初始化的权值：不一定初始化为0
    代价函数：可改为二次代价函数、交叉熵(cross-entropy)、对数似然代价函数（log-likelihood cost）等
    梯度下降法的学习率
    将代价函数的优化方式换为其他：梯度下降等
    epoch次数
    
"""
