#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    视频资料来源：
    https://www.bilibili.com/video/BV1kW411W7pZ?p=22
"""

# ------------------------------------------- 7-2 循环神经网络rnn与长短时记忆神经网络简述 -------------------------------------------

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 每批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成截断的正态分布
    return tf.Variable(initial, name=name)


# 初始化偏执值
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    """
    tf.nn.conv2d 是TensorFlow里面实现卷积的函数
    tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
    :param input: 做卷积的输入图像，要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
            具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    :param filter: 相当于CNN中的卷积核，要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
            具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，
            第三维in_channels，就是参数input的第四维
    :param strides: 卷积时在图像每一维的步长，这是一个一维的向量，长度4，要求strides[0]=strides[3]=1，
            strides[1]和strides[2]分别代表x方向和y方向的步长
    :param padding: string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，
            “SAME”表示不加padding，“VALID”表示加padding。
    :return: 返回一个Tensor，该输出就是我们常说的feature map，类型不变，shape仍然是[batch, height, width, channels]这种形式
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    """
    tf.nn.max_pool 最大池化
    tf.nn.max_pool(value, ksize, strides, padding, name=None)
    :param x: 做卷积的输入图像，要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
            具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
    :ksize: 池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    :param strides: 与卷积层同理
    :param padding: string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，
            “SAME”表示不加padding，“VALID”表示加padding。
            返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
    :return: 返回一个Tensor，该输出就是我们常说的feature map，类型不变，shape仍然是[batch, height, width, channels]这种形式
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope("input"):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')  # 28*28
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope("x_image"):
        # 改变x的格式转为4D的向量[batch, in_height, in_width, in_channels]
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x-image')

with tf.name_scope("Conv1"):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope("W_conv1"):
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')  # 5*5的采样窗口，32个卷积核从一个平面（通道数为1）抽取特征
    with tf.name_scope("b_conv1"):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏执值

    # 把x_image和权重向量进行卷积，再加上偏执值，然后应用于relu激活函数
    with tf.name_scope("conv2d_1"):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope("relu"):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope("h_pool1"):
        h_pool1 = max_pool_2x2(h_conv1)  # 进行max-pooling

with tf.name_scope("Conv2"):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope("W_conv2"):
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')  # 5*5的采样窗口，64个卷积核从32个平面抽取特征
    with tf.name_scope("b_conv2"):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏执值

    # 把h_pool1和权值向量进行卷积，再加上偏执值，然后应用于relu激活函数
    with tf.name_scope("conv2d_2"):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope("relu"):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope("h_pool2"):
        h_pool2 = max_pool_2x2(h_conv2)  # 进行maxpooling

# 28*28的图像第一次卷积后还是28*28（padding='SAME'）,第一次池化后变成14*14，最后得到32张14*14的平面
# 第二次卷积后为14*14（padding='SAME'），第二次池化后为7*7，最后得到64张7*7的平面
# 经过上面的操作后得到64张7*7的平面

with tf.name_scope("fc1"):
    # 初始化第一个全连接层的权值
    with tf.name_scope("W_fc1"):
        W_fc1 = weight_variable([7 * 7 * 64, 100], name='W_fc1')  # 上一层有7*7*64个神经元，全连接层有100个神经元
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([100], name='b_fc1')  # 100个节点

    # 把池化层2的输出扁平化为1维
    with tf.name_scope("h_pool2_flat"):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')  # 因为h_pool2是一个tensor[-1, 7, 7, 64]
    # 求第一个全连接层的输出relu
    with tf.name_scope("wx_plus_b1"):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope("relu"):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob用来表示神经元的输出概率
    with tf.name_scope("keep_prob"):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope("h_fc1_drop"):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope("fc2"):
    # 初始化第二个全连接层
    with tf.name_scope("W_fc2"):
        W_fc2 = weight_variable([100, 10], name='W_fc2')  # 上一层有100个神经元，全连接层有10个神经元
    with tf.name_scope("b_fc2"):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope("wx_plus_b2"):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope("softmax"):
        # 计算输出softmax
        prediction = tf.nn.softmax(wx_plus_b2)

with tf.name_scope("cross_entropy"):
    # 交叉墒代价函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope("train"):
    # 使用AdamOptimizier进行优化
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("accuracy"):
    with tf.name_scope("correct_prediction"):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope("accuracy"):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('Data/TF/logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('Data/TF/logs/test', sess.graph)
    for i in range(1000):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:  # 记录测试集的summary与accuracy
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Train Accuracy= " + str(train_acc))

train_writer.close()
test_writer.close()

# ------------------------------------------- 7-3 循环神经网络lstm代码实现 -------------------------------------------

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 输入图片是28*28
n_inputs = 28  # 输入一行的长度 取决于图片一行28像素
max_time = 28  # 一共28行
lstm_size = 100  # 100个隐层单元
n_classes = 10  # 10个分类
batch_size = 50  # 每批次50样本
n_batch = mnist.train.num_examples // batch_size  # 计算一共多少批次

# 这里的none表示第一个维度可以任意，取决于放多少张图片
x = tf.placeholder(tf.float32, [None, 784])
# 正确的标签
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权值
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
# 初始化偏置值
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, Weights, Biases):
    # inputs = [batch_size,max_time,n_inputs]
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM基本CELL
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    """
        tf.compat.v1.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None,
                                    parallel_iterations=None, swap_memory=False, time_major=False, scope=None)
        
    """
    # final_state[0]是cell state
    # final_state[1]是hidden_state
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], Weights) + Biases)
    return results


# 计算RNN的返回结果
prediction = RNN(x, weights, biases)
# 损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 把结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + ",Testing Accuracy=" + str(acc))
