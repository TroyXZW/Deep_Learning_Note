#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
    视频资料来源：
    https://www.bilibili.com/video/BV1kW411W7pZ?p=11
"""

# ------------------------------------------- 4-2 多层网络通过防止过拟合，增加模型的准确率 -------------------------------------------
"""
    防止过拟合：增加数据集、正则化、dropout
    dropout可能会让迭代时收敛慢，但有效的减小了过拟合
"""
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# 创建一个简单的神经网络
"""网络为784个输入，10个输出，隐层第一层200个，第二层100个"""
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))  # 这里我们使用一个截断的正态分布初始化W
b1 = tf.Variable(tf.zeros([1, 200]))
# 激活层
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)  # 激活函数为双曲正切函数
# drop层
L1_drop = tf.nn.dropout(L1, keep_prob)  # keep_prob设置？%的神经元是工作的

# 第二层
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 100]))
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 第三层
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 二次代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用剃度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(20):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})  # keep_prob: 0.5，相当于有70%的神经元在工作

        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        print("Iter" + str(epoch) + ",Testing Accuracy " + str(test_acc) + "Training Accuracy " + str(train_acc))

# ------------------------------------------- 4-3 修改优化器进一步提升准确率 -------------------------------------------
"""
    tensorflow中的优化器有很多种，最常用的是AdamOptimizer，这里就通过adma和衰减的学习率加上之前学的多层结构，使手写数字模型准确率达到98%以上
"""
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("Data/TF/MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
lr = tf.Variable(0.001, dtype=tf.float32)

# 创建一个简单的神经网络
"""网络为784个输入，10个输出，隐层第一层200个，第二层100个"""
W1 = tf.Variable(tf.truncated_normal([784, 200], stddev=0.1))  # 这里我们使用一个截断的正太分布初始化W
b1 = tf.Variable(tf.zeros([1, 200]))
# 激活层
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)  # 激活函数为双曲正切函数
# drop层
L1_drop = tf.nn.dropout(L1, keep_prob)

# 第二层
W2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([1, 100]))
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prob)

# 第三层，即输出层
W3 = tf.Variable(tf.truncated_normal([100, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([1, 10]))
prediction = tf.nn.softmax(tf.matmul(L2_drop, W3) + b3)

# 交叉熵代价函数
# loss=tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用Adam
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)#得到98.3的正确率
# train_step = tf.train.AdadeltaOptimizer(0.1).minimize(loss)  #97.34的准确率
# train_step = tf.train.AdamOptimizer().minimize(loss)  # 97.8
# train_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(loss) #98.06
# train_step = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(loss) # 98.29

# 初始化变量
init = tf.global_variables_initializer()

# 结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))  # 将0.001 * (0.95 ** epoch)赋值给lr，每迭代一次，lr变为原来的0.95倍
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})

        learning_rate = sess.run(lr)
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        # print("Iter" + str(epoch) + ",learning rate " + str(learning_rate) + ",Testing Accuracy " + str(
        #     test_acc) + "Training Accuracy " + str(train_acc))
        print("Iter{}, Learning Rate:{}, Training Accuracy:{}, Testing Accuracy:{}" \
              .format(str(epoch), str(learning_rate), str(train_acc), str(test_acc)))
