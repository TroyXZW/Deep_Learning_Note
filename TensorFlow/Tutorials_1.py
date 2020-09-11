#!/usr/bin/env python 
# -*- coding:utf-8 -*-

"""
    视频资料来源：
    https://www.bilibili.com/video/BV1kW411W7pZ?p=4
"""

# ------------------------------------------- 2-1 tensorflow中的图 -------------------------------------------

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

m1 = tf.constant([[3, 3]])  # 创建一个常量op
m2 = tf.constant([[2], [3]])
product = tf.matmul(m1, m2)  # 创建一个矩阵乘法op，把m1和m2传入
print(product)  # 这里将输出一个Tensor
'''
    #定义一个会话1，启动默认图
    sess=tf.Session()  #旧版本
    result=sess.run(product)    #调用run方法来执行矩阵乘法
    print(result)
    sess.close()
'''
# 另一种定义会话的方式（常用）
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

# ------------------------------------------- 2-2 tensorflow变量的使用 -------------------------------------------

x = tf.Variable([1, 2])
a = tf.constant([3, 3])

sub = tf.subtract(x, a)  # 增加一个减法op
add = tf.add(x, sub)  # 增加一个加法op

# 注意变量再使用之前要再sess中做初始化，但是下边这种初始化方法不会指定变量的初始化顺序
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#################分割线#####################
# 创建一个名字为‘counter’的变量 初始化0
state = tf.Variable(0, name='counter')
new_value = tf.add(state, 1)  # 创建一个op，作用是使state加1
# tf.assign 赋值操作，直接用接口而不是用等号，这也就说明tensorflow不是用python实现的，是有c++实现的，你如果想把python中的值在session中修改，必须要用这种接口的方式
update = tf.assign(state, new_value)  # 赋值op，将new_value赋值给state
init = tf.global_variables_initializer()  # 在python中确实初始化了，但是在tf的session中你并没有初始化

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))  # 先打印出0
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))

# ------------------------------------------- 2-3 tensorflow中的Fetch、Feed -------------------------------------------

# Fetch概念 在session中同时运行多个op
input1 = tf.constant(3.0)  # constant()是常量不用进行init初始化
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])  # 这里的[]就是Fetch操作，多个op同时运行
    print(result)

# Feed
# 创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
# 定义乘法op，op被调用时可通过Feed的方式将input1、input2传入
output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # feed的数据以字典的形式传入，即不用在上面给imput1和imput2复制，可以通过下面的feed_dict喂给
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))

# ------------------------------------------- 2-4 拟合线性函数的k和b -------------------------------------------

import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 0.2  # 这里我们设定已知直线的k为0.1 b为0.2得到y_data

# 构造一个线性模型
b = tf.Variable(0.)  # 变量，后期的梯度下降对其进行改变
k = tf.Variable(0.)  # 变量，后期的梯度下降对其进行改变
y = k * x_data + b  # 初始的拟合曲线

# 二次代价函数（白话：两数之差平方后取平均值）
loss = tf.reduce_mean(tf.square(y_data - y))
# 定义一个梯度下降法来进行训练的优化器（其实就是按梯度下降的方法改变线性模型k和b的值，注意这里的k和b一开始初始化都为0.0，后来慢慢向0.1、0.2靠近）
optimizer = tf.train.GradientDescentOptimizer(0.2)  # 这里的0.2是梯度下降的系数也可以是0.3...
# 最小化代价函数(训练的方式就是使loss值最小，loss值可能是随机初始化100个点与模型拟合出的100个点差的平方相加...等方法)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))  # 这里使用fetch的方式只是打印k、b的值，每20次打印一下，改变k、b的值是梯度下降优化器的工作
