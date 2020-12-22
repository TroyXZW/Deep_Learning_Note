# -*- coding: utf-8 -*-

import tensorflow.compat.v1 as tf
import numpy as np
import sys
import pandas as pd

# --------------------------------- tf.sequence_mask ---------------------------------
tf.disable_v2_behavior()
mask1 = tf.sequence_mask(list(range(1, 3)), 3, dtype=tf.float32)  # 矩阵list的长为行数，5为列数，按list填充1
mask2 = tf.expand_dims(mask1, -1)  # 给-1维增加一个维度
mask2_0 = tf.expand_dims(mask1, 0)  # 给最外层加括号
mask2_1 = tf.expand_dims(mask1, 1)  # 给次外层加括号
mask3 = tf.tile(mask2, [1, 1, 5])  # 给-1维扩充五倍
mask3_0 = tf.tile(mask2, [1, 2, 5])  # 给次外层充两倍
mask3_1 = tf.tile(mask2, [2, 1, 5])  # 给最外层扩充两倍
with tf.Session() as sess:
    print(mask1.eval())
    print(mask2.eval())
    print(mask2_0.eval())
    print(mask2_1.eval())
    print(mask3.eval())
    print(mask3_0.eval())
    print(mask3_1.eval())

# --------------------------------- tf.concat ---------------------------------
a = [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]
aa = tf.convert_to_tensor(a)
b = [[[-1, -2, -3], [-4, -5, -6]], [[-7, -8, -9], [-10, -11, -12]]]
bb = tf.convert_to_tensor(b)
cc = tf.concat([aa, bb], axis=-1)
cc_0 = tf.concat([aa, bb], axis=0)
cc_1 = tf.concat([aa, bb], axis=1)
cc_2 = tf.concat([aa, bb], axis=2)  # 与axis=-1同理
with tf.Session() as sess:
    print(aa.eval())
    print(bb.eval())
    print(cc.eval())
    print(cc_0.eval())
    print(cc_1.eval())
    print(cc_2.eval())

x = [[1, 2, 3], [4, 5, 6]]
y = [[7, 8], [11, 12]]
xx = tf.convert_to_tensor(x)
yy = tf.convert_to_tensor(y)
zz = tf.concat([xx, yy], axis=-1)
with tf.Session() as sess:
    print(zz.eval())


# --------------------------------- tf.gather ---------------------------------
def build_map(df, col_name):
    key = df[col_name].unique().tolist()
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


a = {
    'uid': [22, 33, 44, 13, 423, 4233, 53, 45, 657, 75],
    'pid': [2314, 4523, 5456, 6546, 7658, 6432, 7575, 8654, 6422, 5221],
    'name': ['张三'] * 3 + ['李四'] * 3 + ['王五'] * 2 + ['⼩王'] + ['李四'], }
df = pd.DataFrame(a)
name_key = build_map(df, "name")
pid_key = build_map(df, "pid")
print(df)
df1 = df[['pid', 'name']].drop_duplicates()
print(df1)
brand_list = df1['name'].tolist()
print(brand_list)
brand_list = tf.convert_to_tensor(brand_list, dtype=tf.int32)
hist_i = [[3, 7], [2, 5]]
hist_b = tf.gather(brand_list, hist_i)  # 给定索引hist_i，从brand_list取出
with tf.Session() as sess:
    print(sess.run(brand_list))
    print(sess.run(hist_b))


# --------------------------------- tf.nn.embedding_lookup ---------------------------------
def test1():
    input_ids = tf.placeholder(tf.int32, shape=[None], name="input_ids")
    embedding = tf.Variable(np.identity(5, dtype=np.int32))  # np.identity(5, dtype=np.int32)生成对角阵
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("embedding=\n", embedding.eval())
    print("input_embedding=\n",
          sess.run(input_embedding, feed_dict={input_ids: [1, 2, 3, 0, 3, 2, 1]}))  # 利用feed_dict提供的索引从input_embedding取


def test2():
    input_ids_1 = tf.placeholder(dtype=tf.int32, shape=[3, 2])
    input_ids_2 = tf.placeholder(dtype=tf.int32, shape=[3, 2])
    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding1 = tf.nn.embedding_lookup(embedding, input_ids_1)
    input_embedding2 = tf.nn.embedding_lookup(embedding, input_ids_2)
    input_embedding = tf.concat([input_embedding1, input_embedding2],
                                axis=2)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("embedding=\n", embedding.eval())
    print("input_embedding=\n", sess.run(input_embedding, feed_dict={input_ids_1: [[1, 2], [2, 1], [3, 3]],
                                                                     input_ids_2: [[3, 3], [2, 1], [1, 2]]}))


def test3():
    input_ids = tf.placeholder(dtype=tf.int32, shape=[3, 1])
    embedding = tf.Variable(np.identity(5, dtype=np.int32))
    input_embedding = tf.nn.embedding_lookup(embedding, input_ids)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("embedding=\n", embedding.eval())
    print("input_embedding=\n", sess.run(input_embedding, feed_dict={input_ids: [[1], [2], [3]]}))


test1()
test2()
test3()

