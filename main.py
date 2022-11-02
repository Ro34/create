# -*- coding: utf-8 -*-
"""
Created on 2020-10-31

@author: 李运辰
"""
# 导入数据包
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import glob
import os

# # 输入
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.astype('float32')

# # 数据预处理
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

# 归一化 到【-1，1】
train_images = (train_images - 127.5) / 127.5

BTATH_SIZE = 256
BUFFER_SIZE = 60000

# 输入管道
datasets = tf.data.Dataset.from_tensor_slices(train_images)

# 打乱乱序，并取btath_size
datasets = datasets.shuffle(BUFFER_SIZE).batch(BTATH_SIZE)


# # 生成器模型
def generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,), use_bias=False))
    # Dense全连接层，input_shape=(100,)长度100的随机向量，use_bias=False，因为后面有BN层
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 激活

    # 第二层
    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 激活

    # 输出层
    model.add(layers.Dense(28 * 28 * 1, use_bias=False, activation='tanh'))
    model.add(layers.BatchNormalization())

    model.add(layers.Reshape((28, 28, 1)))  # 变成图片 要以元组形式传入

    return model


# # 辨别器模型
def discriminator_model():
    model = keras.Sequential()
    model.add(layers.Flatten())

    model.add(layers.Dense(512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 激活

    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 激活

    model.add(layers.Dense(1))  # 输出数字，>0.5真实图片

    return model


# # loss函数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # from_logits=True因为最后的输出没有激活


# # 生成器损失函数
def generator_loss(fake_out):  # 希望fakeimage的判别输出fake_out判别为真
    return cross_entropy(tf.ones_like(fake_out), fake_out)


# # 判别器损失函数
def discriminator_loss(real_out, fake_out):  # 辨别器的输出 真实图片判1，假的图片判0
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss


# # 优化器

generator_opt = tf.keras.optimizers.Adam(1e-4)  # 学习速率
discriminator_opt = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 500
noise_dim = 100  # 长度为100的随机向量生成手写数据集
num_exp_to_generate = 16  # 每步生成16个样本
seed = tf.random.normal([num_exp_to_generate, noise_dim])  # 生成随机向量观察变化情况

# # 训练
generator = generator_model()
discriminator = discriminator_model()


# # 定义批次训练函数
def train_step(images):
    noise = tf.random.normal([num_exp_to_generate, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # 判别真图片
        real_out = discriminator(images, training=True)
        # 生成图片
        gen_image = generator(noise, training=True)
        # 判别生成图片
        fake_out = discriminator(gen_image, training=True)

        # 损失函数判别
        gen_loss = generator_loss(fake_out)
        disc_loss = discriminator_loss(real_out, fake_out)

    # 训练过程
    # 生成器与生成器可训练参数的梯度
    gradient_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradient_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # 优化器优化梯度
    generator_opt.apply_gradients(zip(gradient_gen, generator.trainable_variables))
    discriminator_opt.apply_gradients(zip(gradient_disc, discriminator.trainable_variables))


# # 可视化
def generator_plot_image(gen_model, test_noise):
    pre_images = gen_model(test_noise, training=False)
    # 绘图16张图片在一张4x4
    fig = plt.figure(figsize=(4, 4))
    for i in range(pre_images.shape[0]):
        plt.subplot(4, 4, i + 1)  # 从1开始排
        plt.imshow((pre_images[i, :, :, 0] + 1) / 2, cmap='gray')  # 归一化，灰色度
        plt.axis('off')  # 不显示坐标轴
    plt.show()


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)
        # print('第'+str(epoch+1)+'次训练结果')
        if epoch % 10 == 0:
            print('第' + str(epoch + 1) + '次训练结果')
            generator_plot_image(generator, seed)


train(datasets, EPOCHS)