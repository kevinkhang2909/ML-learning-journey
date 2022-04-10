import tensorflow as tf
import numpy as np
import os
import random


class ArcMarginProduct(tf.keras.layers.Layer):
    """
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    """

    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(np.pi - m)
        self.mm = tf.math.sin(np.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class GeMPoolingLayer(tf.keras.layers.Layer):
    """
    Implements Generalized-Mean Pooling layer
    Reference:
        https://arxiv.org/pdf/1711.02512.pdf
    """

    def __init__(self, p=1., eps=1e-6):
        super().__init__()
        self.p = p
        self.eps = eps

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=self.eps, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1. / self.p)
        return inputs

    def get_config(self):
        return {
            'p': self.p,
            'eps': self.eps
        }


def get_model(params, n_class, img_size, cosine_lr_fn):
    backbone = tf.keras.applications.EfficientNetB3(weights='imagenet', include_top=False)
    margin = ArcMarginProduct(
        n_classes=n_class,
        s=params['s'],
        m=params['m'],
        name='arc_margin_product',
        dtype='float32'
    )

    inp = tf.keras.layers.Input(shape=img_size + (3,), name='image')
    label = tf.keras.layers.Input(shape=(), name='label')
    x = tf.keras.applications.efficientnet.preprocess_input(inp)
    x = backbone(x)
    x = GeMPoolingLayer()(x)
    x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(), activation=None)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = margin([x, label])

    output = tf.keras.layers.Softmax(dtype='float32')(x)

    model = tf.keras.models.Model(inputs=[inp, label], outputs=[output])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_lr_fn),
        loss=[tf.keras.losses.SparseCategoricalCrossentropy()],
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    return model


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
