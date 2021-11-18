import tensorflow as tf
from tensorflow.keras import losses

from .content_loss import ContentLoss


def k_l_divergence(mean, logvar):
    return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))


def generator_loss(y):
    return -tf.reduce_mean(y)


class DiscriminatorHingeLoss(losses.Loss):
    def __init__(self, reduction=..., name=None):
        super().__init__(reduction=reduction, name=name)
        self.hinge_loss = losses.Hinge()

    def call(self, y, is_real):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(y, label)


class FeatureMatchingLoss(losses.Loss):
    def __init__(self, reduction=..., name=None):
        super().__init__(reduction=reduction, name=name)
        self.mean_absolute_error = losses.MeanAbsoluteError()

    def call(self, real_features, fake_features):
        return tf.reduce_mean(self.mean_absolute_error(real_features, fake_features))
