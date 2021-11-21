import tensorflow as tf
from tensorflow.keras import losses, applications, Model


class ContentLoss(losses.Loss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        # The encoder layers weights are basically an increasing geometric progression (1 / 2 ^ n).
        # Earlier features carry less weight and later layers carry more weight.
        # Reference: https://arxiv.org/pdf/1711.11585.pdf
        self.encoder_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.feature_model = self._build_vgg_19()
        self.mean_absolute_error = losses.MeanAbsoluteError()

    def _build_vgg_19(self):
        vgg = applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        return Model(vgg.input, layer_outputs, name="VGG")

    def call(self, real_image, fake_image):
        x = tf.reverse(real_image, axis=[-1])
        y = tf.reverse(fake_image, axis=[-1])
        x = applications.vgg19.preprocess_input(127.5 * (x + 1))
        y = applications.vgg19.preprocess_input(127.5 * (y + 1))
        feat_real = self.feature_model(x)
        feat_fake = self.feature_model(y)
        loss = 0
        for i in range(len(feat_real)):
            loss += self.encoder_weights[i] * self.mean_absolute_error(
                feat_real[i], feat_fake[i]
            )
        return loss
