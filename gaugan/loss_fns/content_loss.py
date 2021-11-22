import tensorflow as tf
from tensorflow.keras import losses, applications, Model


class ContentLoss(losses.Loss):
    """VGG19 based feature matching loss"""

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
        real_features = applications.vgg19.preprocess_input(127.5 * (real_image + 1))
        fake_features = applications.vgg19.preprocess_input(127.5 * (fake_image + 1))
        real_features = self.feature_model(real_features)
        fake_features = self.feature_model(fake_features)
        loss = 0
        for i in range(len(real_features)):
            loss += self.encoder_weights[i] * self.mean_absolute_error(
                real_features[i], fake_features[i]
            )
        return loss
