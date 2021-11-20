from tensorflow import keras
from tensorflow.keras import losses, applications


class ContentLoss(losses.Loss):
    def __init__(self, reduction=..., name=None):
        super().__init__(reduction=reduction, name=name)
        self.feature_extraction_model = self._build_vgg19_model()
        # The encoder layers weights are basically an increasing geometric progression (1 / 2 ^ n).
        # Earlier features carry less weight and later layers carry more weight.
        # Reference: https://arxiv.org/pdf/1711.11585.pdf
        self.encoder_layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.mean_absolute_error = losses.MeanAbsoluteError(reduction="none")

    def _build_vgg19_model(self):
        vgg19_model = applications.VGG19(include_top=False, weights="imagenet")
        return keras.Model(
            vgg19_model.input,
            [vgg19_model.get_layer(x).output for x in self.encoder_layers],
            name="vgg19_model",
        )

    def call(self, y_true, y_pred):
        y_true = 127.5 * (y_true + 1)
        y_pred = 127.5 * (y_pred + 1)
        y_true = applications.vgg19.preprocess_input(y_true)
        y_pred = applications.vgg19.preprocess_input(y_pred)
        y_true_features = self.feature_extraction_model(y_true)
        y_pred_features = self.feature_extraction_model(y_pred)
        # loss = 0.0
        # for i in range(len(self.encoder_layers)):
        #     loss += self.encoder_layer_weights[i] * self.mean_absolute_error(
        #         y_true_features[i], y_pred_features[i]
        #     )
        loss = self.encoder_layer_weights * self.mean_absolute_error(
            y_true_features, y_pred_features
        )
        return loss
