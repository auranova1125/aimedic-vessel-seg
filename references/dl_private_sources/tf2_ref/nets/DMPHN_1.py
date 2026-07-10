"""Single-scale RGB DMPHN model used by the GOPRO training and test scripts."""

import math

import tensorflow as tf

Model = tf.keras.Model
layers = tf.keras.layers


def make_kernel_initializer(output_channels, kernel_size):
    """Create the variance-scaled normal initializer used by the DMPHN blocks."""
    standard_deviation = math.sqrt(
        2 / (kernel_size * kernel_size * output_channels)
    ) / 2
    return tf.random_normal_initializer(0, standard_deviation)


class ResidualBlock(Model):
    """Two-convolution residual branch without an internal skip addition."""

    def __init__(self, output_channels):
        super().__init__()
        self.act = layers.ReLU()
        self.c1 = layers.Conv2D(
            output_channels,
            3,
            1,
            "same",
            kernel_initializer=make_kernel_initializer(output_channels, 3),
        )
        self.c2 = layers.Conv2D(
            output_channels,
            3,
            1,
            "same",
            kernel_initializer=make_kernel_initializer(output_channels, 3),
        )

    def call(self, input_tensor, **kwargs):
        return self.c2(self.act(self.c1(input_tensor)))


class Encoder(Model):
    """Three-stage convolutional encoder with residual feature refinement."""

    def __init__(self):
        super().__init__()
        self.c1 = layers.Conv2D(
            32,
            3,
            1,
            padding="same",
            kernel_initializer=make_kernel_initializer(32, 3),
        )
        self.c2 = ResidualBlock(32)
        self.c3 = ResidualBlock(32)
        self.c4 = layers.Conv2D(
            64,
            3,
            2,
            padding="same",
            kernel_initializer=make_kernel_initializer(64, 3),
        )
        self.c5 = ResidualBlock(64)
        self.c6 = ResidualBlock(64)
        self.c7 = layers.Conv2D(
            128,
            3,
            2,
            padding="same",
            kernel_initializer=make_kernel_initializer(128, 3),
        )
        self.c8 = ResidualBlock(128)
        self.c9 = ResidualBlock(128)

    def call(self, input_tensor, **kwargs):
        features = self.c1(input_tensor)
        features = self.c2(features) + features
        features = self.c3(features) + features
        features = self.c4(features)
        features = self.c5(features) + features
        features = self.c6(features) + features
        features = self.c7(features)
        features = self.c8(features) + features
        return self.c9(features) + features


class Decoder(Model):
    """Three-stage decoder that produces an RGB sharp image."""

    def __init__(self):
        super().__init__()
        self.c10 = ResidualBlock(128)
        self.c11 = ResidualBlock(128)
        self.c12 = layers.Conv2DTranspose(
            64,
            4,
            2,
            padding="same",
            kernel_initializer=make_kernel_initializer(128, 3),
        )
        self.c13 = ResidualBlock(64)
        self.c14 = ResidualBlock(64)
        self.c15 = layers.Conv2DTranspose(
            32,
            4,
            2,
            padding="same",
            kernel_initializer=make_kernel_initializer(64, 3),
        )
        self.c16 = ResidualBlock(32)
        self.c17 = ResidualBlock(32)
        self.c18 = layers.Conv2D(
            3,
            3,
            padding="same",
            kernel_initializer=make_kernel_initializer(32, 3),
        )

    def call(self, input_tensor, **kwargs):
        features = self.c10(input_tensor) + input_tensor
        features = self.c11(features) + features
        features = self.c12(features)
        features = self.c13(features) + features
        features = self.c14(features) + features
        features = self.c15(features)
        features = self.c16(features) + features
        features = self.c17(features) + features
        return self.c18(features)


class DMPHN1(Model):
    """Single-scale encoder-decoder DMPHN for RGB image deblurring."""

    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()

    def call(self, input_tensor, **kwargs):
        return self.dec(self.enc(input_tensor))
