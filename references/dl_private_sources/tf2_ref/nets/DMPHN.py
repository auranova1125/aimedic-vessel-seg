"""Hierarchical single-channel DMPHN variants for large image experiments."""

import math

import tensorflow as tf

Model = tf.keras.Model
layers = tf.keras.layers

IMAGE_HEIGHT = 3400
IMAGE_WIDTH = 3400


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
            1,
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
    """Three-stage decoder that produces a single-channel output image."""

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
            1,
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


class DMPHN_1(Model):
    """Single-scale DMPHN variant for single-channel inputs."""

    def __init__(self):
        super().__init__()
        self.enc1 = Encoder()
        self.dec1 = Decoder()

    def call(self, input_tensor, **kwargs):
        return self.dec1(self.enc1(input_tensor))


class DMPHN_12(Model):
    """Two-level DMPHN variant with horizontal half-image processing."""

    def __init__(self):
        super().__init__()
        self.enc1 = Encoder()
        self.dec1 = Decoder()
        self.enc2 = Encoder()
        self.dec2 = Decoder()

    def call(self, input_tensor, **kwargs):
        left_half = input_tensor[:, :, : IMAGE_WIDTH // 2, :]
        right_half = input_tensor[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]

        level_two_features = tf.concat(
            [self.enc2(left_half), self.enc2(right_half)],
            axis=2,
        )
        level_two_output = self.dec2(level_two_features)
        level_one_features = (
            self.enc1(input_tensor + level_two_output) + level_two_features
        )
        return self.dec1(level_one_features)


class DMPHN_124(Model):
    """Three-level DMPHN variant with four quarter-image branches."""

    def __init__(self):
        super().__init__()
        self.enc1 = Encoder()
        self.dec1 = Decoder()
        self.enc2 = Encoder()
        self.dec2 = Decoder()
        self.enc3 = Encoder()
        self.dec3 = Decoder()

    def call(self, input_tensor, **kwargs):
        left_half = input_tensor[:, :, : IMAGE_WIDTH // 2, :]
        right_half = input_tensor[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]
        left_top = left_half[:, : IMAGE_HEIGHT // 2, :, :]
        left_bottom = left_half[:, IMAGE_HEIGHT // 2 : IMAGE_HEIGHT, :, :]
        right_top = right_half[:, : IMAGE_HEIGHT // 2, :, :]
        right_bottom = right_half[:, IMAGE_HEIGHT // 2 : IMAGE_HEIGHT, :, :]

        left_level_three_features = tf.concat(
            [self.enc3(left_top), self.enc3(left_bottom)],
            axis=1,
        )
        right_level_three_features = tf.concat(
            [self.enc3(right_top), self.enc3(right_bottom)],
            axis=1,
        )
        left_level_three_output = self.dec3(left_level_three_features)
        right_level_three_output = self.dec3(right_level_three_features)

        level_two_features = tf.concat(
            [
                self.enc2(left_half + left_level_three_output)
                + left_level_three_features,
                self.enc2(right_half + right_level_three_output)
                + right_level_three_features,
            ],
            axis=2,
        )
        level_two_output = self.dec2(level_two_features)
        level_one_features = (
            self.enc1(input_tensor + level_two_output) + level_two_features
        )
        return self.dec1(level_one_features)


class DMPHN_1248(Model):
    """Four-level DMPHN variant with eight eighth-image branches."""

    def __init__(self):
        super().__init__()
        self.enc1 = Encoder()
        self.dec1 = Decoder()
        self.enc2 = Encoder()
        self.dec2 = Decoder()
        self.enc3 = Encoder()
        self.dec3 = Decoder()
        self.enc4 = Encoder()
        self.dec4 = Decoder()

    def call(self, input_tensor, **kwargs):
        left_half = input_tensor[:, :, : IMAGE_WIDTH // 2, :]
        right_half = input_tensor[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]
        left_top = left_half[:, : IMAGE_HEIGHT // 2, :, :]
        left_bottom = left_half[:, IMAGE_HEIGHT // 2 : IMAGE_HEIGHT, :, :]
        right_top = right_half[:, : IMAGE_HEIGHT // 2, :, :]
        right_bottom = right_half[:, IMAGE_HEIGHT // 2 : IMAGE_HEIGHT, :, :]
        left_top_left = left_top[:, :, : IMAGE_WIDTH // 2, :]
        left_top_right = left_top[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]
        left_bottom_left = left_bottom[:, :, : IMAGE_WIDTH // 2, :]
        left_bottom_right = left_bottom[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]
        right_top_left = right_top[:, :, : IMAGE_WIDTH // 2, :]
        right_top_right = right_top[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]
        right_bottom_left = right_bottom[:, :, : IMAGE_WIDTH // 2, :]
        right_bottom_right = right_bottom[:, :, IMAGE_WIDTH // 2 : IMAGE_WIDTH, :]

        left_top_features = tf.concat(
            [self.enc4(left_top_left), self.enc4(left_top_right)],
            axis=1,
        )
        left_bottom_features = tf.concat(
            [self.enc4(left_bottom_left), self.enc4(left_bottom_right)],
            axis=1,
        )
        right_top_features = tf.concat(
            [self.enc4(right_top_left), self.enc4(right_top_right)],
            axis=1,
        )
        right_bottom_features = tf.concat(
            [self.enc4(right_bottom_left), self.enc4(right_bottom_right)],
            axis=1,
        )
        left_top_output = self.dec4(left_top_features)
        left_bottom_output = self.dec4(left_bottom_features)
        right_top_output = self.dec4(right_top_features)
        right_bottom_output = self.dec4(right_bottom_features)

        left_level_three_features = tf.concat(
            [
                self.enc3(left_top + left_top_output) + left_top_features,
                self.enc3(left_bottom + left_bottom_output) + left_bottom_features,
            ],
            axis=1,
        )
        right_level_three_features = tf.concat(
            [
                self.enc3(right_top + right_top_output) + right_top_features,
                self.enc3(right_bottom + right_bottom_output) + right_bottom_features,
            ],
            axis=1,
        )
        left_level_three_output = self.dec2(left_level_three_features)
        right_level_three_output = self.dec2(right_level_three_features)

        level_two_features = tf.concat(
            [
                self.enc2(left_half + left_level_three_output)
                + left_level_three_features,
                self.enc2(right_half + right_level_three_output)
                + right_level_three_features,
            ],
            axis=2,
        )
        level_two_output = self.dec2(level_two_features)
        level_one_features = (
            self.enc1(input_tensor + level_two_output) + level_two_features
        )
        return self.dec1(level_one_features)
