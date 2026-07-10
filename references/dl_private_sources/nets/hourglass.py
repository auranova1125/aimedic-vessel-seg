"""Hourglass-style network for stacked-volume segmentation."""

import tensorflow as tf

import config

layers = tf.keras.layers
Model = tf.keras.Model


def conv2d(output_channels, kernel_size, stride, use_bias=False):
    """Create a regularized same-padded convolution layer."""
    return layers.Conv2D(
        output_channels,
        kernel_size,
        stride,
        "same",
        use_bias=use_bias,
        kernel_regularizer=tf.keras.regularizers.l2(config.WEIGHT_DECAY),
    )


def conv2d_transpose(output_channels, kernel_size, stride):
    """Create a regularized same-padded transpose convolution layer."""
    return layers.Conv2DTranspose(
        output_channels,
        kernel_size,
        stride,
        "same",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(config.WEIGHT_DECAY),
    )


def mish(input_tensor):
    """Apply the Mish activation function."""
    return input_tensor * tf.nn.tanh(tf.nn.softplus(input_tensor))

class MultiScaleResBlock(Model):
    """Residual block with three progressively narrower convolution branches."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.c1 = conv2d(int(output_channels / 2), 3, 1)
        self.bn1 = layers.BatchNormalization()
        self.c2 = conv2d(int(output_channels / 4), 3, 1)
        self.bn2 = layers.BatchNormalization()
        self.c3 = conv2d(int(output_channels / 4), 3, 1)
        self.bn3 = layers.BatchNormalization()
        self.c4 = (
            conv2d(output_channels, 1, 1)
            if input_channels != output_channels
            else None
        )
        self.bn4 = (
            layers.BatchNormalization()
            if input_channels != output_channels
            else None
        )

    def call(self, x, **kwargs):
        c1 = mish(self.bn1(self.c1(x)))
        c2 = mish(self.bn2(self.c2(c1)))
        c3 = self.bn3(self.c3(c2))
        c4 = tf.concat([c1, c2, c3], axis=3)

        skip = self.bn4(self.c4(x)) if self.c4 is not None else x
        c4 = mish(c4 + skip)

        return c4


class BlockLayer(Model):
    """Apply three multi-scale residual blocks in sequence."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block1 = MultiScaleResBlock(input_channels, output_channels[0])
        self.block2 = MultiScaleResBlock(output_channels[0], output_channels[1])
        self.block3 = MultiScaleResBlock(output_channels[1], output_channels[2])

    def call(self, x, **kwargs):
        return self.block3(self.block2(self.block1(x)))


class Down(Model):
    """Reduce spatial resolution by a factor of two."""

    def __init__(self, output_channels):
        super().__init__()
        self.conv = conv2d(output_channels, 3, 2)
        self.bn = layers.BatchNormalization()

    def call(self, x, **kwargs):
        return mish(self.bn(self.conv(x)))


class Up(Model):
    """Increase spatial resolution by a factor of two."""

    def __init__(self, output_channels):
        super().__init__()
        self.conv = conv2d_transpose(output_channels, 3, 2)
        self.bn = layers.BatchNormalization()

    def call(self, x, **kwargs):
        return mish(self.bn(self.conv(x)))


class HourglassSegmentationNet(Model):
    """Encode a slice window and predict one mask for every input slice."""

    def __init__(self):
        super().__init__()
        self.fc1 = conv2d(32, 3, 1)
        self.fbn1 = layers.BatchNormalization()
        self.fc2 = conv2d(32, 3, 1)
        self.fbn2 = layers.BatchNormalization()
        self.fc3 = conv2d(32, 3, 1)
        self.fbn3 = layers.BatchNormalization()
        self.fc4 = BlockLayer(32, [64, 64, 64])

        self.down1 = Down(64)
        self.enc1 = BlockLayer(64, [128, 128, 128])
        self.down2 = Down(128)
        self.enc2 = BlockLayer(128, [128, 128, 128])
        self.down3 = Down(128)
        self.enc3 = BlockLayer(128, [128, 128, 128])
        self.down4 = Down(128)

        self.mid = BlockLayer(128, [128, 128, 128])

        self.up4 = Up(128)
        self.dec3 = MultiScaleResBlock(128, 128)
        self.up3 = Up(128)
        self.dec2 = MultiScaleResBlock(128, 128)
        self.up2 = Up(128)
        self.dec1 = MultiScaleResBlock(128, 128)
        self.up1 = Up(64)

        self.con3 = BlockLayer(128, [128, 128, 128])
        self.con2 = BlockLayer(128, [128, 128, 128])
        self.con1 = BlockLayer(128, [128, 128, 128])
        self.con0 = BlockLayer(64, [64, 64, 64])

        self.drop1 = layers.Dropout(0.25)
        self.pc1 = conv2d(64, 3, 1)
        self.pbn1 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.25)
        self.pc2 = conv2d(64, 3, 1)
        self.pbn2 = layers.BatchNormalization()
        self.pc3 = conv2d(config.WINDOW_SIZE, 1, 1, True)

    def call(self, x, **kwargs):
        c1 = mish(self.fbn1(self.fc1(x)))
        c2 = mish(self.fbn2(self.fc2(c1)))
        c3 = mish(self.fbn3(self.fc3(c2)))
        c4 = self.fc4(c3)

        dc1 = self.enc1(self.down1(c4))
        dc2 = self.enc2(self.down2(dc1))
        dc3 = self.enc3(self.down3(dc2))

        mid = self.mid(self.down4(dc3))

        uc3 = self.dec3(self.up4(mid) + self.con3(dc3))
        uc2 = self.dec2(self.up3(uc3) + self.con2(dc2))
        uc1 = self.dec1(self.up2(uc2) + self.con1(dc1))
        uc0 = self.up1(uc1) + self.con0(c4)

        final = mish(self.pbn1(self.pc1(self.drop1(uc0))))
        final = mish(self.pbn2(self.pc2(self.drop2(final))))
        final = tf.sigmoid(self.pc3(final))

        return final
