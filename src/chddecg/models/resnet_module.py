import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.regularizers import l2

class InputConv(layers.Layer):
    def __init__(self, filter_num=24, kernel_size=30, stride=2, name=None, **kwargs):
        super(InputConv, self).__init__(name=name, **kwargs)
        self.conv = layers.Conv1D(
            filters=filter_num, kernel_size=kernel_size, strides=stride,
            padding='same', activation='relu', kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )
        self.bn = layers.BatchNormalization()
        self.pool = layers.MaxPooling1D(pool_size=2)
        self.dropout = layers.Dropout(0.3)

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.pool(x)
        x = self.dropout(x, training=training)
        return x

class ResBlock(layers.Layer):
    def __init__(self, filter_num=12, kernel_size=3, stride=2, name=None, **kwargs):
        super(ResBlock, self).__init__(name=name, **kwargs)
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = layers.Conv1D(
            filters=filter_num, kernel_size=kernel_size, strides=stride,
            padding='same', activation='relu', kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)

        self.conv2 = layers.Conv1D(
            filters=filter_num, kernel_size=kernel_size, strides=1,
            padding='same', activation=None, kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )
        self.bn2 = layers.BatchNormalization()

        self.shortcut_conv = layers.Conv1D(
            filters=filter_num, kernel_size=1, strides=stride,
            padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        ) if stride != 1 else None
        self.shortcut_bn = layers.BatchNormalization() if stride != 1 else None

    def call(self, inputs, training=False):
        x = inputs
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.stride != 1 or inputs.shape[-1] != self.filter_num:
            shortcut = self.shortcut_conv(shortcut)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

class SE(layers.Layer):
    def __init__(self, filter_sq, input_channel, name=None, **kwargs):
        super(SE, self).__init__(name=name, **kwargs)
        self.squeeze = layers.GlobalAveragePooling1D()
        self.excite1 = layers.Dense(
            units=filter_sq, activation='relu', kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )
        self.excite2 = layers.Dense(
            units=input_channel, activation='sigmoid', kernel_initializer='he_normal',
            kernel_regularizer=l2(0.05)
        )
        self.reshape = layers.Reshape((1, input_channel))

    def call(self, inputs, training=False):
        x = self.squeeze(inputs)
        x = self.excite1(x)
        x = self.excite2(x)
        x = self.reshape(x)
        return layers.Multiply()([inputs, x])
