import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
import numpy as np
import math

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name=None, **kwargs):
        super(MultiHeadAttention, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

    def build(self, input_shape):
        # input_shape is a list of shapes: [q_shape, k_shape, v_shape]
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 3:
            raise ValueError(f"Expected input_shape to be a list of 3 shapes, got {input_shape}")

        q_shape, k_shape, v_shape = input_shape  # Unpack the shapes

        self.Wq = tf.keras.layers.Dense(self.d_model, kernel_initializer='glorot_normal')
        self.Wk = tf.keras.layers.Dense(self.d_model, kernel_initializer='glorot_normal')
        self.Wv = tf.keras.layers.Dense(self.d_model, kernel_initializer='glorot_normal')
        self.dense = tf.keras.layers.Dense(self.d_model, kernel_initializer='glorot_normal')

        # Build each dense layer with the appropriate shape
        self.Wq.build(q_shape)
        self.Wk.build(k_shape)
        self.Wv.build(v_shape)
        # The output shape of the attention mechanism after concatenation
        self.dense.build((None, None, self.d_model))

        super(MultiHeadAttention, self).build(input_shape)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, mask=None):
        q, k, v = inputs

        # Cast all inputs to the same dtype to avoid type mismatch
        input_dtype = q.dtype

        batch_size = tf.shape(q)[0]

        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Ensure consistent dtype for the attention calculation
        q = tf.cast(q, input_dtype)
        k = tf.cast(k, input_dtype)
        v = tf.cast(v, input_dtype)

        scaled_attention = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        # Cast output back to input dtype
        output = tf.cast(output, input_dtype)

        return output

    def scaled_dot_product_attention(self, q, k, v, mask):
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # Cast dk to the same dtype as q
        dk = tf.cast(tf.shape(k)[-1], q.dtype)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # Fixed mask handling - check if mask is not None before using it
        if mask is not None and not isinstance(mask, list):
            # Ensure mask has the right dtype
            mask_value = tf.cast(-1e9, q.dtype)
            scaled_attention_logits += (mask * mask_value)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def compute_output_shape(self, input_shape):
        # Since input_shape is a list, return the shape of the query (first element)
        return (input_shape[0][0], input_shape[0][1], self.d_model)

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads
        })
        return config

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name=None, **kwargs):
        super(EncoderLayer, self).__init__(name=name, **kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.mha.build([input_shape, input_shape, input_shape])

        # Make sure FFN respects the input dtype
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.dff, activation='relu', kernel_initializer='he_normal'),
            tf.keras.layers.Dense(self.d_model, kernel_initializer='he_normal')
        ])
        self.ffn.build((None, None, self.d_model))

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm1.build(input_shape)
        self.layernorm2.build(input_shape)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        super(EncoderLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        input_dtype = inputs.dtype

        # Ensure attention operations maintain dtype consistency
        attn_output = self.mha([inputs, inputs, inputs])
        attn_output = tf.cast(attn_output, input_dtype)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        out1 = tf.cast(out1, input_dtype)

        ffn_output = self.ffn(out1)
        ffn_output = tf.cast(ffn_output, input_dtype)

        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        out2 = tf.cast(out2, input_dtype)

        return out2

    def compute_output_shape(self, input_shape):
        # The output shape retains the batch and sequence length, but the feature dimension becomes d_model
        return (input_shape[0], input_shape[1], self.d_model)

    def get_config(self):
        config = super(EncoderLayer, self).get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate
        })
        return config

class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, feature_dim, kernel_size, stride, dropout_rate=0.1, use_residual=True, name=None, **kwargs):
        super(TemporalAttention, self).__init__(name=name, **kwargs)
        self.feature_dim = feature_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv1D(
            filters=self.feature_dim,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same',
            kernel_initializer='he_normal'
        )
        self.conv.build(input_shape)

        # Use LayerNormalization instead of GroupNormalization for better dtype handling
        self.bn = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.bn.build(self.conv.compute_output_shape(input_shape))

        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        super(TemporalAttention, self).build(input_shape)

    def call(self, inputs, training=False):
        input_dtype = inputs.dtype

        attention = self.conv(inputs)
        attention = self.bn(attention)
        attention = self.sigmoid(attention)
        attention = self.dropout(attention, training=training)

        # Cast to maintain dtype consistency
        attention = tf.cast(attention, input_dtype)

        output = inputs * attention
        if self.use_residual:
            output = output + inputs

        return output

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'dropout_rate': self.dropout_rate,
            'use_residual': self.use_residual
        })
        return config
