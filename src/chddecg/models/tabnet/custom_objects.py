import tensorflow as tf
from tensorflow.keras import layers

def glu(inputs, dim=None):
    if dim is None:
        dim = inputs.shape[-1]
        if dim is None:
            raise ValueError("Last dimension must be defined for GLU, but got None")
        if dim % 2 != 0:
            raise ValueError(f"Last dimension ({dim}) must be divisible by 2 for GLU")
        dim = dim // 2
    a = inputs[..., :dim]
    b = inputs[..., dim:]
    output = a * tf.nn.sigmoid(b)
    return tf.cast(output, inputs.dtype)

def sparsemax(logits, axis=-1):
    # Get the input dtype
    input_dtype = logits.dtype

    # Graph-mode compatible implementation
    logits = logits - tf.reduce_max(logits, axis=axis, keepdims=True)
    z_sorted = tf.sort(logits, direction='DESCENDING', axis=axis)
    dim = tf.shape(logits)[axis]
    dim_float = tf.cast(dim, logits.dtype)
    indices = tf.cast(tf.range(dim), logits.dtype) + 1.0
    cumsum = tf.cumsum(z_sorted, axis=axis)
    condition = z_sorted > (cumsum - 1.0) / indices
    k = tf.reduce_sum(tf.cast(condition, tf.int32), axis=axis, keepdims=True)
    kth_values = tf.gather(cumsum, tf.maximum(k - 1, 0), batch_dims=1, axis=axis)
    tau = (kth_values - 1) / tf.cast(k, logits.dtype)
    tau = tf.expand_dims(tau, axis=axis)

    # Use the same dtype for zero as the input
    zero = tf.cast(0.0, input_dtype)
    output = tf.maximum(zero, logits - tau)

    return tf.cast(output, input_dtype)

class GroupNormalization(layers.Layer):
    def __init__(self, groups=2, epsilon=1e-5, dtype=tf.float16, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon
        self._layer_dtype = dtype
        self.axis = axis
        print(f"Initialized GroupNormalization with _layer_dtype={self._layer_dtype}")

    def build(self, input_shape):
        print(f"Building GroupNormalization with input_shape={input_shape}")
        channels = input_shape[self.axis]
        self.gamma = self.add_weight(
            shape=(channels,),
            initializer='ones',
            name='gamma',
            dtype=self._layer_dtype
        )
        self.beta = self.add_weight(
            shape=(channels,),
            initializer='zeros',
            name='beta',
            dtype=self._layer_dtype
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Save original dtype
        input_dtype = inputs.dtype

        # Ensure inputs are in float32 for better numerical stability during computation
        inputs = tf.cast(inputs, tf.float32)

        # Use tf.nn.moments for calculating mean and variance
        mean, var = tf.nn.moments(inputs, axes=[1], keepdims=True)

        # Normalization
        normalized = (inputs - mean) / tf.sqrt(var + self.epsilon)

        # Cast gamma and beta to float32 for the operations
        gamma = tf.cast(self.gamma, tf.float32)
        beta = tf.cast(self.beta, tf.float32)

        # Apply gamma and beta
        output = gamma * normalized + beta

        # Cast back to the original input dtype
        output = tf.cast(output, input_dtype)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'groups': self.groups,
            'epsilon': self.epsilon,
            '_layer_dtype': self._layer_dtype,
            'axis': self.axis
        })
        return config

class AssertFiniteLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(AssertFiniteLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        # In graph mode, we'll use tf.debugging.assert_all_finite which works properly
        # This will only run during training, not during inference
        if tf.executing_eagerly():
            tf.debugging.check_numerics(inputs, message=f"NaN or Inf detected in {self.name}")
        else:
            # This version works in graph mode
            inputs = tf.debugging.assert_all_finite(
                inputs, 
                message=f"NaN or Inf detected in {self.name}"
            )
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super(AssertFiniteLayer, self).get_config()

custom_objects = {
    'glu': glu,
    'sparsemax': sparsemax,
    'GroupNormalization': GroupNormalization,
    'AssertFiniteLayer': AssertFiniteLayer
}
