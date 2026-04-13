import tensorflow as tf
from tensorflow.keras import layers
from .tabnet.custom_objects import glu, sparsemax, custom_objects

class TabNet_downsampling(tf.keras.layers.Layer):
    def __init__(self, num_features, feature_dim, output_dim, num_decision_steps=2,
                 relaxation_factor=1.5, sparsity_coefficient=1e-3, norm_type='group',
                 name=None, **kwargs):
        super(TabNet_downsampling, self).__init__(name=name, **kwargs)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient

    def build(self, input_shape):
        # Use simple BatchNormalization instead of GroupNormalization
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-6)
        self.bn.build(input_shape)

        # Initial feature transformer
        self.fc = tf.keras.layers.Dense(self.feature_dim, kernel_initializer='he_normal')
        self.fc.build(input_shape)

        # Final output projection
        self.out_fc = tf.keras.layers.Dense(self.output_dim, kernel_initializer='he_normal')
        # Build with the correct shape from feature_dim
        features_shape = (None, self.feature_dim // 2)  # Accounting for GLU output size
        self.out_fc.build(features_shape)

        super(TabNet_downsampling, self).build(input_shape)

    def call(self, inputs, training=False):
        # Save input dtype for consistent casting
        input_dtype = inputs.dtype

        # Simple forward pass
        x = self.bn(inputs, training=training)
        x = self.fc(x)
        x = glu(x, dim=self.feature_dim // 2)
        x = tf.cast(x, input_dtype)

        # Final output projection
        output = self.out_fc(x)
        output = glu(output, dim=self.output_dim // 2)

        # Ensure output has the right dtype
        output = tf.cast(output, input_dtype)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim // 2)

    def get_config(self):
        config = super(TabNet_downsampling, self).get_config()
        config.update({
            'num_features': self.num_features,
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'num_decision_steps': self.num_decision_steps,
            'relaxation_factor': self.relaxation_factor,
            'sparsity_coefficient': self.sparsity_coefficient
        })
        return config
