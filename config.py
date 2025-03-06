import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Add, Activation

class ECAAttention(Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super(ECAAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = Conv1D(1, kernel_size=self.kernel_size, padding='same', use_bias=False)
        super(ECAAttention, self).build(input_shape)

    def call(self, inputs):
        attention = GlobalAveragePooling1D()(inputs)
        attention = tf.expand_dims(attention, -1)
        attention = self.conv(attention)
        attention = tf.nn.sigmoid(attention)
        return inputs * attention[:, None, :]

class AdaptiveLateDropout(Layer):
    def __init__(self, rate=0.5, start_step=10, **kwargs):
        super(AdaptiveLateDropout, self).__init__(**kwargs)
        Self.rate = rate
        Self.start_step = start_step
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)

    def call(self, Inputs, training=None):
        if training:
            apply_dropout = tf.cast(self.step >= self.start_step, tf.bool)
            outputs = tf.cond(
                apply_dropout,
                lambda: Dropout(self.rate)(inputs, training=training),
                lambda: inputs
            )
            self.step.assign_add(1)
            return outputs
        return inputs

def Conv1DResidualBlock(filters, kernel_size, dilation_rate=1, dropout_rate=0.2, activation='relu'):
    def apply(inputs):
        # Shortcut connection
        shortcut = inputs

        # Convolutional Layer
        x = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

        # ECA Attention
        x = ECAAttention(kernel_size=5)(x)

        x = Conv1D(filters, 1, use_bias=False)(x)
        x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        x = Add()([shortcut, x])
        return x
    return apply

def TransformerWithAttentionBlock(dim, num_heads, dropout_rate=0.1, activation='relu'):
    def apply(inputs):
        # Multi-Head Self-Attention
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)(inputs, inputs)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = Add()([inputs, attn_output])  # Residual connection

        x = Dense(dim * 4, activation=activation)(out1)
        x = Dense(dim)(x)
        x = AdaptiveLateDropout(rate=dropout_rate)(x)
        return Add()([out1, x])  # Residual connection
    return apply


def build_model(input_shape, num_classes, dim=128):
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv1D(dim, kernel_size=3, padding='causal', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual Convolutional Blocks
    x = Conv1DResidualBlock(dim, kernel_size=3, dropout_rate=0.2)(x)
    x = Conv1DResidualBlock(dim, kernel_size=3, dropout_rate=0.2)(x)

    # Transformer Blocks
    x = TransformerWithAttentionBlock(dim, num_heads=4, dropout_rate=0.2)(x)
    x = TransformerWithAttentionBlock(dim, num_heads=4, dropout_rate=0.2)(x)

    # Global Pooling and Classification
    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return tf.keras.Model(inputs, outputs)

model = build_model(input_shape=(100, 64), num_classes=10)
model.summary()
