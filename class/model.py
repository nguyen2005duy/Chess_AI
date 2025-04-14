# File 1: model.py
import tensorflow as tf
from keras import layers, models
from keras._tf_keras.keras import losses

def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def custom_cce(y_true, y_pred):
    return losses.categorical_crossentropy(y_true, y_pred)

def create_chess_model():
    """Create a CNN model with policy and value heads"""
    input_shape = (8, 8, 18)  # 12 pieces + 6 channels for game state
    policy_shape = 4672  # Maximum number of possible moves (8x8x73)

    inputs = layers.Input(shape=input_shape)

    # Common trunk
    x = layers.Conv2D(256, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(5):
        residual = x
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)

    # Policy head
    policy = layers.Conv2D(73, 3, padding='same')(x)
    policy = layers.Flatten()(policy)
    policy = layers.Softmax(name='policy')(policy)

    # Value head
    value = layers.Conv2D(1, 1, activation='relu')(x)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation='relu')(value)
    value = layers.Dense(1, activation='tanh', name='value')(value)

    return models.Model(inputs=inputs, outputs=[policy, value])