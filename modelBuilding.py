import tensorflow as tf
from tensorflow import keras

def build_model():
    '''
    Build a simple CNN with two Convolution + MaxPooling layers and two Dense layers

    The model take 32x32 single-channel images (32, 32, 1)
    :return: the model
    '''
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(43, activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model