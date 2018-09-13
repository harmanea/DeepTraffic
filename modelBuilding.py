from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def build_simple_model():
    '''
    Build a simple model with three Dense layers

    The model takes 32x32 single-channel images (32, 32, 1)
    :return: the model
    '''
    model = keras.Sequential()

    model.add(Flatten(input_shape=(32, 32, 1)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(43, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def build_model():
    '''
    Build a CNN with two Convolution + MaxPooling layers and two Dense layers

    The model takes 32x32 single-channel images (32, 32, 1)
    :return: the model
    '''
    model = keras.Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(43, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model