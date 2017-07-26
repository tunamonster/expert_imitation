import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.models import Sequential

def init_model(in_shape, out_shape):
    print('creating model...')
    model = Sequential()
    model.add(Dense(256, input_dim=in_shape))
    model.add(Activation('tanh'))

    model.add(Dense(256))
    model.add(Activation('tanh'))

    model.add(Dense(out_shape))

    rms = RMSprop()
    model.compile(loss='mean_squared_error', optimizer=rms, metrics=['mae'])
    print('model created')
    return model