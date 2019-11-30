import tensorflow as tf
#import logging
from absl import logging
from tensorflow import keras
from tensorflow.keras.initializers import glorot_uniform as xavier
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization

#logging = logging.getlogging()

class ConvModel(keras.Model):
    def __init__(self, nclasses=6):
        super(ConvModel, self).__init__(name='conv_model')
        self.nclasses = nclasses
        self.format = "channels_first"
        #combined
        self.dropout = Dropout(0.5)
        self.dense = Dense(nclasses)
        logging.info('initialized ConvModel')
    
    def _convlayer(self, inputs, filters, kernel_size):
        return Conv2D(filters, kernel_size=kernel_size, activation='relu',
            data_format=self.format)(inputs)

    def _maxpool(self, inputs, pool_size):
        return MaxPooling2D(pool_size=pool_size, strides=1,
            data_format=self.format)(inputs)

    def _inner(self, inputs, units):
        inputs = Flatten(data_format=self.format)(inputs)
        inputs = Dense(units=units, activation='relu',
            kernel_initializer = xavier,
            bias_initializer = Constant)(inputs)
        return inputs

    def _dropout(self, inputs, rate):
       return  Dropout(rate=rate)(inputs)

    def call(self, x_inputs, u_inputs, v_inputs):
        # x
        x = self._convlayer(x_inputs, 12, (8,5))
        x = self._maxpool(x, (2,2))
        x = self._convlayer(x, 20, (7,3))
        x = self._maxpool(x, (2,2))
        x = self._convlayer(x, 28, (6,3))
        x = self._maxpool(x, (2,1))
        x = self._convlayer(x, 36, (3,3))
        x = self._maxpool(x, (1,1))
        x = self._inner(x, 258)
        x = self._dropout(x, 0.5)
        # u
        u = self._convlayer(u_inputs, 12, (8,5))
        u = self._maxpool(u, (2,2))
        u = self._convlayer(u, 20, (7,3))
        u = self._maxpool(u, (2,2))
        u = self._convlayer(u, 28, (6,3))
        u = self._maxpool(u, (2,1))
        u = self._convlayer(u, 36, (3,3))
        u = self._maxpool(u, (1,1))
        u = self._inner(u, 256)
        u = self._dropout(u, 0.5)
        # v
        v = self._convlayer(v_inputs, 12, (8,5))
        v = self._maxpool(v, (2,2))
        v = self._convlayer(v, 20, (7,3))
        v = self._maxpool(v, (2,2))
        v = self._convlayer(v, 28, (6,3))
        v = self._maxpool(v, (2,1))
        v = self._convlayer(v, 36, (3,3))
        v = self._maxpool(v, (1,1))
        v = self._inner(v, 256)
        v = self._dropout(v, 0.5)
        # combined
        m = concatenate([x, u, v], axis=1)
        m = self._inner(m, 128)
        m = self._dropout(m, 0.5)
        m = self.dense(m)
        return self.dropout(m)

    def compute_output_shape(self, input_shape):
        # we must override this function if we want to use the subclass
        # model as part of a functional-style model - otherwise it is
        # optional
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.nclasses
        return tf.TensorShape(shape)
