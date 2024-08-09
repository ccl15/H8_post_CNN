import tensorflow as tf
from tensorflow.keras import layers
'''
Convolution two times
'''

class Convolution(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, (3,3), padding='same', activation=layers.LeakyReLU(), kernel_initializer='he_normal')
        self.conv2 = layers.Conv2D(filters, (3,3), padding='same', activation=layers.LeakyReLU(), kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
    def __call__(self, x, training):
        x = self.bn1( self.conv1(x), training=training)
        x = self.bn2( self.conv2(x), training=training)
        return x

class Model(tf.keras.Model):
    def __init__(self, filters, levels, FC_units):
        super().__init__()
        convM = []
        for i in range(levels-1):
            convM.append( Convolution((2**i)*filters) )
            convM.append( layers.MaxPool2D((2,2)) )
        convM.append( Convolution(2**(i+1)*filters) )
        convM.append( layers.GlobalAveragePooling2D() )
        self.convM = convM

        fcM = []
        for units in FC_units:
            fcM.append( layers.Dense(units, activation=layers.LeakyReLU(), kernel_initializer='he_normal') )
        fcM.append( layers.Dense(1, activation='relu') )
        self.fcM = fcM

    def __call__(self, x, attr, training=False):
        for layer in self.convM:
            x = layer(x, training) if isinstance(layer, Convolution) else layer(x)
        x = tf.concat([x, attr], axis=1)
        for layer in self.fcM:
            x = layer(x)
        return x