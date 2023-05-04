#!/user/kh3191/.conda/envs/tf/bin/python
import tensorflow as tf
import numpy as np
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

import argparse
from argparse import RawTextHelpFormatter
def parse_option():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--l2', type=float, default=0.0002)
    parser.add_argument('--n_epochs', type=int, default=32)
    parser.add_argument('--bias', type=bool, default=False)
    opt = parser.parse_args()
    return opt

opt = parse_option()
print(opt)
optimizer = tf.keras.optimizers.SGD(learning_rate=opt.lr)
    

DEVICE = "/gpu:0"

with tf.device(DEVICE):
    # %% load in the training data, and do some processing
    (xtrain,ytrain), (xtest,ytest) = tf.keras.datasets.mnist.load_data()
    ## convert to one-hot vectors
    ytrainp = tf.keras.utils.to_categorical(ytrain)
    ytestp = tf.keras.utils.to_categorical(ytest)
    ## convert to [0,1] and reshape to NumObs x 28 x 28
    xtrainp = tf.cast(np.reshape(xtrain/255.,[-1,28,28]),tf.float32)
    xtestp = tf.cast(np.reshape(xtest/255.,[-1,28,28]),tf.float32)
    #print(f'xtrainp:{xtrainp.shape} ytrainp:{ytrainp.shape}')
    #print(f'xtestp:{xtestp.shape} ytestp:{ytestp.shape}')

    model = tf.keras.models.Sequential([
        # Layer 1
        tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, activation = 'relu', input_shape = (28,28,1), 
                               kernel_regularizer=tf.keras.regularizers.L2(opt.l2)),
        # Layer 2
        tf.keras.layers.Conv2D(filters = 16, kernel_size = 5, strides = 1, use_bias=opt.bias),
        tf.keras.layers.BatchNormalization(),
        # — — — — — — — — — — — — — — — — #
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),
        tf.keras.layers.Dropout(opt.dropout),
        # — — — — — — — — — — — — — — — — #
        # Layer 3
        tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, activation = 'relu', 
                               kernel_regularizer=tf.keras.regularizers.L2(opt.l2)),
        # Layer 4
        tf.keras.layers.Conv2D(filters = 32, kernel_size = 3, strides = 1, use_bias=opt.bias),
        tf.keras.layers.BatchNormalization(),
        # — — — — — — — — — — — — — — — — #
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size = 2, strides = 2),
        tf.keras.layers.Dropout(opt.dropout),
        tf.keras.layers.Flatten(),
        # — — — — — — — — — — — — — — — — #
        # Layer 5
        tf.keras.layers.Dense(units = 180, use_bias=False), # no bias to ensure params<100000
        tf.keras.layers.BatchNormalization(),
        # — — — — — — — — — — — — — — — — #
        tf.keras.layers.Activation('relu'),
        # — — — — — — — — — — — — — — — — #
        # Layer 6
        tf.keras.layers.Dense(units = 110, use_bias=False), # no bias to ensure params<100000
        tf.keras.layers.BatchNormalization(),
        # — — — — — — — — — — — — — — — — #
        tf.keras.layers.Activation('relu'),
        # — — — — — — — — — — — — — — — — #
        # Layer 7
        tf.keras.layers.Dense(units = 50, use_bias=opt.bias),
        tf.keras.layers.BatchNormalization(),
        # — — — — — — — — — — — — — — — — #
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(opt.dropout),
        # — — — — — — — — — — — — — — — — #
        # Layer 8
        tf.keras.layers.Dense(units = 10, activation = 'softmax')
        ])
    print(model.summary())

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics='accuracy')
    model.fit(xtrainp, ytrainp, epochs=opt.n_epochs, verbose=1)

    test_loss, test_acc = model.evaluate(xtestp, ytestp)
    print(f'Test loss: {test_loss:.4f}\nTest accuracy: {test_acc}')
