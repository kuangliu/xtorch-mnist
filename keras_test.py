import gzip
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def load_data():
    print('loading data...')
    path = './data/mnist.pkl.gz'
    f = gzip.open(path)
    data = pickle.load(f)
    f.close()
    return data


batch_size = 128
nb_classes = 10
nb_epoch = 20

(X_train, y_train), (X_test, y_test) = load_data()

# X_train.shape
# y_train.shape
# X_test.shape
# y_test.shape

X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255


Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


net = Sequential()
net.add(Dense(output_dim=512, input_dim=784))
net.add(Activation('relu'))
net.add(Dropout(0.2))
net.add(Dense(512))
net.add(Activation('relu'))
net.add(Dropout(0.2))
net.add(Dense(10))
net.add(Activation('softmax'))
net.summary()

net.compile(loss='categorical_crossentropy',
            optimizer=RMSprop(),
            metrics=['accuracy'])

history = net.fit(X_train,Y_train,
                  batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test,Y_test))
score = net.evaluate(X_test,y_test,verbose=1)
