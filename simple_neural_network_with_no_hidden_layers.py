import numpy as np
from keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(x_train.shape[0], 'train samples','shape',x_train.shape)
print(x_test.shape[0], 'test samples',x_test.shape)

#reshaping the x_test,x_train to 60000*784
reshape=784
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(10000,784)
print(x_train.shape[0], 'train samples','shape',x_train.shape)
print(x_test.shape[0], 'test samples',x_test.shape)

#change int to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#normalize the traing and testing data
x_train=x_train/255
x_test=x_test/255


# convert class vectors to binary class matrices
from keras.utils import np_utils
NB_CLASSES = 10 # number of outputs = number of digits
print('before converting',y_test[0])
y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print( 'After converting',y_test[0])

#we now create a model(with no hidden layers) with 10 neurans(since we use 10 class labels),and input is 784(features)
#our activation function is softmax
from keras.models import Sequential
from keras.layers.core import Dense,Activation
model=Sequential()
model.add(Dense(NB_CLASSES,input_shape=(reshape,)))
model.add(Activation('softmax'))
model.summary()

#now we need compile the model,with optimser as SGD,loss function as  Categorical cross-entropy and to evaluate we use accuracy metrics
from keras.optimizers import SGD
model.compile(loss='categorical_crossentropy', optimizer=SGD(),
metrics=['accuracy'])

#now we train our model with epochs=200, and batch size as 128 and validation_split=0.2,verbose=1(progress bar)
history=model.fit(x_train,y_train,epochs=200,batch_size=128,validation_split=0.2,verbose=1)

#finally we evaluate the model
score = model.evaluate(x_test, y_test, verbose=1)
print("Test score:", score[0])
print('Test accuracy:', score[1])
