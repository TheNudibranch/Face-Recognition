from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

X_data = np.load('X_data.npy')
y_data = np.load('y_data.npy')
print('y shape: ', y_data.shape)
print('X shape: ', X_data.shape)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=55, train_size=0.80)

shape_1 = X_train.shape
shape_2 = X_test.shape # We will need the shapes for later resizing

X_train = X_train.astype('float32')
X_test = X_test.astype('float32') #Converting the pixel values to floats

X_train /= 255
X_test /= 255 #This will make all the pixel values between 0 and 1

y_train = keras.utils.to_categorical(y_train, 8)
y_test = keras.utils.to_categorical(y_test, 8) # One-hot encoding

epochs = 2
batch_size = 150

# X_train_fix = np.resize(X_train, (shape_1[0], shape_1[1], shape_1[2], 1))
# X_test_fix = np.resize(X_test, (shape_2[0], shape_2[1], shape_2[2], 1))

model = Sequential()


model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3)))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=50, epochs=13)
score = model.evaluate(X_test, y_test, verbose=1)
print(score)
model.save('Conv_0.2.h5')