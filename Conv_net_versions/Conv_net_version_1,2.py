################ This conv net is almost identical to 0.2 and 0.3. The only different is that it will use that haar cascade images
# Version 1 does not contain any calm images
# The last dense layer is adjusted to reflect that
# Now has jitter

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

X_data = np.load('X_haar_data_no_ca.npy')
y_data = np.load('y_haar_data_no_ca.npy')

s = np.arange(X_data.shape[0])#shuffles xdata and ydata but preserves relationships
np.random.seed(22)
np.random.shuffle(s)
X_data = X_data[s]
y_data = y_data[s]

X_data = X_data.astype('float32')#casts x data to float
X_data /= 255 #normalizes x data

X_validate = X_data[-57:]
X_data = X_data[:-57]
itshape = X_data.shape[0] #I want its length but I'm going to turn it into a list
y_validate = y_data[-57:]
y_data = y_data[:-57]
y_data = list(y_data)
X_data = list(X_data)

datagen = ImageDataGenerator(
        rotation_range=5,
        horizontal_flip=True,
        channel_shift_range = 15.0,
        brightness_range=(0.6,1.4),shear_range=0.05)

for i in range(itshape):
    t = X_data[i]
    k=0
    x = t.reshape((1,) + t.shape)
    for batch in datagen.flow(x, batch_size=1):
        X_data.append(batch[0])
        y_data.append([y_data[i]])
        k += 1
        if k > 2:
            break
X_data = np.asarray(X_data)
y_data = np.asarray(y_data)

y_hot = keras.utils.to_categorical(y_data, 7) #one hot
y_validate = keras.utils.to_categorical(y_validate, 7)
valset = (X_validate,y_validate)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_data.shape[1:]))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])

filepath="bestsofar.hdf5" #file is constantly overwritten with best model from current run, saves disk space
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max') #saves the best models
earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=6, verbose=0, mode='auto') #makes it stop if the test accuracy does not improve after so many runs
callbacks_list = [checkpoint,earlystop]

model.fit(X_data, y_hot, batch_size=50, epochs=30,validation_data=valset,callbacks = callbacks_list) #uses 10% as test data

np.save('X_validate_1.2.npy', X_validate)
np.save('y_validate_1.2.npy', y_validate)
