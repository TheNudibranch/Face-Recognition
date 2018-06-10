from keras.models import load_model
import numpy as np

pred = []
actu = []

model  = load_model('bestsofar.hdf5')

X = np.load('X_validate_1.2.npy')
y = np.load('y_validate_1.2.npy')

for j,i in zip(y,X):
    pred.append(model.predict(np.resize(i, (1,300,300,3)))[0])
    actu.append(j)

pred_np = np.asarray(pred)
actu_np = np.asarray(actu)

print(pred_np.shape, actu_np.shape)

np.save('pred.npy', pred_np)
np.save('actu.npy', actu_np)
