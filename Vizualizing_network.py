######### This script extracts the features of the convolutional network
from keras import models
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def crop(imgpath):
    global crop_img
    crop_img = 0
    img = cv2.imread(imgpath)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = cascade.detectMultiScale(img)
    for (x,y,w,h) in face:
        sub_face = img[y:y+h, x:x+w]
        crop_img = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
    img1 = crop_img
    img2 = cv2.resize(img1, dsize=(300,300))
    img3 = img2.astype('float32')
    img3 /= 255
    img4 = np.resize(img3, (1,300,300,3))
    return img4

model_path = 'Conv_1_70acc.h5'

model = load_model(model_path)
image_for_abstraction = os.path.join('Testing_images', 'Testing_image.jpg')
print(model.summary())



layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(crop('Disgust_sam.png'))



#### We will now display the channels for each activation layer
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)


images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]# Number of features in feature map

    size = layer_activation.shape[1] # Feature map has size (1, size, size, n_features)

    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))

    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]

            # Post process the image to make it "visually palatable"

            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')

            #Display the grid
            display_grid[col * size: (col +1 ) * size,
                            row * size : (row +1 ) * size] = channel_image
    scale = 1 / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.savefig('sam_{}'.format(layer_name))
    plt.show()