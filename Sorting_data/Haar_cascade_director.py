########## THis is a short program to take the images in the nimstim folder -> haar cascade -> image matrix
import numpy as np
import cv2
import os

def crop(imgpath):
    global cropped
    img = cv2.imread(os.path.join('NimStim\\', imgpath))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = cascade.detectMultiScale(img)
    for (x,y,w,h) in face:
        sub_face = img[y:y+h, x:x+w]
        cropped = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
    return cropped



size = input("What is the shape of the array?")

x1, x2, x3 = size.split()

emotion_list = ["an", "di", "fe", "ha", "ne", "sa", "sp"]

img_list = []
label_list = []
path = os.getcwd()

for i in os.listdir('NimStim'):
    missing = False
    for z,x in enumerate(emotion_list):
        if x.upper() in i or x in i:
            label_list.append([z])
            break
        else:
            if z == 7:
                print('Error for image: ' + str(i))
                missing = True
    if missing == False:
        try:
            img = crop(i)
            img1 = cv2.resize(img, dsize=(int(x1),int(x2)))
            if int(x3) == 1:
                img_list.append(cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY))

            elif int(x3) == 3:
                img_list.append(img1)
        except Exception as e:
            print(e, i)


y = np.asarray(label_list)
x = np.asarray(img_list)
print("X Shape: " , x.shape)
print("y Shape: ", y.shape)

np.save('X_haar_data_w_class.npy' ,x )
np.save('y_haar_data_w_class.npy', y)
