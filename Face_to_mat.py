import cv2
import numpy as np
import os


size = input("What is the shape of the array?")

x1, x2, x3 = size.split()

emotion_list = ["an", "ca", "di", "fe", "ha", "ne", "sa", "sp"]

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
            img = cv2.imread(path + '\\NimStim\\' + i)
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

np.save('X_data.npy' ,x )
np.save('y_data.npy', y)

# for j,i in enumerate(os.listdir('NimStim')): # "i" is the name of the image
#     missing = False
#     for z,x in enumerate(os.listdir('Class_em')):
#         if i in os.listdir('Class_em\\' + x):
#             label_list.append([z])
#             break
#         else:
#             if z == 7:
#                 print('Missing label for image: ' + str(i))
#                 missing = True
#     if missing == False:
#         try:
#             img = cv2.imread(path + '\\NimStim\\' + i)
#             img1 = cv2.resize(img, dsize=(int(x1),int(x2)))
#             img2 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
#             img_list.append(img2)
#         except Exception as e:
#             print(e, i)
#     else:
#         pass
#
# print(len(label_list), len(img_list))
# y = np.asarray(label_list)
# x = np.asarray(img_list)
# np.save('X_data', x)
# np.save('y_data', y)