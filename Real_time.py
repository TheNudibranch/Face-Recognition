import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np


cam = cv2.VideoCapture(0)
cond = True

cv2.namedWindow("Real Time Emotion Detection")

# The following performs haar cascade on the image and return the same image with the haar box drawn around it
def box_image(image):
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = cascade.detectMultiScale(image)
    for (x, y, w, h) in face:
        cv2.rectangle(image, (x,y), (x+w, y+h), (255,0,0), 8)
    return image


#### The following is just to bring up a picture of what the haar cascade would look like if you chose the image you took.
def check_image(image):
    x = False
    rep_img = cv2.imread(image)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = cascade.detectMultiScale(rep_img)
    for (x, y, w, h) in face:
        cv2.rectangle(rep_img, (x,y), (x+w, y+h), (255,0,0), 8)
    cv2.imshow('Is this correct?', rep_img)
    cv2.waitKey(0)
    cv2.destroyWindow('Is this correct?')

###### THe following is used to crop the picture
# NOT return the picture with the rectangle
def crop(imgpath):
    global crop_img
    crop_img = 0
    img = cv2.imread(imgpath)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face = cascade.detectMultiScale(img)
    for (x,y,w,h) in face:
        sub_face = img[y:y+h, x:x+w]
        crop_img = cv2.cvtColor(sub_face, cv2.COLOR_BGR2RGB)
    x = crop_img
    return x


##### The following function is for presenting the final image.
def final_pic(image, emotion):
    fig = plt.figure()
    fig.canvas.set_window_title('Emotion Detection')
    fig.suptitle('Real Time Emotion Detection')
    ax = fig.add_subplot(111)
    ax.text(100, 50, 'Emotion Detected: ' + emotion, fontweight='bold', bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

##### The following classifies the pictures emotion and returns the emotion
def classify(model_name, imagepath):
    emotion_list = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    image = crop(imagepath)
    model = load_model(model_name)
    img1 = cv2.resize(image, dsize=(300,300))
    img2 = img1.astype('float32')
    img2 /= 255 # Make the pixel numbers between 0 and 1
    img3 = np.resize(img2, (1,300,300,3))

    emotion_one_hot = model.predict(img3)
    emotion = emotion_list[np.argmax(emotion_one_hot)]
    return emotion





while cond == True:
    ret, frame = cam.read()
    cv2.imshow("Real Time Emotion Detection", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "Real_time_detection.png"
        cv2.imwrite(img_name, frame)
        check_image('Real_time_detection.png')

cam.release()
cv2.destroyAllWindows()

img = cv2.imread('Real_time_detection.png')
box_im = box_image(img) # This will perform haar cascade and place a box around the face


cv2.imshow('Real Time Emotion Detection', box_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

em = classify('Conv_1_70acc.h5', 'Real_time_detection.png')

final_pic(img, em)