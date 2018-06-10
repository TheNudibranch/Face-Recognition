import shutil, os
import random

emotion_ab = ["an", "ca", "di", "fe", "ha", "ne", "sa", "sp"]
emotion_list = ["Anger", "Calm", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

for i in emotion_list:
    os.mkdir(os.path.join('Jitter_train', i))

for i in emotion_list:
    os.mkdir(os.path.join('Jitter_test', i))

for x,i in enumerate(random.sample(os.listdir('NimStim'), len(os.listdir('NimStim')))):
    for z,j in enumerate(emotion_ab):
        if j.upper() in i or j in i:

            if x <= 620:
                length_list = []
                for emotion in os.listdir('Jitter_train\\{}'.format(emotion_list[z])):
                    if emotion_list[z] in emotion:
                        length_list.append(1)

                dst = emotion_list[z] + '.{}.BMP'.format(len(length_list))
                dst_path = os.path.join('Jitter_train\\{}'.format(emotion_list[z]), dst)
                shutil.copyfile('NimStim\\' + i, dst_path)

            if x >620 :
                length_list = []
                for emotion in os.listdir('Jitter_test\\{}'.format(emotion_list[z])):
                    if emotion_list[z] in emotion:
                        length_list.append(1)
                dst = emotion_list[z] + '.{}.BMP'.format(len(length_list))
                dst_path = os.path.join('Jitter_test\\{}'.format(emotion_list[z]), dst)
                shutil.copyfile('NimStim\\' + i, dst_path)
