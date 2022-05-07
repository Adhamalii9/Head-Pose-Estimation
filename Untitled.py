#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.spatial import distance as dist
from playsound import playsound
#from imutils.video import VideoStream
from imutils import face_utils
#from threading import Thread
import numpy as np
#import multiprocessing
#import argparse
import imutils
#import pygame
from pygame import *
import time
import dlib
import cv2
import os


# In[2]:


mixer.init()
def playsound():
    
    mixer.music.load('wakeup.mp3')
    mixer.music.play()

def stopsound():
    mixer.music.stop()
    
    
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
  
    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
   
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
   
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    
    return distance


# In[3]:


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
COUNTER = 0

print("-> Loading the predictor and detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("-> Starting Video Stream")
vs = cv2.VideoCapture(0)
time.sleep(1.0)


# In[4]:


alarm=False
alarm1=False
drowin=False
yawning=False

def make_1080p():
    vs.set(3, 1920)
    vs.set(4, 1080)

def make_720p():
    vs.set(3, 1280)
    vs.set(4, 720)

def make_480p():
    vs.set(3, 640)
    vs.set(4, 480)

def change_res(width, height):
    vs.set(3, width)
    vs.set(4, height)

make_1080p()

while True:

    r, frame = vs.read()
    cv2.normalize(frame, frame, 0, 500, cv2.NORM_MINMAX)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects: 
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH :
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm==False:
                    alarm=True
                    drowin=True
                    playsound()
                    print('open alarm of drown')

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            if alarm == True:
                alarm=False
                stopsound()
                print('stop alarm of drwon')


        if (distance > YAWN_THRESH):
            yawning=True
            cv2.putText(frame, "Yawn Alert", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if alarm1==False:
                alarm1=True
                playsound()
                print('play alarm of yawn')


        else:
            if alarm1== True:
                alarm1=False
                stopsound()
                print('stop alarm of yawn')

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
stopsound()


# In[ ]:





# In[ ]:




