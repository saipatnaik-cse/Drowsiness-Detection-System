# Drowsiness-Detection-System

Link to download the Dataset
https://github.com/tzutalin/dlib-android/raw/master/data/shape_predictor_68_face_landmarks.dat
Just paste the link in your browser and then you can download it.

NOTE
----
Put all the file i.e alarm.wav and dataset file in the same folder.
Before execution download the python libraries used in the project.      
         

INTRODUCTION
-------------

Car accident is the major cause of death in which around 1.3 million people die every year. Majority of these accidents are caused because of distraction or the drowsiness of
driver. Construction of high-speed highway roads had diminished the margin of error for the driver. The countless number of people drives for long distance every day and night 
on the highway. Lack of sleep or distractions like the phone call, talking with the passenger, etc may lead to an accident. To prevent such accidents, we propose a system which
alerts the driver if the driver gets distracted or feels drowsy. Facial landmarks detection is used with help of image processing of images of the face captured using the camera,
for detection of distraction or drowsiness. This whole system is deployed on portable hardware which can be easily installed in the car for use.

SCOPE
-----

1. This project can be implemented in the form of mobile application to reduce the cost of hardware.
2. This project can be integrated with car, so that automatic speed control can be imparted if the driver is found sleeping.

PURPOSE
--------

Humans have always invented machines and devised techniques to ease and protect their lives, for mundane activities like traveling to work, or for more interesting  purposes like
aircraft travel. With  the advancement in technology,  modes  of  transportation  kept  on  advancing  and  our dependency on it started increasing exponentially. It has greatly
affected our lives as we know it. Now, we can travel to places at a pace that even our grandparents wouldn’t have thought possible. In modern times, almost everyone in this world 
uses some sort of transportation every day. Some people are rich enough to have their own vehicles while others use public transportation. However, there are some rules and codes
of conduct for those who drive irrespective of their social status. One of them is staying alert and active while driving.  Neglecting  our  duties  towards  safer  travel  has
enabled  hundreds  of thousands of  tragedies  to  get  associated  with  this wonderful  invention every year. It may seem like a trivial thing to most folks but following rules
and regulations on the road is of utmost importance. While on road, an automobile wields the most power and in irresponsible hands, it can be destructive and sometimes, that
carelessness can harm lives even of the people on the road. One kind of carelessness is not admitting when we are  too  tired  to  drive.  In  order  to  monitor  and  prevent 
a  destructive outcome from such negligence, many researchers have written research papers on driver drowsiness detection systems. But at times, some of the points and
observations made by the system are not accurate enough. Hence, to provide data and another perspective on the problem at hand, in  order  to  improve  their  implementations and
to  further  optimize  the solution, this project has been done. 

EXPLANATION OF THE CODE PART
----------------------------
**
import numpy as np
import dlib # detect and localize facial landmarks
import cv2
import threading
from threading import Thread
import imutils # image processing
from imutils import face_utils
from scipy.spatial import distance as dist
import pygame
**

