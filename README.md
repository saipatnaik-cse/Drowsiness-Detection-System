# Drowsiness-Detection-System

Link to download the Dataset
https://github.com/tzutalin/dlib-android/raw/master/data/shape_predictor_68_face_landmarks.dat
Just paste the link in your browser and then you can download it.

NOTE
----
Put all the file i.e alarm.wav and dataset file in the same folder. <br />
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

import numpy as np <br />
import dlib # detect and localize facial landmarks <br />
import cv2 <br />
import threading <br />
from threading import Thread <br />
import imutils # image processing <br />
from imutils import face_utils <br />
from scipy.spatial import distance as dist <br />
import pygame <br />


* Here we have imported scipy package to compute the Euclidean distance between facial landmarks point in the eye aspect ratio calculation. <br />
* we have imported Thread class so we can play our alarm in a separate thread from the main thread to ensure our script doesn't pause execution while the alarm sounds. <br />
* Pygame to play the alarm. <br />
* dlib library to localize the facial landmarks. <br />

def sound_alarm(): <br />
    pygame.mixer.init() <br />
    pygame.mixer.music.load("alarm.wav") <br />
    pygame.mixer.music.play() <br />
    
* We have defined the sound_alarm function in which we are initializing  module of mixer (pygame.mixer.init()) .
* pygame.mixer.music.load("path") will load the music file for playback
* pygame.mixer.music.play() start the playback.

def eye_aspect_ratio(eye): <br />
    A = dist.euclidean(eye[1], eye[5]) <br />
    B = dist.euclidean(eye[2], eye[4])   #vertical distance <br />
    C = dist.euclidean(eye[0], eye[3])   #horizontal distance <br />
    ear = (A+B)/(2.0*C) <br />
    return ear <br />
  
  ![facial_landmarks_68markup-768x619](https://user-images.githubusercontent.com/70318294/124358827-b86af600-dc3f-11eb-89c7-3f57ec41cf74.jpg)


    
 * In this part of code ,we define the eye_aspect_ratio function which is used to compute ratio of distances between the vertical eye landmark and the distance between              horizontal eye landmarks. <br />
 * The return value of eye aspect ratio will be approximately constant when the eye is open. the value will then rapid decrease towards zero during a blink. <br />
 * If the eye is closed ,the eye aspect ratio will again remain approximately constant ,but will be much smaller than the ratio when the eye is open. <br />
 * On the top-left we have an eye that is fully open with the eye facial landmarks plotted. Then on the top-right we have an eye that is closed .The bottom then plots the eye aspect ratio over time. <br />
 * As we can see ,the eye aspect ratio is constant (indicating the eye is open ),then rapidly drops to zero, then increases again,indicating  a blink has taken place. <br />
 * In our project, we are monitoring the eye aspect ratio to se if the value falls but does not increase again,thus implying that the person has close their eyes. <br />
 
EYE_AR_THRESH = 0.3 <br />
EYE_AR_CONSEC_FRAMES = 40 <br />

COUNTER = 0 <br />
ALARM_ON = False <br />

![blink_detection_plot](https://user-images.githubusercontent.com/70318294/124358788-80fc4980-dc3f-11eb-8340-499c23b677d8.jpg)

* In this part of code we defined the EYE_AR_THRESH. If the eye aspect ratio falls below this threshold,we will start counting the number of frames the person has closed their eyes. If the number of frames the person has closed their eyes in exceeds EYE_AR_CONSEC_FRAMES ,we will play the alarm. <br />
* Here ,I have taken EYE_AR_CONSEC_FRAMES to be 40,means if person has closed their eyes for 40 consecutive frames ,we will play the alarm. <br />
* Then we defined COUNTER, the total number of consecutive frames where the eye aspect ratio is below EYE_AR_THRESH . <br />
* If COUNTER  exceeds EYE_AR_CONSEC_FRAMES ,then we will update the boolean ALARM_ON <br />



predictor_path = 'shape_predictor_68_face_landmarks.dat' <br />
detector = dlib.get_frontal_face_detector() # return a detector that is a function we can use to retrieve the faces information <br />
predictor = dlib.shape_predictor(predictor_path) <br />




* Grab the indexes of the facial landmarks for the left and right eye,respectively <br /><br />
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] <br />
( rStart , rEnd ) =face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] <br /><br />




cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == False:
        print('Failed to capture frame from camera,Check camera \n')
        break
        # cv2.imshow(frame)

    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)    # determine the facial landmarks for face region
        shape = face_utils.shape_to_np(shape) #converting to numpy array

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR+rightEAR)/2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    d=threading.Thread(target=sound_alarm)
                    d.setDaemon(True)
                    d.start()

                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()

* Here,to extract the eye regions from a set of facial landmarks, we simply need to know the correct array slice indexes. <br />
* Now ,to capture the video we are starting our webcam and then reading the frames , which we then preprocess by resizing it to have a width of 450 pixels and converting it to grayscale. <br />
* applying dlib’s face detector to find and locate the face(s) in the image <br />
* For each of the detected faces, we apply dlib’s facial landmark detector and convert the result to a NumPy array. <br />
* Using NumPy array slicing we can extract the (x, y)-coordinates of the left and right eye, respectively so that we can compute their eye aspect ratios . <br />
* Then we are visualizing each of the eye regions from our frame by using cv2.drawContours function and we are now ready to check to see if the person in our webcam  is starting to show symptoms of drowsiness. <br />
* Then we are checking to see if the eye aspect ratio is below the “blink/closed” eye threshold,EYE_AR_THRESH . If it is, we increment  COUNTER , the total number of consecutive frames where the person has had their eyes closed If COUNTER exceeds EYE_AR_CONSEC_FRAMES , then we assume the person is starting to doze off. <br />
* we created  a separate thread responsible for calling sound_alarm to ensure that our main program isn’t blocked until the sound finishes playing then we draw the text DROWSINESS ALERT ! on our frame. <br />
* if the eye aspect ratio is larger than EYE_AR_THRESH , indicating the eyes are open. If the eyes are open, we reset COUNTER and ensure the alarm is off.
And then finally our drowsiness detector frame display to our screen. <br />

