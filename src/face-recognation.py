import cv2
import os
import numpy as np

training_path = "training"

def detect_face(img):
	#convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#load OpenCV face detector, I am using LBP which is fast
	face_cascade = cv2.CascadeClassifier('lib/lbpcascade_frontalface.xml')

	#let's detect multiscale (some images may be closer to camera than others) images
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	if (len(faces) == 0): #if no faces are detected then return original img
	    return None, None

	#extract the face area
	(x, y, w, h) = faces[0]

	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

detect_face(img)