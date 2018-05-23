import cv2
import os
import numpy as np


def detect_face(img):
	#convert the test image to gray image as opencv face detector expects gray images
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#load OpenCV face detector, I am using LBP which is fast
	#there is also a more accurate but slow Haar classifier
	face_cascade = cv2.CascadeClassifier('lib/lbpcascade_frontalface.xml')

	#let's detect multiscale (some images may be closer to camera than others) images
	#result is a list of faces
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

	#if no faces are detected then return original img
	if (len(faces) == 0):
	    return None, None

	#under the assumption that there will be only one face,
	#extract the face area
	(x, y, w, h) = faces[0]

	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

detect_face()