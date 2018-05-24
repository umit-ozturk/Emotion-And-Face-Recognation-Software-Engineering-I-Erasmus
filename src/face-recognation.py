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

def prepare_training_data(training_path):
	dirs = os.listdir(training_path) # listing training data folder
	faces = []
	users = []
	for user in dirs: # listing user folder
		label = user
		print(user)
		training_user_folder = training_path + "/" + user # user folder
		user_images_names = os.listdir(training_user_folder) # user images
		for image_name in user_images_names:
			if image_name.startswith("."): # Ignore For Ds.Store File
				continue;
			image_path = training_user_folder + "/" + image_name
			image = cv2.imread(image_path)
			cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
			cv2.waitKey(50)
			face, rect = detect_face(image)
			if face is not None:
				faces.append(face)
				users.append(user)
	cv2.destroyAllWindows()

	return faces, users

print("Preparing User data")
faces, users = prepare_training_data(training_path)
print(users)
print("User Data prepared")

# total faces and users
print("Total faces: ", len(faces))
print("Total labels: ", len(users))	