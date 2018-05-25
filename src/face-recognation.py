import cv2
import os
import numpy as np
from PIL import Image

trainingPath = "training"

subjects = ["", "Umit Ozturk", "Elvis Presley"]

def detect_face(img):
	#convert the test image to gray image as opencv face detector expects gray images

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#load OpenCV face detector, I am using LBP which is fast
	cascPath = "lib/haarcascade_frontalface_alt.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)

	#let's detect multiscale (some images may be closer to camera than others) images
	faces = faceCascade.detectMultiScale(
								gray,
								scaleFactor=1.1,
								minNeighbors=6,
								minSize=(100, 100),
							)
	faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
	if (len(faces) == 0): #if no faces are detected then return original img
	    return None, None

	#extract the face area
	(x, y, w, h) = faces[0]

	#return only the face part of the image
	return gray[y:y+w, x:x+h], faces[0]

def image_reader(imagePath):
	image = cv2.imread(imagePath)
	cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
	cv2.waitKey(50)	
	return image

def prepare_training_data(trainingPath):
	dirs = os.listdir(trainingPath) # listing training data folder
	faces = []
	labels = []
	i = 1
	for user in dirs: # listing user folder

		label = i
		i =+ 1
		print(user)
		if user.startswith("."): # Ignore For Ds.Store File
			continue;		
		trainingUserFolder = trainingPath + "/" + user # user folder
		userImagesNames = os.listdir(trainingUserFolder) # user images
		for imageName in userImagesNames:
			if imageName.startswith("."): # Ignore For Ds.Store File
				continue;
			imagePath = trainingUserFolder + "/" + imageName
			image = image_reader(imagePath)
			face, rect = detect_face(image)
			if face is not None:
				faces.append(face)
				labels.append(label)
	cv2.destroyAllWindows()

	return faces, labels

def draw_rectangle(img, rect):
	(x, y, w, h) = rect
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
#function to draw text on give image starting from
#passed (x, y) coordinates. 
def draw_text(img, text, x, y):
	cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
	#make a copy of the image as we don't want to chang original image
	img = test_img.copy()
	#detect face from the image
	face, rect = detect_face(img)
	if face is not None and rect is not None:
		#predict the image using our face recognizer 
		label, confidence = face_recognizer.predict(face)
		#get name of respective label returned by face recognizer
		label_text = subjects[label]

		#draw a rectangle around face detected
		draw_rectangle(img, rect)
		#draw name of predicted person
		draw_text(img, label_text, rect[0], rect[1]-5)		
	else:
		draw_rectangle(img, rect)

	return img


print("Preparing User data")
faces, labels = prepare_training_data(trainingPath)
print(labels)
print("User Data prepared")

# total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))



# video_capture = cv2.VideoCapture(0)
# while True:
# 	# Capture frame-by-frame
# 	ret, frame = video_capture.read()

# 	# detect faces
# 	gray, detected_faces = detect_face(frame)

# 	face_index = 0

# 	# predict output
# 	for face in detected_faces:
# 		(x, y, w, h) = face
# 		if w > 100:
# 			# draw rectangle around face 
# 			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

# 			# extract features
# 			extracted_face = extract_face_features(gray, face, (0.03, 0.05)) #(0.075, 0.05)
# 			print(extracted_face)
# 			# predict smile
# 			prediction_result = predict_face_is_smiling(extracted_face)

# 			# draw extracted face in the top right corner
# 			frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)

# 			# annotate main image with a label
# 			if prediction_result == 1:
# 			    cv2.putText(frame, "SMILING",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)
# 			else:
# 			    cv2.putText(frame, "NOT SMILINGs",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)

# 			# increment counter
# 			face_index += 1
			
# 	# Display the resulting frame
# 	cv2.imshow('Video', frame)

# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break

#load test images
#test_img1 = image_reader("testing/umit1.jpg")
#perform a prediction
#predicted_img1 = predict(test_img1)

#print("Prediction complete")

#display both images
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.waitKey(1)
#cv2.destroyAllWindows()


# accept = input("For capture photo please enter 'y'.")
# if accept == "y":
# 	raise Exception("Program is down")