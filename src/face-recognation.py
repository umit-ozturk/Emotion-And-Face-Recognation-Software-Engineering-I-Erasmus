import cv2
import os
import numpy as np
from scipy.ndimage import zoom
from sklearn.externals import joblib
from sklearn.svm import SVC


trainingPath = "training"

svc_1 = joblib.load('smile.joblib.pkl')


(im_width, im_height) = (1280, 920)

def detect_face(img):
	#convert the test image to gray image as opencv face detector expects gray images

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#load OpenCV face detector, I am using LBP which is fast
	cascPath = "lib/lbpcascade_frontalface.xml"
	faceCascade = cv2.CascadeClassifier(cascPath)

	#let's detect multiscale (some images may be closer to camera than others) images
	faces = faceCascade.detectMultiScale(
								gray,
								scaleFactor=1.3,
								minNeighbors=6,
								minSize=(100, 100),
							)
	if (len(faces) == 0): #if no faces are detected then return original img
		return None, None
	#return only the face part of the image
	return gray, faces


def detect_face_for_face_detection(img):
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

def extract_face_area(gray, face, offset_coefficients):
	(x, y, w, h) = face
	horizontal_offset = offset_coefficients[0] * w
	vertical_offset = offset_coefficients[1] * h
	y = y.astype(np.float64)
	x = x.astype(np.float64)
	w = w.astype(np.float64)
	h = h.astype(np.float64)
	horizontal_offset = horizontal_offset.astype(np.float64)
	vertical_offset = vertical_offset.astype(np.float64)

	vertical_offset_y = y+vertical_offset
	y_h = y+h
	x_horizontal_offset = x+horizontal_offset
	x_horizontal_offset_w = x-horizontal_offset+w

	vertical_offset_y = int(vertical_offset_y)
	y_h = int(y_h)
	x_horizontal_offset = int(x_horizontal_offset)
	x_horizontal_offset_w = int(x_horizontal_offset_w)

	extracted_face = gray[vertical_offset_y:y_h,
								x_horizontal_offset:x_horizontal_offset_w]
	new_extracted_face = zoom(extracted_face, (64. / extracted_face.shape[0],
								64. / extracted_face.shape[1]))
	new_extracted_face = new_extracted_face.astype(np.float32)
	new_extracted_face /= float(new_extracted_face.max())
	return new_extracted_face


def image_reader(imagePath):
	image = cv2.imread(imagePath)
	cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
	cv2.waitKey(1)	
	return image

def prepare_training_data(trainingPath):
	dirs = os.listdir(trainingPath) # listing training data folder
	faces = []
	labels = []
	users = []
	i = 0
	for user in dirs: # listing user folder
		if user.startswith("."): # Ignore For Ds.Store File
			continue;
		label = i
		i = i + 1			
		users.append(user)			
		trainingUserFolder = trainingPath + "/" + user # user folder
		userImagesNames = os.listdir(trainingUserFolder) # user images
		for imageName in userImagesNames:
			if imageName.startswith("."): # Ignore For Ds.Store File
				continue;
			imagePath = trainingUserFolder + "/" + imageName
			image = image_reader(imagePath)

			face, rect = detect_face_for_face_detection(image)
			if face is not None:
				faces.append(face)
				labels.append(label)
	cv2.destroyAllWindows()
	return faces, labels, users




def predict_face_is_smiling(extracted_face):
	return svc_1.predict([extracted_face.ravel()])[0]

def predict(pre_frame, face):
	img = pre_frame.copy()
	#make a copy of the image as we don't want to chang original image
	if face is not None:
		#predict the image using our face recognizer
		face_resize = cv2.resize(face, (im_width, im_height))
		label, confidence = face_recognizer.predict(face_resize)
		print(confidence)

		#get name of respective label returned by face recognizer
		label_text = users[label]
		print(label_text)
		return label_text


print("Preparing User data")
faces, labels, users = prepare_training_data(trainingPath)
print(labels)
print(users)
print("User Data prepared")

# total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_recognizer.train(faces, np.array(labels))



video_capture = cv2.VideoCapture(0)
while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	# detect faces
	gray, faces = detect_face(frame)

	face_index = 0
	if faces is not None:
		print(faces)
		# predict output
		for face in faces:
			(x, y, w, h) = face
			if w > 100:
				# draw rectangle around face 
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

				# extract features
				extracted_face = extract_face_area(gray, face, (0.03, 0.05)) #(0.075, 0.05)
				face = gray[y:y + h, x:x + w]
				label = predict(frame, face)
				cv2.waitKey(50)
				# predict smile
				prediction_result = predict_face_is_smiling(extracted_face)

				# draw extracted face in the top right corner
				frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)
				# annotate main image with a label
				if prediction_result == 1:
				    cv2.putText(frame, label + " is similing",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)
				else:
				    cv2.putText(frame, label + " is not similing",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)

				# increment counter
				face_index += 1

	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

#load test images
#test_img1 = image_reader("testing/umit1.jpg")
#perform a prediction
#predicted_img = predict(test_img1)

#print("Prediction complete")

#display both images
#cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
#cv2.waitKey(1)
#cv2.destroyAllWindows()


# accept = input("For capture photo please enter 'y'.")
# if accept == "y":
# 	raise Exception("Program is down")