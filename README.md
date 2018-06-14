--will be held--

**Work**

A program that recognizes faces and emotions.

**Usage**


```
 source bin/activate
```


```
 cd src
```

if exists your train data in training folder

```
 python face-recognation.py
```

if not exists

```
 python capture_picture.py -p Photo_Number -u User_Name
 python face-recognation.py
```

**Code**
 Import Required Modules

 Before starting the actual coding we need to import the required modules for coding. So let's import them first. 
 
 - *cv2:* is _OpenCV_ module for Python which we will use for face detection and face recognition.
 - *os:* We will use this Python module to read our training directories and file names.
 - *numpy:* We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.
 - *scipy:* SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering.
 - *sklearn:* Classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.



```python
faces, labels, users = prepare_training_data(trainingPath) # prepare_training_data function
face_recognizer = cv2.face.LBPHFaceRecognizer_create() # OpenCV LBP Face Recognation Algorithm
face_recognizer.train(faces, np.array(labels)) # # OpenCV model train function
```
This section sends the prepared user data to the face recognition algorithm. Model trains and learns them.



```python
video_capture = cv2.VideoCapture(0)
while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()
	...

	# Display the resulting frame
	cv2.imshow('Video', frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
```

For webcam.


```python
	# detect faces
	gray, faces = detect_face(frame) # detect_face function

	face_index = 0
	if faces is not None:
		#print(faces)
		# predict output
		for face in faces:
			(x, y, w, h) = face
			if w > 100:
```

If at least one face is detected; for each face, the face coordinates are transferred to a tuple. Because the detect_face(): function returns the coordinates of the face.


```python
		# draw rectangle around face 
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

```
Draw rectangle around face.


```python
		# extract features
		extracted_face = extract_face_area(gray, face, (0.03, 0.05)) #(0.075, 0.05)
		# predict smile
		prediction_result = predict_face_is_smiling(extracted_face)

		# draw extracted face in the top right corner
		frame[face_index * 64: (face_index + 1) * 64, -65:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)
```

Extract face features for smile predict.


```python
		face = gray[y:y + h, x:x + w]
		label = predict(frame, face)
		cv2.waitKey(50)
```






```python
		if prediction_result == 1:
		    cv2.putText(frame, label + " is similing",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)
		else:
		    cv2.putText(frame, label + " is not similing",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 155, 6)

		# increment counter
		face_index += 1
```






```python
 	def detect_face():
```
Convert the test image to gray image as opencv face detector expects gray images.


```python
 	def extract_face_area():
```

```python
	def prepare_training_data():
```


```python
	def predict_face_is_smiling():
```


```python
	def predict():
```


**Testing**

![Unknow](/doc/?.png)