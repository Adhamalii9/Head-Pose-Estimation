# Drowsiness-Detection
According to this paper(https://link.springer.com/chapter/10.1007/978-981-33-4866-0_28)
It is a real-time system which will detect the drowsiness among car drivers by capturing image continuously and will warn the driver whenever they will feel sleepy. The innovation of the present work is established on blinking of eyes and yawn frequency. The per closure value of eye is examined for detection of drowsiness, and whenever it exceeds a certain value, then the driver is recognized to be sleepy. Similarly, we will inspect the yawn value to detect the drowsiness and whenever it exceeds its minimum threshold value it will give yawn alert.

Python programming-Dlib-OpenCV-Face detection-Eye detection-Yawn detection

# The following steps were used to test the system:
Step 1: As the recording starts, the system starts reading the frame, and then, it is
re-scaled and transformed to gray scale images.

Step 2: cv2 detector is carve up. The position of the face is detected by it.

Step 3: Dlib predictor determine facial landmark and find position of eyes and
mouth.

Step 4: The coordinates of eyes and mouth are taken to calculated the EAR ratio
and the yawn distance value to find whether the driver is drowsy or not.

Step 5: The ratio and distance is calculated and compared with the threshold value
which was set as EAR = 0.3 and yawn = 20 to determine whether the person is
sleepy or not. If the EAR is below the threshold, it is determined that driver is
sleepy and so if the yawn value is more than the threshold.

Step 6: If the value remain consistent for given number of frames, the alarm is
raised so as to prevent accident.

# To run this code 
- download this file (https://drive.google.com/drive/folders/1xwzUGqHMG41eXUTnJ-hZsBhiAm083EDi?usp=sharing) , and put it in the same path of GP.ipynb
