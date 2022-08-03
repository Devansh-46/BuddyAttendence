#import all required modules

import cv2
import numpy as np
import face_recognition

#storing and converting the images in to RGB formates

img = face_recognition.load_image_file('images/img.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#For testing the images

imgTest = face_recognition.load_image_file('images/imgtest.jpg')
imgTest = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Finding theface in the images

faceLoc = face_recognition.face_locations(img)[0] #since sending only one image we get its first element

#Encode the face image

encodeimg = face_recognition.face_encodings(img)[0] 
cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (225, 0, 255), 2) #crop the face that is being detected

#Finding the face in the imageTest

faceLocTest = face_recognition.face_locations(imgTest)[0] #since sending only one image we get its first element

#Encode the face imageTest

encodeimgTest = face_recognition.face_encodings(imgTest)[0] 
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (225, 0, 255), 2) #crop the face that is being detected

result = face_recognition.compare_faces([encodeimg],encodeimgTest)
faceDis = face_recognition.face_distance([encodeimg],encodeimgTest) #distance the face that is being detected
print (result, faceDis)
cv2.putText(imgTest, f'{result} {round(faceDis[0],2)}', (50, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

#to show the image
cv2.imshow('Devansh',img)
#to show test image
cv2.imshow('Devansh Test',imgTest)
cv2.waitKey(0)

 