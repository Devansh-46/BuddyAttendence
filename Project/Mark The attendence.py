#import all required modules

from base64 import encode
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'ImagesAttendence' # defining the path to the images for attendence
images = [] # list of images

classNames = []
mylist = os.listdir(path) # list of names of the images in the directory

for cl in mylist:  #import the images and their names and store it in the lists
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg) # adds the image to the listdir
    classNames.append(os.path.splitext(cl)[0]) # adds the image to the listdir

def findencoding(images):  # function to defined to find the encodings of the images and store them in the list
    encodeList = [] # list of encodings for the images
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeimg = face_recognition.face_encodings(img)[0]
        encodeList.append(encodeimg)
    return encodeList


def markattendence(name): # function to defined to markattendence
    with open('Attendence.txt', 'r+') as f:
        myDatalist = f.readlines()
        NameList = []

        for line in myDatalist:
            entry = line.split(', ')
            NameList.append(entry[0])
            if name not in NameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name, dtString}')




encodeKnownFaces = findencoding(images) 
print('encoding Complete...')

cap = cv2.VideoCapture(0) # capture the video in real time

while True: #checking the condition
    success, img = cap.read() # read the image from the capture stream
    img_small = cv2.resize(img, (0,0), None, 0.25, 0.25) # resize the image to the desired size
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB) 

    faces_current_frame = face_recognition.face_locations(img_small)
    encode_current_frame = face_recognition.face_encodings(img_small,faces_current_frame) 
    
    for encodeface, faceLoc in zip(encode_current_frame, encode_current_frame):
        match = face_recognition.compare_faces(encodeKnownFaces, encodeface)
        faceDis = face_recognition.face_distance(encodeKnownFaces, encodeface)
        matchIndex = np.argmin(faceDis)

        if match[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y1-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1-6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markattendence(name)
            

    cv2.imshow('webcame', img)
    cv2.waitKey(1)


