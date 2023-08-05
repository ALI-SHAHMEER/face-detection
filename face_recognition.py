from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime




video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('/home/wolf/Desktop/Face_detection/Data/haarcascade_frontalface_default.xml')

COL_NAME = ['NAME',"TIME"]

with open('Data/names.pkl','rb') as f:
    LABELS = pickle.load(f)
with open('Data/faces_data.pkl','rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES,LABELS)
while True:
    ret,frame = video.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        crop_img = frame[y:y+h,x:x+w, :]
        resized_img = cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        ts = time.time()
        
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        time_stamp = datetime.fromtimestamp(ts).strftime("%a %d %b %Y, %I:%M%p")
        exisit = os.path.isfile("Attendance/Attendance_"+ date +".csv")
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(0,0,255),-1)
        cv2.putText(frame,str(output[0]),(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),1)
        attendance = [str(output[0]), str(time_stamp)]
    cv2.imshow("Frame",frame)
    k = cv2.waitKey(1)
    if k == ord('o'):
        
        if exisit:
            with open("Attendance/Attendance_"+ date +".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
            csvfile.close()
        else:
            with open("Attendance/Attendance_"+ date +".csv","+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAME)
                writer.writerow(attendance)
            csvfile.close()
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()