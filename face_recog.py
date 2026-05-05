import numpy as np
import cv2
import os
from sklearn.neighbors import KNeighborsClassifier
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}


for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(dataset_path + fx)
        face_data.append(data_item)

        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_dataset.reshape((face_dataset.shape[0], -1))
face_labels = np.concatenate(labels,axis=0).ravel()

print(face_labels.shape)
print(face_dataset.shape)

# trainset = np.concatenate((face_dataset,face_labels),axis=1)
# print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(face_dataset,face_labels)
while True:

    ret,frame = cap.read()
    frame = cv2.resize(frame,(500,250))
    if ret == False:
        continue

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for face in faces:
        x,y,w,h = face
        offset = 5
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100)).flatten().reshape(1,-1)
       
        out = knn.predict(face_section)
        probs = knn.predict_proba(face_section)
        confidence = np.max(probs) * 100  
        # print(f"Confidence: {confidence}%")
        distances, indices = knn.kneighbors(face_section, n_neighbors=5)

        avg_distance = np.mean(distances)
        threshold = 20000

        if avg_distance > threshold:
            lab = "Unknown"
        else:
            out = knn.predict(face_section)
            lab = f"{names[int(out[0])]} ({int(confidence)}%)"
        # lab = f"{names[int(out[0])]},{confidence}%"
        cv2.putText(frame,lab,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
    cv2.imshow("Faces",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()