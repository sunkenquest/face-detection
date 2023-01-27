import cv2
from os import listdir, path

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image_directory = "sample faces"

for i in range(5):
    for image_name in listdir(image_directory):
        image_path = path.join(image_directory, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for(x,y,w,h) in faces:
            face_img = img[y:y+h, x:x+w]
            face_img_resized = cv2.resize(face_img, (70,80))
        cv2.imwrite("cropped faces/{}" .format(image_name), face_img_resized)

