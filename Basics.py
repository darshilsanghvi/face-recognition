import numpy as np
import cv2
import face_recognition

imgElon = face_recognition.load_image_file('images/elon musk.jfif')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgElon2 = face_recognition.load_image_file('images/elon musk 2.jfif')
imgElon2 = cv2.cvtColor(imgElon2, cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceloc[3], faceloc[0]), (faceloc[1], faceloc[2]), (255, 0, 255), 2)

faceloc2 = face_recognition.face_locations(imgElon2)[0]
encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
cv2.rectangle(imgElon2, (faceloc2[3], faceloc2[0]), (faceloc2[1], faceloc2[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeElon2)
print(results)
cv2.imshow('Elon Musk', imgElon)
cv2.imshow('ELon Musk 2', imgElon2)
cv2.waitKey(0)
