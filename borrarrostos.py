#Nome: Johanes Rusch
#Link da origem do código: https://www.youtube.com/watch?v=Bbpftt2BLes&list=PLTYLKz3zyxKopRT1cTB4S4asbvUCaVxnF&index=56
#Sites de referência: https://pypi.org/project/face-recognition/
#Resumo:
# Este código identifica rostos e borra os mesmos, facilitando o trabalho, por exemplo, de criadores de conteúdo e até mesmo
# de emissoras de televisão para "esconder" os rostos das pessoas que não autorizaram o uso de sua imagem.

import cv2
from cvzone.FaceDetectionModule import FaceDetector

video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
detector = FaceDetector(minDetectionCon=0.5)


while True:
    _,img = video.read()
    img,bboxes = detector.findFaces(img,draw=False)
    img2 = img.copy()
    if bboxes:
        for bbox in bboxes:
            x,y,w,h = bbox['bbox']
            rec = img[y:y+h,x:x+w]
            # cv2.imshow('Face',rec)
            recBlur = cv2.blur(rec,(30,30))
            # cv2.imshow('Face', recBlur)
            img2[y:y+h,x:x+w] = recBlur


    cv2.imshow('IMG',img)
    cv2.imshow('IMG2', img2)
    cv2.waitKey(1)