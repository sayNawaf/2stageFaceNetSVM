import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector

import cv2
import random
class BlinkDetection:

    def __init__(self):
        
        self.cap = cv2.VideoCapture(0)
        self.detector = FaceMeshDetector(maxFaces=1)
        
        
        self.idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
        self.ratioList = []
        self.blinkCounter = 0
        self.counter = 0
        self.color = (255, 0, 255)
        self.BlinkTimes = random.randint(2, 5)

    def start(self):
        while True:
        
            #if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                #cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
            success, img = self.cap.read()
            img, faces = self.detector.findFaceMesh(img, draw=False)
        
            if faces:
                face = faces[0]
                #for id in idList:
                    #cv2.circle(img, face[id], 5,color, cv2.FILLED)
        
                leftUp = face[159]
                leftDown = face[23]
                leftLeft = face[130]
                leftRight = face[243]
                lenghtVer, _ = self.detector.findDistance(leftUp, leftDown)
                lenghtHor, _ = self.detector.findDistance(leftLeft, leftRight)
        
                #cv2.line(img, leftUp, leftDown, (0, 200, 0), 3)
                #cv2.line(img, leftLeft, leftRight, (0, 200, 0), 3)
        
                ratio = int((lenghtVer / lenghtHor) * 100)
                self.ratioList.append(ratio)
                if len(self.ratioList) > 3:
                    self.ratioList.pop(0)
                ratioAvg = sum(self.ratioList) / len(self.ratioList)
        
                if ratioAvg < 35 and self.counter == 0:
                    self.blinkCounter += 1
                    self.color = (0,200,0)
                    self.counter = 1
                if self.counter != 0:
                    self.counter += 1
                    if self.counter > 10:
                        self.counter = 0
                        self.color = (255,0, 255)
                
                #cvzone.putTextRect(img,f'Blink {BlinkTimes} time',(50,100),colorR=color)
                cvzone.putTextRect(img, f'Blink Count: {self.blinkCounter} of {self.BlinkTimes}', (5, 30),
                                colorR=self.color)
        
                
                img = cv2.resize(img, (900, 600))

                if self.blinkCounter == self.BlinkTimes:

                    break
                #imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
            
        
            cv2.imshow("Image", img)
            cv2.waitKey(25)
