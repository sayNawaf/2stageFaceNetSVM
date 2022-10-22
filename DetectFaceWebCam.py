
from PIL import Image
from numpy import asarray
#from faceRecognition import faceRecognition
import face_recognition
from sklearn.preprocessing import LabelEncoder
# import the opencv library
import cv2
import pickle
import numpy
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector


import random

  
  
# define a video capture object
class WebCamRead:
    def __init__(self):
        
        self.vid = cv2.VideoCapture(0)
        self.SVMmodel = pickle.load(open("SVMmodel.sav", 'rb'))
        # create the detector, using default weights
        self.encoder = LabelEncoder()
        self.encoder.classes_ = numpy.load('Names.npy')
       
        self.Normalizer = pickle.load(open("Normalizer.pkl", 'rb'))
        self.required_size = (160, 160)
        

    def StartCam(self):
        while(True):
            
            # Capture the video frame
            # by frame
            ret, frame = self.vid.read()

            

            

            imageS = cv2.resize(frame,(0,0),None,0.25,0.25)

            imageS = cv2.cvtColor(imageS,cv2.COLOR_BGR2RGB)
            
            #detect all the locations of face in the frame
            face_locations = face_recognition.face_locations(imageS)

            #get the the face encodings of all the faces in the frame
            face_embeddings = face_recognition.face_encodings(imageS,face_locations)
            
            #itterating through each of the face locations and its respective encoding
            for location,encoding in zip(face_locations,face_embeddings):
                #print("22:",encoding.shape)
                encoding = encoding.reshape(1,-1)
                embedding = self.Normalizer.transform(encoding)
                prediction = self.SVMmodel.predict(embedding)
                
                prob = self.SVMmodel.predict_proba(encoding)
                
                class_index = asarray(prediction[0])
                print(prob[0,class_index]*100)
                class_index = class_index.reshape(1,-1)
                
                name = self.encoder.inverse_transform(class_index)[0]
                
                
                x1,x2,y2,y1 = location
                x1,x2,y2,y1 = x1*4,x2*4,y2*4,y1*4
                #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
            
            
        
            
                    
            #display webcam feed
            cv2.imshow("webcame",frame)
            cv2.waitKey(1)
  
