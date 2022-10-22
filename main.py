from generateDataset import GenDataset
from SVMclassifiertraining import trainSVM
from DetectFaceWebCam import WebCamRead
from DetectEyeBlink import BlinkDetection
from HandGestureDetection import HandGesture
import random
import time
pathTodata = "Users"

option = random.randint(0, 1)
train = False


if train == True:
    trainX, trainy, testX, testy = GenDataset(pathTodata)
    trainSVM(trainX, trainy, testX, testy)

if option == 0:
    bd = BlinkDetection()
    bd.start()
    
else:
    hg = HandGesture()
    hg.start()
    
time.sleep(2)
CamReader = WebCamRead()
CamReader.StartCam()
