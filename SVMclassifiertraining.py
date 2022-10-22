# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import numpy
import pickle

def trainSVM(trainX, trainy, testX, testy):
    # load dataset
    #data = load('faces-embeddings.npz')
    #trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    # normalize input vectors

    in_encoder = Normalizer(norm='l2')
    in_encoder.fit(trainX)
    pickle.dump(in_encoder, open('Normalizer.pkl', 'wb'))
    print(trainX.shape)
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    numpy.save('Names.npy', out_encoder.classes_)

    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    # fit model
    
    model = SVC(kernel='linear', probability=True)
    print("Training+++++")
    model.fit(trainX, trainy)
    print("finished training")
    pickle.dump(model, open("SVMmodel.sav", 'wb'))
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score
    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))