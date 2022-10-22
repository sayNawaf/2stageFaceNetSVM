# FaceAuth
ABSTRACT
Latest Facial Recognition systems have achieved great popularity and and diverse domain applications due to its ease of installment and high accuracy.
FaceNet though generates a good face embeddings still suffers from some issues when implemented in a practical real world scenario,my implementation i overcome most of its practical issues.

some issues with FaceNet in real World implmentation are as follows:

1) UNREGISTERED FACES
if a face is not registered with a FaceNet embedding... it will map the closest embedding present in the database when the unregisterd face uses the system for authentication.keeping a minimum threshold to map a person with its registered embedding does improve the situation but doesnt solve it completly,because the threshold value varies depending upon the all the poeple who registered,hence results in poor accuracy.

2) Proxy Face Bypass using a Photo
faceNet generates embeddings for a face irrespective the source of input ie can be through a photo...hence makes it easy to bypass through a proxy

My solution

1) UNREGISTERED FACES
Upon registering all faces embeddings and before the Authentication process begins the script automatically trains a SVM classifier to classify an embedding to respective Person by making each registered face as a Class to classified an embedding as input.
this way rather then simply mapping embedding to closest face embeding using distance its learns to classify uniquely based on the registered group of faces..also include an UNKNOWN Class which contains faces of people not present in the group of registered faces this way SVM will be able to classify an unregistered person.
for great perfomance and robustness i used 5 pics for each class ie 4 for training and 1 for validation.

2) Proxy Face Bypass using a Photo
script runs a 2 stage authentication process where 1st stage verifies if person in camera is a live person by prompting him/her to do a random gesture which could be anything from the following,
to blink at a specified amount of times(number generated randomly from 1 to 5),thumnbs up,thumbs down,peace and fist.on succesfully perfoming the prompted gesture the script starts the 2nd stage which is the face authentication.

