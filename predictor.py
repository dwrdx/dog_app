import cv2                
import numpy as np

from keras.preprocessing import image                  
from keras.applications.resnet50 import decode_predictions, ResNet50
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input 
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

from dog import dog_names



def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('./saved_models/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = resnet50_preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def extract_VGG19(tensor):
	return VGG19(weights='imagenet', include_top=False).predict(vgg19_preprocess_input(tensor))

def predict_dog_breed(img_path):
    '''The function will convert the given image to a tensor then
    extract bottleneck features with VGG19. With those features, It uses 
    our trained model to predict the breed of the dog in the given image
    
    Paramters: img_path - path of the iamge
    Returns: breed name of the dog
    '''
    dog_model = Sequential()
    dog_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
    #dog_model.add(Dense(300, activation='relu'))
    #VGG19_model.add(Dropout(0.2))
    dog_model.add(Dense(1000, activation='relu'))
    dog_model.add(Dropout(0.3))
    dog_model.add(Dense(133, activation='softmax'))

    dog_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    dog_model.load_weights('./saved_models/weights.best.VGG19.hdf5')  
    
    tensor = path_to_tensor(img_path)
    feature = extract_VGG19(tensor)
    pred = np.argmax(dog_model.predict(feature))
    return dog_names[pred]

def funny_dog_breed_predictor(img_path):
    '''Predict the breed of the input picture, return a tuple,
    the 1st element of the tuple is True if the picture contains a human, 
    the 2nd element of the tuple is True if the picture contains a dog,
    the 3rd element of the tuple is the breed of the dog, or the breed of the 
    dog that the human looks like, or None if no human or dog is detected in 
    the picture
    
    Parameter: img_path - image path of the input picture
    Return: (is_human, is_dog, breed)
    '''
    is_human = face_detector(img_path)
    # is_human = False
    is_dog = dog_detector(img_path)
    if is_human or is_dog:
        breed = predict_dog_breed(img_path)
    else:
        breed = 'unknow'
        
    return is_human, is_dog, breed