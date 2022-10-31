import keras
import tensorflow as tf
import json, cv2
from keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50

with open('config.json') as json_file:
    data = json.load(json_file)

num_classes = data['num_classes']
input_shape = tuple(data['input_shape'])

def creating_model():
    #Importing the ResNet-50 pre-trained model
    ResNet50_model = ResNet50(include_top=False, input_shape=input_shape, classes=num_classes)
    
    for layers in ResNet50_model.layers:
        layers.trainable=False
        
    #Defining the final layers of the model
    resnet50 = Flatten()(ResNet50_model.output)
    resnet50 = Dropout(0.5)(resnet50)
    resnet50 = Dense(512,activation='relu')(resnet50)
    resnet50 = Dropout(0.2)(resnet50)
    resnet50 = Dense(128,activation='relu')(resnet50)
    resnet50 = Dropout(0.2)(resnet50)
    resnet50 = Dense(32,activation='relu')(resnet50)
    resnet50 = Dropout(0.5)(resnet50)
    resnet50 = Dense(8,activation='softmax')(resnet50)
    resnet50_final_model = Model(inputs=ResNet50_model.input, outputs=resnet50)
    
    #Compiling the model
    resnet50_final_model.compile(loss='sparse_categorical_crossentropy', 
                                 optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                                 metrics=['sparse_categorical_accuracy'])   
    return resnet50_final_model


def loading_weights(model, path):
    #Loading weights from path
    model.load_weights(path)
    return model

def load_image(path, dim=(224, 224)):
    #Loading images
    img = cv2.imread(path)
    img = cv2.resize(img, dim)
    img = img.reshape(1, dim[0], dim[1], 3)
    return img

def get_prediction(model, img):
    pred = model.predict(img)
    pred = list(pred.argmax(axis = -1))
    return pred[0]