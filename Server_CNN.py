import flask, json, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #INFO and WARNING messages are not printed (put it before importing tf)
import tensorflow as tf
import keras
from keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from gevent.pywsgi import WSGIServer
from wsgidav import __version__ 

tf.get_logger().setLevel('ERROR')

app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False

with open('config.json') as json_file:
    data = json.load(json_file)

ip_address = data['host']
port = data['port']
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

def start_app():
    http_server = WSGIServer((ip_address, port), app, log=app.logger)
    
    http_server.set_environ({"SERVER_SOFTWARE": "WsgiDAV/{} ".format(__version__) +
                                            http_server.base_env["SERVER_SOFTWARE"]})
    print('\n----- Age Detection Server -----\nRunning {} on http://{}:{}'
          .format(http_server.get_environ()["SERVER_SOFTWARE"],
                  ip_address, port))
    
    http_server.serve_forever() 
    
if __name__ == "__main__":
    start_app()    