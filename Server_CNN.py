import flask, json, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #INFO and WARNING messages are not printed (put it before importing tf)
from flask import request, jsonify
from Utilities import creating_model, loading_weights, load_image, get_prediction
from gevent.pywsgi import WSGIServer
from wsgidav import __version__ 


app = flask.Flask(__name__)
app.config['DEBUG'] = True
app.config['JSON_SORT_KEYS'] = False

with open('config.json') as json_file:
    data = json.load(json_file)

ip_address = data['host']
port = data['port']
weights_path = data['weights_path']

model = creating_model()
model = loading_weights(model, weights_path)

@app.route('/')
def home():
    "Check Server Operation"
    try:
        result = {
                    'Result':"OK",
                    'Data':"Server Ready"
                 }
            
    except:
        result = {
                    'Result':"NOT OK",
                    'Data':"Server Not Ready"
                 }
            
        
    response = jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response 

@app.route('/api/v1/age_detection/get_prediction', methods=['POST']) 
def api_get_prediction():
    """Get prediction given an image"""
    
    img_path = request.json.get('img_path')
    img = load_image(img_path)
    
    pred = get_prediction(model, img)
    
    result = {
                  'Result':'OK',
                  'Prediction': int(pred)
              }

    response = jsonify(result)
    response.headers.set('Content-Type', 'application/json')
    
    return response



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