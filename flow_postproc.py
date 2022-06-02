#import all the library needed
import numpy as np
#from pyiotown_wicom import postprocess
import pyiotown.post
from joblib import load, dump
from sklearn.ensemble import IsolationForest
import threading

counter = 0

#Thresholding value
upper_thresh = 10
lowe_thresh = 0

#Sliding window setting (depends on the data collection cycle)
#in this case, data collection cyle is 1 minute
batch_size = 60 # 60 = 1 hour
train_number = 1440 # 1440 = 1 day

def train(): #For retraining model & overwriting model
    global arr_sensor

    #model initialization
    model = IsolationForest(n_estimators=100, max_samples=500, random_state=42, contamination=0.05)

    #data preprocess
    arr_sensor = arr_sensor.reshape(-1,1)

    #model training
    model.fit(arr_sensor)

    #save model
    dump(model, 'path/filename.joblib')

def post_process(message):
    global arr_sensor

    sensor = np.array([message['data']['sensor']]).T

    if counter ==0:
        #mode 1: Using initial model
        model = load('path/filename.joblib')
        counter += 1
    
    elif counter <= train_number:
        #mode 2: Keep using initial model until the data stored in array(window) is enough
        counter += 1
    
    elif counter == (train_number + 1) :
        #mode 3: retrain the model
        thread = threading.Thread(target=train)
        if thread.is_alive():
            print('thread still running')          
        else:
            print('thread is starting')
            thread.start()
        counter += 1
        thread.join()
    
    elif counter <= (train_number + 1 + batch_size):
        #mode 4: sliding window method
        counter += 1

    else:
        #optimize the array size of sliding window
        arr_sensor =  arr_sensor[-train_number:]
        counter = (train_number+1)
    
    #input stream data to the window
    arr_sensor = np.append(arr_sensor,sensor)

    #preprocess the data for anomaly detection
    newsensor = sensor.reshape(1,-1)

    #anomaly detection / Isolation Forest Prediction
    anomaly_score =  model.decision_function(newsensor)
    anomaly_sensor = model.predict(newsensor)

    #clustering between normal & abnormal
    if anomaly_sensor > 0 and float(sensor[0]) > lowe_thresh and float(sensor[0]) < upper_thresh: #normal condition
        sensor_status = 'normal'
    else: #abnormal condition
        sensor_status = 'abnormal'

    #Store the data in order to send it back to IoT.own
    changedata = {}
    changedata['sensor_status'] = sensor_status
    changedata['sensor_value'] = float(sensor[0])
    changedata['anomaly_score'] = round(float(anomaly_score[0]),2)

    message['data'] = changedata
    return message


if __name__ == '__main__':
    arr_sensor = np.array([[]])
    postproc_name = 'post_process name'
    url = "url name"
    username = "username"
    password = "token"
    #postprocess(url,postproc_name,post_process, username, password)
    pyiotown.post.postprocess(url,postproc_name,post_process, username, password)
