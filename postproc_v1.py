#import all the library needed
import numpy as np
from pyiotown_wicom import postprocess
import pyiotown.post
from joblib import load, dump
from sklearn.ensemble import IsolationForest
import threading
import warnings
import sys

warnings.filterwarnings("ignore")

counter = 1

#Thresholding value
upper_thresh_temp = 40
lower_thresh_temp = 5

upper_thresh_hum = 60
lower_thresh_hum = 5

#Sliding window setting (depends on the data collection cycle)
#in this case, data collection cyle is 1 minute
batch_size = 60 # 60 = 1 hour
train_number = 1440 # 1440 = 1 day

def train(): #For retraining model & overwriting model
    global arr_sensor_temp, arr_sensor_hum

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_temp = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_hum = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_sensor_temp = arr_sensor_temp.reshape(-1,1)
    arr_sensor_hum = arr_sensor_hum.reshape(-1,1)

    #model training
    model_temp.fit(arr_sensor_temp)
    model_hum.fit(arr_sensor_hum)

    #save model
    dump(model_temp, 'model\model_temp2.joblib')
    dump(model_hum, 'model\model_hum2.joblib')

def post_process(message):
    global arr_sensor_temp, arr_sensor_hum
    global counter
    global model_temp, model_hum

    sensor_type = message['data']['dtype']
    if sensor_type == 'thtd':
        sensor_temp = np.array([message['data']['val0']]).T
        sensor_hum = np.array([message['data']['val1']]).T

        #input stream data to the window
        arr_sensor_temp = np.append(arr_sensor_temp,sensor_temp)
        arr_sensor_hum = np.append(arr_sensor_hum,sensor_hum)

        if counter == 1: #len(arr_sensor_temp) == 1:
            #mode 1: Using initial model
            model_temp = load('model\model_temp2.joblib')
            model_hum = load('model\model_hum2.joblib')
            counter += 1
        
        elif counter <= train_number: #len(arr_sensor_temp) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter += 1
        
        elif counter == (train_number + 1) : #len(arr_sensor_temp) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter += 1
            thread.join()
        
        elif counter == (train_number+2): #len(arr_sensor_temp) == (train_number+2):
            #mode 4: load retrain model
            model_temp = load('model\model_temp2.joblib')
            model_hum = load('model\model_hum2.joblib')
            counter += 1

        elif counter <= (train_number + batch_size): #len(arr_sensor_temp) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_temp =  arr_sensor_temp[-(2*train_number+batch_size):] #[-train_number:]
            arr_sensor_hum =  arr_sensor_hum[-(2*train_number+batch_size):] #[-train_number:]
            counter = (train_number+1)

        #preprocess the data for anomaly detection
        newsensor_temp = sensor_temp.reshape(1,-1)
        newsensor_hum = sensor_hum.reshape(1,-1)

        #anomaly detection / Isolation Forest Prediction
        anomaly_score_temp =  model_temp.decision_function(newsensor_temp)
        anomaly_sensor_temp = model_temp.predict(newsensor_temp)

        anomaly_score_hum =  model_hum.decision_function(newsensor_hum)
        anomaly_sensor_hum = model_hum.predict(newsensor_hum)

        #clustering between normal & abnormal
        if anomaly_score_temp >= -0.15 and float(sensor_temp[0]) > lower_thresh_temp and float(sensor_temp[0]) < upper_thresh_temp : #normal condition
            sensor_temp_status = 'normal'
        else: #abnormal condition
            sensor_temp_status = 'abnormal'

        if anomaly_score_hum >= -0.15 and float(sensor_hum[0]) > lower_thresh_hum and float(sensor_hum[0]) < upper_thresh_hum: #normal condition
            sensor_hum_status = 'normal'
        else: #abnormal condition
            sensor_hum_status = 'abnormal'

        #Store the data in order to send it back to IoT.own
        changedata = {}
        changedata['sensor_temp_status'] = sensor_temp_status
        changedata['sensor_temp'] = float(sensor_temp[0])
        changedata['anomaly_score_temp'] = round(float(anomaly_score_temp[0]),2)

        changedata['sensor_hum_status'] = sensor_hum_status
        changedata['sensor_hum'] = float(sensor_hum[0])
        changedata['anomaly_score_hum'] = round(float(anomaly_score_hum[0]),2)

    message['data'] = changedata
    return message


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token]")
        exit(1)
    arr_sensor_temp = np.array([[]])
    arr_sensor_hum = np.array([[]])
    postproc_name = 'post_process name'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    postprocess(url,postproc_name,post_process, username, password)
    #pyiotown.post.postprocess(url,postproc_name,post_process, username, password)
