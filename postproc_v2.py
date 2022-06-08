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

counter_thtd = 1
counter_fire = 1
counter_tdhd = 1
counter_wlvl = 1
counter_wlak = 1

#Thresholding value
upper_thresh_temp = 40
lower_thresh_temp = 5

upper_thresh_hum = 60
lower_thresh_hum = 5

upper_thresh_tempfire = 40
lower_thresh_tempfire = 5

upper_thresh_temp2 = 40
lower_thresh_temp2 = 5

upper_thresh_hum2 = 60
lower_thresh_hum2 = 5

upper_thresh_wlvl = 3500
lower_thresh_wlvl = 250

#Sliding window setting (depends on the data collection cycle)
#in this case, data collection cyle is 1 minute
batch_size = 60 # 60 = 1 hour
train_number = 1440 # 1440 = 1 day

def train_thtd(): #For retraining model & overwriting model thtd sensor
    global arr_sensor_temp1, arr_sensor_hum1

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_temp = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_hum = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_sensor_temp1 = arr_sensor_temp1.reshape(-1,1)
    arr_sensor_hum1 = arr_sensor_hum1.reshape(-1,1)

    #model training
    model_temp.fit(arr_sensor_temp1)
    model_hum.fit(arr_sensor_hum1)

    #save model
    dump(model_temp, 'model\model_temp2.joblib')
    dump(model_hum, 'model\model_hum2.joblib')

def train_fire(): #For retraining model & overwriting model fire sensor
    global arr_sensor_tempfire, arr_sensor_fire

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_tempfire = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_fire = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_sensor_tempfire = arr_sensor_tempfire.reshape(-1,1)
    arr_sensor_fire = arr_sensor_fire.reshape(-1,1)

    #model training
    model_tempfire.fit(arr_sensor_tempfire)
    model_fire.fit(arr_sensor_fire)

    #save model
    dump(model_tempfire, 'model\model_temp3.joblib')
    dump(model_fire, 'model\model_fire.joblib')

def train_tdhd(): #For retraining model & overwriting model tdhd sensor
    global arr_sensor_temp2, arr_sensor_hum2, arr_sensor_door

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_temp2 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_hum2 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_door = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=0.01)
    
    #data preprocess
    arr_sensor_temp2 = arr_sensor_temp2.reshape(-1,1)
    arr_sensor_hum2 = arr_sensor_hum2.reshape(-1,1)
    arr_sensor_door = arr_sensor_door.reshape(-1,1)

    #model training
    model_temp2.fit(arr_sensor_temp2)
    model_hum2.fit(arr_sensor_hum2)
    model_door.fit(arr_sensor_door)

    #save model
    dump(model_temp2, 'model\model_temp2.joblib')
    dump(model_hum2, 'model\model_hum1.joblib')
    dump(model_door, 'model\model_door.joblib')

def train_wlvl(): #For retraining model & overwriting model wlvl sensor
    global arr_sensor_waterlevel

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_wlvl = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    
    #data preprocess
    arr_sensor_wlvl = arr_sensor_wlvl.reshape(-1,1)

    #model training
    model_wlvl.fit(arr_sensor_wlvl)

    #save model
    dump(model_wlvl, 'model\model_waterlevel.joblib')

def train_wlak(): #For retraining model & overwriting model wlak sensor
    global arr_sensor_waterleak

    #model initialization
    estimator = 100
    samples = 500
    randstate = 42
    outlier_fraction = 0.01
    model_wlak = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    
    #data preprocess
    arr_sensor_waterleak = arr_sensor_waterleak.reshape(-1,1)

    #model training
    model_wlak.fit(arr_sensor_waterleak)

    #save model
    dump(model_wlak, 'model\model_waterleak.joblib')


def post_process(message):
    print(message['data'])
    global arr_sensor_temp1, arr_sensor_hum1
    global arr_sensor_tempfire, arr_sensor_fire
    global arr_sensor_temp2, arr_sensor_hum2, arr_sensor_door
    global arr_sensor_waterlevel, arr_sensor_waterleak
    global counter_thtd, counter_fire, counter_tdhd
    global counter_wlak, counter_wlvl
    global model_temp,model_hum
    global model_temp2,model_hum2, model_door
    global model_tempfire,model_fire
    global model_wlak, model_wlvl

    sensor_type = message['data']['dtype']
    if sensor_type == 'thtd':
        sensor_temp = np.array([message['data']['val0']]).T
        sensor_hum = np.array([message['data']['val1']]).T

        #input stream data to the window
        arr_sensor_temp1 = np.append(arr_sensor_temp1,sensor_temp)
        arr_sensor_hum1 = np.append(arr_sensor_hum1,sensor_hum)

        if counter_thtd == 1: #len(arr_sensor_temp1) == 1:
            #mode 1: Using initial model
            model_temp = load('model\model_temp2.joblib')
            model_hum = load('model\model_hum2.joblib')
            counter_thtd += 1
        
        elif counter_thtd <= train_number: #len(arr_sensor_temp1) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter_thtd += 1
        
        elif counter_thtd == (train_number + 1) : #len(arr_sensor_temp1) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train_thtd)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter_thtd += 1
            thread.join()
        
        elif counter_thtd == (train_number+2): #len(arr_sensor_temp1) == (train_number+2):
            #mode 4: load retrain model
            model_temp = load('model\model_temp2.joblib')
            model_hum = load('model\model_hum2.joblib')
            counter_thtd += 1

        elif counter_thtd <= (train_number + batch_size): #len(arr_sensor_temp1) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter_thtd += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_temp1 =  arr_sensor_temp1[-(2*train_number+batch_size):] #[-train_number:]
            arr_sensor_hum1 =  arr_sensor_hum1[-(2*train_number+batch_size):] #[-train_number:]
            counter_thtd = (train_number+1)

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

    elif sensor_type == 'fire':
        sensor_fire = np.array([message['data']['val0']]).T
        sensor_tempfire = np.array([message['data']['val1']]).T

        #input stream data to the window
        arr_sensor_fire = np.append(arr_sensor_fire,sensor_fire)
        arr_sensor_tempfire = np.append(arr_sensor_tempfire,sensor_tempfire)

        if counter_fire == 1: #len(arr_sensor_fire) == 1:
            #mode 1: Using initial model
            model_fire = load('model\model_fire.joblib')
            model_tempfire = load('model\model_temp3.joblib')
            counter_fire += 1
        
        elif counter_fire <= train_number: #len(arr_sensor_fire) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter_fire += 1
        
        elif counter_fire == (train_number + 1) : #len(arr_sensor_fire) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train_fire)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter_fire += 1
            thread.join()
        
        elif counter_fire == (train_number+2): #len(arr_sensor_fire) == (train_number+2):
            #mode 4: load retrain model
            model_fire = load('model\model_fire.joblib')
            model_tempfire = load('model\model_temp3.joblib')
            counter_fire += 1

        elif counter_fire <= (train_number + batch_size): #len(arr_sensor_fire) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter_fire += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_fire =  arr_sensor_fire[-(2*train_number+batch_size):] #[-train_number:]
            arr_sensor_tempfire =  arr_sensor_tempfire[-(2*train_number+batch_size):] #[-train_number:]
            counter_fire = (train_number+1)

        #preprocess the data for anomaly detection
        newsensor_fire = sensor_fire.reshape(1,-1)
        newsensor_tempfire = sensor_tempfire.reshape(1,-1)

        #anomaly detection / Isolation Forest Prediction
        anomaly_score_tempfire =  model_tempfire.decision_function(newsensor_tempfire)
        anomaly_sensor_tempfire = model_tempfire.predict(newsensor_tempfire)

        anomaly_score_fire =  model_fire.decision_function(newsensor_fire)
        anomaly_sensor_fire = model_fire.predict(newsensor_fire)

        #clustering between normal & abnormal
        if anomaly_score_tempfire >= -0.15 and float(sensor_tempfire[0]) > lower_thresh_tempfire and float(sensor_tempfire[0]) < upper_thresh_tempfire : #normal condition
            sensor_tempfire_status = 'normal'
        else: #abnormal condition
            sensor_tempfire_status = 'abnormal'

        if anomaly_score_fire >= 0 and float(sensor_fire[0]) == 0: #normal condition
            sensor_fire_status = 'normal'
        else: #abnormal condition
            sensor_fire_status = 'abnormal'

        #Store the data in order to send it back to IoT.own
        changedata = {}
        changedata['sensor_tempfire_status'] = sensor_tempfire_status
        changedata['sensor_tempfire'] = float(sensor_tempfire[0])
        changedata['anomaly_score_tempfire'] = round(float(anomaly_score_tempfire[0]),2)

        changedata['sensor_fire_status'] = sensor_fire_status
        changedata['sensor_fire'] = float(sensor_fire[0])
        changedata['anomaly_score_fire'] = round(float(anomaly_score_fire[0]),2)

    elif sensor_type == 'tdhd':
        sensor_temp2 = np.array([message['data']['val0']]).T
        sensor_hum2 = np.array([message['data']['val1']]).T
        sensor_door = np.array([message['data']['val2']]).T

        #input stream data to the window
        arr_sensor_temp2 = np.append(arr_sensor_temp2,sensor_temp2)
        arr_sensor_hum2 = np.append(arr_sensor_hum2,sensor_hum2)
        arr_sensor_door = np.append(arr_sensor_door,sensor_door)

        if counter_tdhd == 1: #len(arr_sensor_temp2) == 1:
            #mode 1: Using initial model
            model_temp2 = load('model\model_temp2.joblib')
            model_hum2 = load('model\model_hum1.joblib')
            model_door = load('model\model_door.joblib')
            counter_tdhd += 1
        
        elif counter_tdhd <= train_number: #len(arr_sensor_temp2) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter_tdhd += 1
        
        elif counter_tdhd == (train_number + 1) : #len(arr_sensor_temp2) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train_tdhd)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter_tdhd += 1
            thread.join()
        
        elif counter_tdhd == (train_number+2): #len(arr_sensor_temp2) == (train_number+2):
            #mode 4: load retrain model
            model_temp2 = load('model\model_temp2.joblib')
            model_hum2 = load('model\model_hum1.joblib')
            model_door = load('model\model_door.joblib')
            counter_tdhd += 1

        elif counter_tdhd <= (train_number + batch_size): #len(arr_sensor_temp2) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter_tdhd += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_temp2 =  arr_sensor_temp2[-(2*train_number+batch_size):] #[-train_number:]
            arr_sensor_hum2 =  arr_sensor_hum2[-(2*train_number+batch_size):] #[-train_number:]
            arr_sensor_door =  arr_sensor_door[-(2*train_number+batch_size):] #[-train_number:]
            counter_tdhd = (train_number+1)

        #preprocess the data for anomaly detection
        newsensor_temp2 = sensor_temp2.reshape(1,-1)
        newsensor_hum2 = sensor_hum2.reshape(1,-1)
        newsensor_door = sensor_door.reshape(1,-1)

        #anomaly detection / Isolation Forest Prediction
        anomaly_score_temp2 =  model_temp2.decision_function(newsensor_temp2)
        anomaly_sensor_temp2 = model_temp2.predict(newsensor_temp2)

        anomaly_score_hum2 =  model_hum2.decision_function(newsensor_hum2)
        anomaly_sensor_hum2 = model_hum2.predict(newsensor_hum2)

        anomaly_score_door =  model_door.decision_function(newsensor_door)
        anomaly_sensor_door = model_door.predict(newsensor_door)

        #clustering between normal & abnormal
        if anomaly_score_temp2 >= -0.15 and float(sensor_temp2[0]) > lower_thresh_temp2 and float(sensor_temp2[0]) < upper_thresh_temp2 : #normal condition
            sensor_temp2_status = 'normal'
        else: #abnormal condition
            sensor_temp2_status = 'abnormal'

        if anomaly_score_hum2 >= -0.15 and float(sensor_hum2[0]) > lower_thresh_hum2 and float(sensor_hum2[0]) < upper_thresh_hum2: #normal condition
            sensor_hum2_status = 'normal'
        else: #abnormal condition
            sensor_hum2_status = 'abnormal'
        
        if anomaly_score_door >= 0 and float(sensor_door[0]) == 0: #thresholding for binary sensor
            sensor_door_status = 'normal'
        else: #abnormal condition
            sensor_door_status = 'abnormal'

        #Store the data in order to send it back to IoT.own
        changedata = {}
        changedata['sensor_temp2_status'] = sensor_temp2_status
        changedata['sensor_temp2'] = float(sensor_temp2[0])
        changedata['anomaly_score_temp2'] = round(float(anomaly_score_temp2[0]),2)

        changedata['sensor_hum2_status'] = sensor_hum2_status
        changedata['sensor_hum2'] = float(sensor_hum2[0])
        changedata['anomaly_score_hum2'] = round(float(anomaly_score_hum2[0]),2)

        changedata['sensor_door_status'] = sensor_door_status
        changedata['sensor_door'] = float(sensor_door[0])
        changedata['anomaly_score_door'] = round(float(anomaly_score_door[0]),2)


    elif sensor_type == 'wlvl':
        sensor_wlvl = np.array([message['data']['val0']]).T

        #input stream data to the window
        arr_sensor_waterlevel = np.append(arr_sensor_waterlevel,sensor_wlvl)

        if counter_wlvl == 1: #len(arr_sensor_waterlevel) == 1:
            #mode 1: Using initial model
            model_wlvl = load('model\model_waterlevel.joblib')
            counter_wlvl += 1
        
        elif counter_wlvl <= train_number: #len(arr_sensor_waterlevel) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter_wlvl += 1
        
        elif counter_wlvl == (train_number + 1) : #len(arr_sensor_waterlevel) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train_wlvl)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter_wlvl += 1
            thread.join()
        
        elif counter_wlvl == (train_number+2): #len(arr_sensor_waterlevel) == (train_number+2):
            #mode 4: load retrain model
            model_wlvl = load('model\model_waterlevel.joblib')
            counter_wlvl += 1

        elif counter_wlvl < (train_number + batch_size): #len(arr_sensor_waterlevel) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter_wlvl += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_waterlevel =  arr_sensor_waterlevel[-(2*train_number+batch_size):] #[-train_number:]
            counter_wlvl = (train_number+1)

        #preprocess the data for anomaly detection
        newsensor_wlvl = sensor_wlvl.reshape(1,-1)

        #anomaly detection / Isolation Forest Prediction
        anomaly_score_wlvl =  model_wlvl.decision_function(newsensor_wlvl)
        anomaly_sensor_wlvl = model_wlvl.predict(newsensor_wlvl)

        #clustering between normal & abnormal
        if anomaly_score_wlvl > 0 and float(sensor_wlvl[0]) > lower_thresh_wlvl and float(sensor_wlvl[0]) < upper_thresh_wlvl:
            sensor_wlvl_status = 'normal'
        else:
            sensor_wlvl_status = 'abnormal'

        #Store the data in order to send it back to IoT.own
        changedata = {}
        changedata['sensor_waterlevel_status'] = sensor_wlvl_status
        changedata['sensor_waterlevel'] = float(sensor_wlvl[0])
        changedata['anomaly_score_waterlevel'] = round(float(anomaly_score_wlvl[0]),2)

    elif sensor_type == 'wlak':
        sensor_wlak = np.array([message['data']['val0']]).T

        #input stream data to the window
        arr_sensor_waterleak = np.append(arr_sensor_waterleak,sensor_wlak)

        if counter_wlak == 1: #len(arr_sensor_waterleak) == 1:
            #mode 1: Using initial model
            model_wlak = load('model\model_waterleak.joblib')
            counter_wlak += 1
        
        elif counter_wlak <= train_number: #len(arr_sensor_waterleak) <= train_number:
            #mode 2: Keep using initial model until the data stored in array(window) is enough
            counter_wlak += 1
        
        elif counter_wlak == (train_number + 1) : #len(arr_sensor_waterleak) == (train_number+1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train_wlak)
            if thread.is_alive():
                print('thread still running')          
            else:
                print('thread is starting')
                thread.start()
            counter_wlak += 1
            thread.join()
        
        elif counter_wlak == (train_number+2): #len(arr_sensor_waterleak) == (train_number+2):
            #mode 4: load retrain model
            model_wlak = load('model\model_waterleak.joblib')
            counter_wlak += 1

        elif counter_wlak < (train_number + batch_size): #len(arr_sensor_waterleak) <= (train_number + batch_size):
            #mode 5: sliding window method
            counter_wlak += 1

        else:
            #optimize the array size of sliding window
            arr_sensor_waterleak =  arr_sensor_waterleak[-(2*train_number+batch_size):] #[-train_number:]
            counter_wlak = (train_number+1)

        #preprocess the data for anomaly detection
        newsensor_wlak = sensor_wlak.reshape(1,-1)

        #anomaly detection / Isolation Forest Prediction
        anomaly_score_wlak =  model_wlak.decision_function(newsensor_wlak)
        anomaly_sensor_wlak = model_wlak.predict(newsensor_wlak)

        #clustering between normal & abnormal
        if anomaly_score_wlak >= 0 and float(sensor_wlak[0]) == 0: #thresholding for binary sensor
            sensor_wlak_status = 'normal'
        else: #abnormal condition
            sensor_wlak_status = 'abnormal'

        #Store the data in order to send it back to IoT.own
        changedata = {}
        changedata['sensor_waterleak_status'] = sensor_wlak_status
        changedata['sensor_waterleak'] = float(sensor_wlak[0])
        changedata['anomaly_score_waterleak'] = round(float(anomaly_score_wlak[0]),2)

    else:
        print('data not supported: train the initial model')
        changedata = message['data']

    message['data'] = changedata
    print(changedata)
    return message


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token]")
        exit(1)
    arr_sensor_temp1 = np.array([[]]) #thtd
    arr_sensor_hum1 = np.array([[]]) #thtd
    arr_sensor_temp2 = np.array([[]]) #tdhd
    arr_sensor_hum2 = np.array([[]]) #tdhd
    arr_sensor_door = np.array([[]]) #tdhd
    arr_sensor_fire = np.array([[]]) #fire
    arr_sensor_tempfire = np.array([[]]) #fire
    arr_sensor_waterlevel = np.array([[]]) #wlvl
    arr_sensor_waterleak = np.array([[]]) #wlak
    postproc_name = 'post_process name'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]
    postprocess(url,postproc_name,post_process, username, password)
    #pyiotown.post.postprocess(url,postproc_name,post_process, username, password)
