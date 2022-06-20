from pyiotown import post
from pyiotown_wicom import postprocess
import json
import numpy as np
from joblib import load, dump
from sklearn.ensemble import IsolationForest
import threading
import sys
import warnings
warnings.filterwarnings('ignore')

counter = 1

def train():
    global arr_hum, arr_temp
    global outlier_fraction
    global sensorID

    #model setting
    estimator = 100
    samples = 500
    randstate = 42

    #model initialization
    model_hum = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_temp = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_hum = arr_hum.reshape(-1,1)
    arr_temp = arr_temp.reshape(-1,1)

    #model training
    model_hum.fit(arr_hum)
    model_temp.fit(arr_temp)

    #filename
    var1 = 'model\model_'
    var_hum = '_hum.joblib'
    var_temp = '_temp.joblib'
    filename_hum_model = var1 + sensorID + var_hum
    filename_temp_model = var1 + sensorID + var_temp

    #save/overwrite model
    dump(model_hum, filename_hum_model)
    dump(model_temp, filename_temp_model)

    print('update the model')

def receive_data(rawdata):
    raw_data = rawdata['data']
    #take the data
    print(raw_data)
    test = len(raw_data.split())
    if raw_data == 'Missing':
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor)))
        #change to json format
        sensor = pythonObj['data']
    return sensor

def post_process(rawdata):
    global arr_hum, arr_temp
    global counter
    global model_hum, model_temp
    global threshold_hum_lower, threshold_hum_higher
    global threshold_temp_lower, threshold_temp_higher
    global door_thresh
    global batch_size, train_number
    global anomaly_threshVal0, anomaly_threshVal1
    global sensorID

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'tdhd':
        sensor_temp = np.array([message['val0']]).T
        sensor_hum = np.array([message['val1']]).T
        sensor_door = message['val2']

        #input stream data to the window
        arr_temp = np.append(arr_temp, sensor_temp)
        arr_hum = np.append(arr_hum, sensor_hum)

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_hum = '_hum.joblib'
                var_temp = '_temp.joblib'
                filename_hum_model = var1 + sensorID + var_hum
                filename_temp_model = var1 + sensorID + var_temp
                model_temp = load(filename_temp_model)
                model_hum = load(filename_hum_model)
            except: #if there is no spesificied model
                #filename
                filename_temp_model = 'model\model_temp1.joblib'
                filename_hum_model = 'model\model_hum1.joblib'
                model_temp = load(filename_temp_model)
                model_hum = load(filename_hum_model)
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_temp_model)
                print(filename_hum_model)
                counter += 1

        elif counter <= train_number:
            #mode2: Keep using initial model until the data stored in array
            counter += 1

        elif counter == (train_number + 1):
            #mode 3: retrain the model
            thread = threading.Thread(target=train)
            if thread.is_alive():
                print('thread still running')
            else:
                print('thread is starting')
                thread.start()
            counter += 1
            thread.join()
        
        elif counter == (train_number+2):
            #model 4: load retrain model

            #filename
            var1 = 'model\model_'
            var_hum = '_hum.joblib'
            var_temp = '_temp.joblib'
            filename_hum_model = var1 + sensorID + var_hum
            filename_temp_model = var1 + sensorID + var_temp

            #load model
            model_temp = load(filename_temp_model)
            model_hum = load(filename_hum_model)
            #print('model loaded')
            counter += 1

        elif counter <= (train_number + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            arr_temp = arr_temp[-(2*train_number+batch_size):]
            arr_hum = arr_hum[-(2*train_number+batch_size):]
            counter = (train_number+1)

        #preprocess the data for anomaly detection
        sensor_temp_reshape = sensor_temp.reshape(1,-1)
        sensor_hum_reshape = sensor_hum.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_temp = model_temp.decision_function(sensor_temp_reshape)
        anomaly_score_hum = model_hum.decision_function(sensor_hum_reshape)

        #print(anomaly_score_temp)
        #print(sensor_temp[0])
        #print(anomaly_score_hum)
        #print(sensor_hum[0])

        #clustering between normal & abnormal
        if anomaly_score_temp >= anomaly_threshVal0 and float(sensor_temp[0]) > threshold_temp_lower and float(sensor_temp[0]) < threshold_temp_higher:
            #normal condition
            sensor_temp_status = 'normal'
        else:
            #abnormal condition
            sensor_temp_status = 'abnormal'

        if anomaly_score_hum >= anomaly_threshVal1 and float(sensor_hum[0]) > threshold_hum_lower and float(sensor_hum[0]) < threshold_hum_higher:
            #normal condition
            sensor_hum_status = 'normal'
        else:
            #abnormal condition
            sensor_hum_status = 'abnormal'

        if door_thresh == 'NC': #normally closed
            if float(sensor_door) == 0:
                #normal condition
                sensor_door_status = 'normal/closed'
            else:
                #abnormal condition
                sensor_door_status = 'abnormal/open'

        else: #NO / Normally Open
            if float(sensor_door) == 1:
                #normal condition
                sensor_door_status = 'normal/open'
            else:
                #abnormal condition
                sensor_door_status = 'abnormal/closed'

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['result_temp'] = sensor_temp_status
        changedata['result_hum'] = sensor_hum_status
        changedata['result_door'] = sensor_door_status
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 15:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal0] [up_threshVal0] [low_threshVal1] [up_thresVal1] [door NC/NO] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal0] [anomaly_threshVal1] [sensor ID]")
        exit(1)

    #initialize array
    arr_temp = np.array([[]])
    arr_hum = np.array([[]])

    #IoT.own setting
    postproc_name = 'tdhd'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #thresholding value
    threshold_temp_lower = int(sys.argv[4])
    threshold_temp_higher = int(sys.argv[5])
    threshold_hum_lower = int(sys.argv[6])
    threshold_hum_higher = int(sys.argv[7])
    door_thresh = sys.argv[8]

    #sliding window setting
    batch_size = int(sys.argv[9])
    train_number = int(sys.argv[10])

    #model parameter
    outlier_fraction = float(sys.argv[11])

    #clustering setting
    anomaly_threshVal0 = float(sys.argv[12])
    anomaly_threshVal1 = float(sys.argv[13])

    #model name for spesific sensor
    sensorID = sys.argv[14]
    
    postprocess(url,postproc_name,post_process, username, password)