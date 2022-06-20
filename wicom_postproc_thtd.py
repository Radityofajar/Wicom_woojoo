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
    global arr_hum1, arr_temp1
    global outlier_fraction
    global sensorID

    #model setting
    estimator = 100
    samples = 500
    randstate = 42

    #model initialization
    model_hum1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)
    model_temp1 = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_hum1 = arr_hum1.reshape(-1,1)
    arr_temp1 = arr_temp1.reshape(-1,1)

    #model training
    model_hum1.fit(arr_hum1)
    model_temp1.fit(arr_temp1)

    #filename
    var1 = 'model\model_'
    var_hum = '_hum1.joblib'
    var_temp = '_temp1.joblib'
    filename_hum_model = var1 + sensorID + var_hum
    filename_temp_model = var1 + sensorID + var_temp

    #save/overwrite model
    dump(model_hum1, filename_hum_model)
    dump(model_temp1, filename_temp_model)

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
    global arr_hum1, arr_temp1
    global counter
    global model_hum1, model_temp1
    global threshold_hum1_lower, threshold_hum1_higher
    global threshold_temp1_lower, threshold_temp1_higher
    global batch_size, train_number
    global anomaly_threshVal0, anomaly_threshVal1
    global sensorID

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'thtd':
        sensor_temp = np.array([message['val0']]).T
        sensor_hum = np.array([message['val1']]).T

        #input stream data to the window
        arr_temp1 = np.append(arr_temp1, sensor_temp)
        arr_hum1 = np.append(arr_hum1, sensor_hum)

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_hum = '_hum1.joblib'
                var_temp = '_temp1.joblib'
                filename_hum_model = var1 + sensorID + var_hum
                filename_temp_model = var1 + sensorID + var_temp
                model_temp1 = load(filename_temp_model)
                model_hum1 = load(filename_hum_model)
            except: #if there is no spesificied model
                #filename
                filename_temp_model = 'model\model_temp1.joblib'
                filename_hum_model = 'model\model_hum1.joblib'
                model_temp1 = load(filename_temp_model)
                model_hum1 = load(filename_hum_model)
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
            var_hum = '_hum1.joblib'
            var_temp = '_temp1.joblib'
            filename_hum_model = var1 + sensorID + var_hum
            filename_temp_model = var1 + sensorID + var_temp

            #load model
            model_temp1 = load(filename_temp_model)
            model_hum1 = load(filename_hum_model)
            #print('model loaded')
            counter += 1

        elif counter <= (train_number + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            arr_temp1 = arr_temp1[-(2*train_number+batch_size):]
            arr_hum1 = arr_hum1[-(2*train_number+batch_size):]
            counter = (train_number+1)

        #preprocess the data for anomaly detection
        sensor_temp_reshape = sensor_temp.reshape(1,-1)
        sensor_hum_reshape = sensor_hum.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_temp = model_temp1.decision_function(sensor_temp_reshape)
        anomaly_score_hum = model_hum1.decision_function(sensor_hum_reshape)

        #print(anomaly_score_temp)
        #print(sensor_temp[0])
        #print(anomaly_score_hum)
        #print(sensor_hum[0])

        #clustering between normal & abnormal
        if anomaly_score_temp >= anomaly_threshVal0 and float(sensor_temp[0]) > threshold_temp1_lower and float(sensor_temp[0]) < threshold_temp1_higher:
            #normal condition
            sensor_temp_status = 'normal'
        else:
            #abnormal condition
            sensor_temp_status = 'abnormal'

        if anomaly_score_hum >= anomaly_threshVal1 and float(sensor_hum[0]) > threshold_hum1_lower and float(sensor_hum[0]) < threshold_hum1_higher:
            #normal condition
            sensor_hum_status = 'normal'
        else:
            #abnormal condition
            sensor_hum_status = 'abnormal'

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['result_temp'] = sensor_temp_status
        changedata['result_hum'] = sensor_hum_status
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 14:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal0] [up_threshVal0] [low_threshVal1] [up_thresVal1] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal0] [anomaly_threshVal1] [sensor ID]")
        exit(1)

    #initialize array
    arr_temp1 = np.array([[]])
    arr_hum1 = np.array([[]])

    #IoT.own setting
    postproc_name = 'thtd'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #thresholding value
    threshold_temp1_lower = int(sys.argv[4])
    threshold_temp1_higher = int(sys.argv[5])
    threshold_hum1_lower = int(sys.argv[6])
    threshold_hum1_higher = int(sys.argv[7])

    #sliding window setting
    batch_size = int(sys.argv[8])
    train_number = int(sys.argv[9])

    #model parameter
    outlier_fraction = float(sys.argv[10])

    #clustering setting
    anomaly_threshVal0 = float(sys.argv[11])
    anomaly_threshVal1 = float(sys.argv[12])

    #model name for spesific sensor
    sensorID = sys.argv[13]
    
    postprocess(url,postproc_name,post_process, username, password)