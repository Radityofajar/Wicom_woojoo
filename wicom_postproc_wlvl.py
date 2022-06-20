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
    global arr_wlvl
    global outlier_fraction
    global sensorID

    #model setting
    estimator = 100
    samples = 500
    randstate = 42

    #model initialization
    model_waterlevel = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    arr_wlvl = arr_wlvl.reshape(-1,1)

    #model training
    model_waterlevel.fit(arr_wlvl)

    #filename
    var1 = 'model\model_'
    var_wlvl= '_waterlevel.joblib'
    filename_wlvl_model = var1 + sensorID + var_wlvl

    #save/overwrite model
    dump(model_waterlevel, filename_wlvl_model)

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
    global arr_wlvl
    global counter
    global model_waterlevel
    global threshold_wlvl_lower, threshold_wlvl_higher
    global batch_size, train_number
    global anomaly_threshVal0
    global sensorID

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'wlvl':
        sensor_wlvl = np.array([message['val0']]).T

        #input stream data to the window
        arr_wlvl = np.append(arr_wlvl, sensor_wlvl)

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_wlvl = '_waterlevel.joblib'
                filename_wlvl_model = var1 + sensorID + var_wlvl
                model_waterlevel = load(filename_wlvl_model)

            except: #if there is no spesificied model
                #filename
                filename_wlvl_model = 'model\model_waterlevel.joblib'
                model_waterlevel = load(filename_wlvl_model)
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_wlvl_model)
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
            var_wlvl = '_waterlevel.joblib'
            filename_wlvl_model = var1 + sensorID + var_wlvl

            #load model
            model_waterlevel = load(filename_wlvl_model)
            #print('model loaded')
            counter += 1

        elif counter <= (train_number + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            arr_wlvl = arr_wlvl[-(2*train_number+batch_size):]
            counter = (train_number+1)

        #preprocess the data for anomaly detection
        sensor_wlvl_reshape = sensor_wlvl.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_wlvl = model_waterlevel.decision_function(sensor_wlvl_reshape)

        #print(anomaly_score_wlvl)
        #print(sensor_wlvl[0])

        #clustering between normal & abnormal

        #Water level sensor
        if anomaly_score_wlvl >= anomaly_threshVal0 and float(sensor_wlvl[0]) > threshold_wlvl_lower and float(sensor_wlvl[0]) < threshold_wlvl_higher:
            #normal condition
            sensor_wlvl_status = 'normal'
        else:
            #abnormal condition
            sensor_wlvl_status = 'abnormal'

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['result_wlvl'] = sensor_wlvl_status
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 11:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal0] [up_thresVal0] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal1] [sensor ID]")
        exit(1)

    #initialize array
    arr_wlvl = np.array([[]])

    #IoT.own setting
    postproc_name = 'wlvl'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #thresholding value
    threshold_wlvl_lower = int(sys.argv[4])
    threshold_wlvl_higher = int(sys.argv[5])

    #sliding window setting
    batch_size = int(sys.argv[6])
    train_number = int(sys.argv[7])

    #model parameter
    outlier_fraction = float(sys.argv[8])

    #clustering setting
    anomaly_threshVal0 = float(sys.argv[9])

    #model name for spesific sensor
    sensorID = sys.argv[10]
    
    postprocess(url,postproc_name,post_process, username, password)