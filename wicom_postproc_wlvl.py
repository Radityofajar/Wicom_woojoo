from pyiotown_wicom import postprocess
import json
import numpy as np
from joblib import load, dump
from sklearn.ensemble import IsolationForest
from collections import Counter
import threading
import sys
import warnings
warnings.filterwarnings('ignore')

counter = 1

def train(sensor_nid, outlier_fraction):
    global outlier_fraction_param
    global nid_library

    #model setting
    estimator = 100
    samples = 500
    randstate = 42

    #outlier parameter
    if outlier_fraction == 0:
        outlier_fraction = 0.001 # 1% of contamination
    elif outlier_fraction >= outlier_fraction_param:
        outlier_fraction = outlier_fraction_param
    else:
        outlier_fraction = outlier_fraction

    #model initialization
    model_waterlevel = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    nid_library[sensor_nid] = nid_library[sensor_nid].reshape(-1,1)

    #model training
    model_waterlevel.fit(nid_library[sensor_nid])

    #filename
    var1 = 'model\model_'
    var_wlvl= '_waterlevel.joblib'
    filename_wlvl_model = var1 + sensor_nid + var_wlvl

    #save/overwrite model
    dump(model_waterlevel, filename_wlvl_model)

    print('update the model')

def receive_data(rawdata):
    raw_data = rawdata['data']
    #take the data
    print(raw_data)
    # raw_data --> 'POST /api/v1.0/data HTTP/1.1\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: 120\r\nToken: 079da3dd77569523551c9ddd8c8e57c4f4ee71bea5e848d68785b06545d098ac\r\n\r\n{"type": "2","nid": "WS000001FFFF123456", "data": {"dtype":"wlvl", "nid":"WS000001FFFF123456", "val0":727}}\r\n\r\n'
    test = len(raw_data.split())
    if raw_data == 'Missing':
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        #datasensor --> ['{"type":', '"2","nid":', '"WS000001FFFF123456",', '"data":', '{"dtype":"wlvl",', '"nid":"WS000001FFFF123456",', '"val0":719}}']
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor)))
        #change to json format
        sensor = pythonObj['data']
        #sensor --> {"dtype":"wlvl","nid":"WS000001FFFF123456","val0":25.6,"val1":57.8}
    return sensor

def post_process(rawdata):
    global counter
    global model_waterlevel
    global threshold_wlvl_lower, threshold_wlvl_higher
    global batch_size, train_number
    global anomaly_threshVal0, anomaly_threshVal0_param
    global nid_library

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)

    #check sensor type
    sensor_type = message['dtype']
    if sensor_type == 'wlvl':
        sensor_wlvl = np.array([float(message['val0'])]).T

        #check sensor nid
        sensor_nid = message['nid']
        if sensor_nid not in nid_library.keys(): #check wheteher nid is new or not
            nid_library[sensor_nid] = np.array([[]]) #make a new array for new nid
            nid_library['anomaly_score'] = np.array([[]]) #make a new array for new nid
            nid_library['anomaly_status'] = np.array([[]]) #make a new array for new nid
        
        #input stream data to the window
        nid_library[sensor_nid] = np.append(nid_library[sensor_nid], sensor_wlvl)
        print(nid_library[sensor_nid])

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_wlvl = '_waterlevel.joblib'
                filename_wlvl_model = var1 + sensor_nid + var_wlvl
                model_waterlevel = load(filename_wlvl_model)

            except: #if there is no spesificied model
                #filename
                filename_wlvl_model = 'model\model_waterlevel.joblib'
                model_waterlevel = load(filename_wlvl_model) #load model
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_wlvl_model)
                anomaly_threshVal0 = 0.0
                counter += 1

        elif counter <= batch_size:
            #mode2: Keep using initial model until the data stored in array
            counter += 1

        elif counter == (batch_size + 1):
            #mode 3: retrain the model

            #calculate the outlier_fraction
            outlier = Counter(nid_library['anomaly_status'])#wlvl
            outlier_fraction = outlier['abnormal'] / len(nid_library['anomaly_status'])

            #multithreading
            thread = threading.Thread(target=train, args=(sensor_nid,outlier_fraction))
            if thread.is_alive():
                print('thread still running')
            else:
                print('thread is starting')
                thread.start()
            counter += 1
            thread.join()
        
        elif counter == (batch_size+2):
            #model 4: load retrain model

            #filename
            var1 = 'model\model_'
            var_wlvl = '_waterlevel.joblib'
            filename_wlvl_model = var1 + sensor_nid + var_wlvl

            #load model
            model_waterlevel = load(filename_wlvl_model)
            #print('model loaded')

            #calculate the anomaly score threshold for water level
            anomaly_score_wlvl_mean = nid_library['anomaly_score'].mean()
            anomaly_score_wlvl_std = nid_library['anomaly_score'].std()
            anomaly_score_wlvl_cal = anomaly_score_wlvl_mean - (anomaly_score_wlvl_std * anomaly_threshVal0_param)

            if anomaly_score_wlvl_cal <= -0.15:
                anomaly_threshVal0 = -0.15
            elif anomaly_score_wlvl_cal >= 0.02:
                anomaly_threshVal0 = 0.02
            else:
                anomaly_threshVal0 = anomaly_score_wlvl_cal

            counter += 1

        elif counter <= (batch_size + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window
            nid_library[sensor_nid] = nid_library[sensor_nid][-(train_number+batch_size):]
            nid_library['anomaly_score'] = nid_library[sensor_nid][-(train_number+batch_size):]
            nid_library['anomaly_status'] = nid_library[sensor_nid][-(train_number+batch_size):]
            counter = (batch_size+1)

        #preprocess the data for anomaly detection
        sensor_wlvl_reshape = sensor_wlvl.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_wlvl = model_waterlevel.decision_function(sensor_wlvl_reshape)

        print(anomaly_score_wlvl)
        print(sensor_wlvl[0])

        print(anomaly_threshVal0)

        #clustering between normal & abnormal

        #Water level sensor
        if anomaly_score_wlvl >= anomaly_threshVal0:
            if float(sensor_wlvl[0]) > threshold_wlvl_lower:
                if float(sensor_wlvl[0]) < threshold_wlvl_higher:
                    #normal condition
                    sensor_wlvl_status = 'normal'
                else:
                    sensor_wlvl_status = 'abnormal/too high'
            else:
                sensor_wlvl_status = 'abnormal/too low'
        else:
            #abnormal condition
            sensor_wlvl_status = 'abnormal'

        #append value of anomaly score and sensor status
        nid_library['anomaly_score'] = np.append(nid_library['anomaly_score'], float(anomaly_score_wlvl))
        nid_library['anomaly_status'] = np.append(nid_library['anomaly_status'], sensor_wlvl_status) 

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['val0'] = float(message['val0'])
        changedata['result_v0'] = sensor_wlvl_status
        changedata['anomaly_score'] = float(anomaly_score_wlvl)
        changedata['anomaly_score_threshold'] = float(anomaly_threshVal0)
        rawdata['data'] = changedata
        print(rawdata)

        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 10:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal0] [up_thresVal0] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal0]")
        exit(1)

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
    outlier_fraction_param = float(sys.argv[8])

    #clustering setting
    anomaly_threshVal0_param = float(sys.argv[9])

    #initialize nid library
    nid_library = {}
    
    postprocess(url,postproc_name,post_process, username, password)
