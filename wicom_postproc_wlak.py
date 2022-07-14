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

def train(sensor_nid, outlier_fraction):
    global nid_library
    global outlier_fraction_param

    #model setting
    estimator = 100
    samples = 1000
    randstate = 42

    #outlier parameter
    if outlier_fraction == 0:
        outlier_fraction = 0.001 # 0.1% of contamination
    elif outlier_fraction >= outlier_fraction_param:
        outlier_fraction = outlier_fraction_param
    else:
        outlier_fraction = outlier_fraction

    model_waterleak = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction)

    #data preprocess
    data_wlak = nid_library[sensor_nid].reshape(-1,1)

    #model training
    model_waterleak.fit(data_wlak)

    #filename
    var1 = 'model\model_'
    var_wlak = '_waterleak.joblib'
    filename_wlak_model = var1 + sensor_nid + var_wlak
    #save/overwrite model
    dump(model_waterleak, filename_wlak_model)

    print('update the model')

def receive_data(rawdata):
    raw_data = rawdata['data']
    #take the data
    print(raw_data)
    # raw_data --> 'POST /api/v1.0/data HTTP/1.1\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: 120\r\nToken: 079da3dd77569523551c9ddd8c8e57c4f4ee71bea5e848d68785b06545d098ac\r\n\r\n{"type": "2","nid": "WS000001FFFF123456", "data": {"dtype":"wlak", "nid":"WS000001FFFF123456", "val0":1}}\r\n\r\n'
    test = len(raw_data.split())
    if raw_data == 'Missing':
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        #datasensor --> ['{"type":', '"2","nid":', '"WS000001FFFF123456",', '"data":', '{"dtype":"wlak",', '"nid":"WS000001FFFF123456",', '"val0":1}}']
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor)))#change to json format
        sensor = pythonObj['data']
        #sensor --> {"dtype":"wlak","nid":"WS000001FFFF123456","val0":1}

    return sensor

def post_process(rawdata):
    global counter
    global model_waterleak
    global batch_size, train_number
    global nid_library

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'wlak':
        sensor_wlak =  np.array([message['val0']]).T #take the water leak data

        #check sensor nid
        sensor_nid = message['nid']
        score_nid = 'score_' + str(sensor_nid)
        status_nid = 'status_' + str(sensor_nid)
        counter = 'counter' + str(sensor_nid)
        if sensor_nid not in nid_library.keys(): #check wheteher nid is new or not
            nid_library[sensor_nid] = np.array([[]]) #make a new array for new nid
            nid_library[score_nid] = np.array([[]]) #make a new array for new nid
            nid_library[status_nid] = np.array([[]]) #make a new array for new nid
            nid_library[counter] = 1 #set counter
            
        #input stream data to the window
        nid_library[sensor_nid] = np.append(nid_library[sensor_nid], sensor_wlak)

        #print counter
        print('counter: ' + str(nid_library[counter]))

        if nid_library[counter] == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_wlak = '_waterleak.joblib'
                filename_wlak_model = var1 + sensor_nid + var_wlak
                #load model
                model_waterleak = load(filename_wlak_model)
            except: #if there is no specified model
                #filename
                filename_wlak_model = 'model\model_waterleak.joblib'
                #load model
                model_waterleak = load(filename_wlak_model)
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_wlak_model)
                anomaly_threshVal0 = 0.0
                nid_library[counter] +=1

        elif nid_library[counter] <= batch_size:
            #mode2: Keep using initial model until the data stored in array
            nid_library[counter] += 1

        elif nid_library[counter] == (batch_size + 1):
            #mode 3: retrain the model

            #calculate the outlier fraction
            outlier = Counter(nid_library[status_nid])
            outlier_fraction = outlier['abnormal'] / len(nid_library[status_nid])
            print('outlier fraction: '+str(outlier_fraction))
            
            #Multithreading            
            thread = threading.Thread(target=train, args=(sensor_nid, outlier_fraction))
            if thread.is_alive():
                print('thread still running')
            else:
                print('thread is starting')
                thread.start()
            nid_library[counter] += 1
            thread.join()

        elif nid_library[counter] == (batch_size+2):
            #model 4: load retrain model

            #filename
            var1 = 'model\model_'
            var_wlak = '_waterleak.joblib'
            filename_wlak_model = var1 + sensor_nid + var_wlak

            #load model
            model_waterleak = load(filename_wlak_model)

            #calculate the anomaly score threshold for temperature
            anomaly_score_wlak_mean = nid_library[score_nid].mean()
            anomaly_score_wlak_std = nid_library[score_nid].std()
            anomaly_score_wlak_cal = anomaly_score_wlak_mean - (anomaly_score_wlak_std*1.5)
            
            if anomaly_score_wlak_cal <= -0.15:
                anomaly_threshVal0 = -0.15
            elif anomaly_score_wlak_cal >= 0.1:
                anomaly_threshVal0 = 0.1
            else:
                anomaly_threshVal0 = anomaly_score_wlak_cal
            
            nid_library[counter] += 1

        elif nid_library[counter] <= (batch_size + batch_size):
            #mode 5: sliding window method
            nid_library[counter] += 1

        else:
            #optimize the array size of sliding window
            nid_library[sensor_nid] = nid_library[sensor_nid][-(train_number+2*batch_size):]
            nid_library[score_nid] = nid_library[score_nid][-(train_number+2*batch_size):]
            nid_library[status_nid] = nid_library[status_nid][-(train_number+2*batch_size):]
            nid_library[counter] = (batch_size+1)

        #preprocess the data for anomaly detection
        sensor_wlak_reshape = sensor_wlak.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_wlak = model_waterleak.decision_function(sensor_wlak_reshape)

        #print(anomaly_score_wlak)
        #print(sensor_wlak[0])

        if float(sensor_wlak) == 0:
            #normal condition
            sensor_wlak_status = 'normal/no_leak'
        else:
            #abnormal condition
            sensor_wlak_status = 'abnormal/leak'

        #append value of anomaly score and sensor status
        nid_library[score_nid] = np.append(nid_library[score_nid], float(anomaly_score_wlak))
        nid_library[status_nid] = np.append(nid_library[status_nid], sensor_wlak_status)

        print('window_size: ' + str(len(nid_library[sensor_nid])))

        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['val0'] = float(sensor_wlak[0])
        changedata['result_val0'] = sensor_wlak_status
        changedata['anomaly_score_val0'] = float(anomaly_score_wlak)
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 7:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [batchsize] [train_number] [outlier_fraction]")
        exit(1)

    #IoT.own setting
    postproc_name = 'wlak'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #sliding window setting
    batch_size = int(sys.argv[4])
    train_number = int(sys.argv[5])

    #model parameter
    outlier_fraction_param = float(sys.argv[6])

    #initialize nid library
    nid_library = {} #for val0
    postprocess(url,postproc_name,post_process, username, password)
