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

def train(sensor_nid, outlier_fraction1, outlier_fraction2):
    global nid_library, nid_library_2
    global outlier_fraction_param

    #model setting
    estimator = 100
    samples = 1000
    randstate = 42

    #outlier parameter
    if outlier_fraction1 == 0: #for fire
        outlier_fraction1 = 0.001 # 0.1% of contamination
    elif outlier_fraction1 >= outlier_fraction_param:
        outlier_fraction1 = outlier_fraction_param
    else:
        outlier_fraction1 = outlier_fraction1

    if outlier_fraction2 == 0: #for temperature
        outlier_fraction2 = 0.001 # 0.1% of contamination
    elif outlier_fraction2 >= outlier_fraction_param:
        outlier_fraction2 = outlier_fraction_param
    else:
        outlier_fraction2 = outlier_fraction2

    #model initialization
    model_fire = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction1)
    model_temp = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction2)

    #data preprocess
    data_temp = nid_library[sensor_nid].reshape(-1,1)
    data_fire = nid_library_2[sensor_nid].reshape(-1,1)

    #model training
    model_temp.fit(data_temp)
    model_fire.fit(data_fire)

    #filename
    var1 = 'model\model_'
    var_temp = '_temp.joblib'
    var_fire = '_fire.joblib'
    filename_temp_model = var1 + sensor_nid + var_temp
    filename_fire_model = var1 + sensor_nid + var_fire

    #save/overwrite model
    dump(model_temp, filename_temp_model)
    dump(model_fire, filename_fire_model)

    print('update the model')

def receive_data(rawdata):
    raw_data = rawdata['data']
    #take the data
    print(raw_data)
    # raw_data --> 'POST /api/v1.0/data HTTP/1.1\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: 120\r\nToken: 079da3dd77569523551c9ddd8c8e57c4f4ee71bea5e848d68785b06545d098ac\r\n\r\n{"type": "2","nid": "WS000001FFFF123456", "data": {"dtype":"fire", "nid":"WS000001FFFF123456", "val0":0,"val1":26.2}}\r\n\r\n'
    test = len(raw_data.split())
    if raw_data == 'Missing' or test < 11:
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        #datasensor --> ['{"type":', '"2","nid":', '"WS000001FFFF123456",', '"data":', '{"dtype":"fire",', '"nid":"WS000001FFFF123456",', '"val0":1,', '"val1":29.0}}']
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor))) #change to json format
        sensor = pythonObj['data']
        #sensor --> {"dtype":"fire","nid":"WS000001FFFF123456","val0":0,"val1":27.8}
    return sensor

def post_process(rawdata):
    global counter
    global model_temp, model_fire
    global threshold_temp_lower, threshold_temp_higher
    global batch_size, train_number
    global anomaly_threshVal1, anomaly_threshVal1_param
    global nid_library, nid_library_2

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)

    #check the sensor type
    sensor_type = message['dtype']
    if sensor_type == 'fire':
        sensor_fire = np.array([message['val0']]).T
        sensor_temp = np.array([message['val1']]).T
        
        #check sensor nid
        sensor_nid = message['nid']
        if sensor_nid not in nid_library.keys(): #check wheteher nid is new or not
            score_nid = 'score_' + sensor_nid
            status_nid = "status_" + sensor_nid
            nid_library[sensor_nid] = np.array([[]]) #make a new array for new nid (fire)
            nid_library[score_nid] = np.array([[]]) #make a new array for new nid (fire)
            nid_library[status_nid] = np.array([[]]) #make a new array for new nid (fire)
            nid_library_2[sensor_nid] = np.array([[]]) #make a new array for new nid (temperature)
            nid_library_2[score_nid] = np.array([[]]) #make a new array for new nid (temperature)
            nid_library_2[status_nid] = np.array([[]]) #make a new array for new nid (temperature)

        #input stream data to the window
        nid_library[sensor_nid] = np.append(nid_library[sensor_nid], sensor_fire) #fire
        nid_library_2[sensor_nid] = np.append(nid_library_2[sensor_nid], sensor_temp) #temp
        print(nid_library[sensor_nid])

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_fire = '_fire.joblib'
                var_temp = '_temp.joblib'
                filename_fire_model = var1 + sensor_nid + var_fire
                filename_temp_model = var1 + sensor_nid + var_temp
                #load model
                model_fire = load(filename_fire_model)
                model_temp = load(filename_temp_model)
                
            except: #if there is no specified model
                #filename
                filename_fire_model = 'model\model_fire.joblib'
                filename_temp_model = 'model\model_temp1.joblib'
                #load model
                model_fire = load(filename_fire_model)
                model_temp = load(filename_temp_model)
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_fire_model)
                print(filename_temp_model)
                anomaly_threshVal0 = 0.0
                anomaly_threshVal1 = 0.0
                counter += 1

        elif counter <= batch_size:
            #mode2: Keep using initial model until the data stored in array
            counter += 1

        elif counter == (batch_size + 1):
            #mode 3: retrain the model

            #calculate the outlier fraction
            outlier1 = Counter(nid_library[status_nid]) #fire
            outlier2 = Counter(nid_library_2[status_nid]) #temp
            outlier_fraction1 = outlier1['abnormal'] / len(nid_library[status_nid]) #fire
            outlier_fraction2 = outlier2['abnormal'] / len(nid_library_2[status_nid]) #temp
            print(outlier_fraction1)
            print(outlier_fraction2)
            #multithreading
            thread = threading.Thread(target=train, args=(sensor_nid,outlier_fraction1, outlier_fraction2,))
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
            var_fire = '_fire.joblib'
            var_temp = '_temp.joblib'
            filename_fire_model = var1 + sensor_nid + var_fire
            filename_temp_model = var1 + sensor_nid + var_temp

            #load model
            model_fire = load(filename_fire_model)
            model_temp = load(filename_temp_model)
            #print('model loaded')

            #calculate the anomaly score threshold for fire
            anomaly_score_fire_mean = nid_library[score_nid].mean()
            anomaly_score_fire_std = nid_library[score_nid].std()
            anomaly_score_fire_cal = anomaly_score_fire_mean - (anomaly_score_fire_std*1.5)
            
            if anomaly_score_fire_cal <= -0.15:
                anomaly_threshVal0 = -0.15
            elif anomaly_score_fire_cal >= 0.01:
                anomaly_threshVal0 = 0.01
            else:
                anomaly_threshVal0 = anomaly_score_fire_cal

            #calculate the anomaly score threshold for temperature
            anomaly_score_temp_mean = nid_library_2[score_nid].mean()
            anomaly_score_temp_std = nid_library_2[score_nid].std()
            anomaly_score_temp_cal = anomaly_score_temp_mean - (anomaly_score_temp_std*anomaly_threshVal1_param)
            
            if anomaly_score_temp_cal <= -0.15:
                anomaly_threshVal1 = -0.15
            elif anomaly_score_temp_cal >= 0.01:
                anomaly_threshVal1 = 0.01
            else:
                anomaly_threshVal1 = anomaly_score_temp_cal

            counter += 1

        elif counter <= (batch_size + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window for fire
            nid_library[sensor_nid] = nid_library[sensor_nid][-(train_number+2*batch_size):]
            nid_library[score_nid] = nid_library[score_nid][-(train_number+2*batch_size):]
            nid_library[status_nid] = nid_library[status_nid][-(train_number+2*batch_size):]
            #optimize the array size of sliding window for temp
            nid_library_2[sensor_nid] = nid_library_2[sensor_nid][-(train_number+2*batch_size):]
            nid_library_2[score_nid] = nid_library_2[score_nid][-(train_number+2*batch_size):]
            nid_library_2[status_nid] = nid_library_2[status_nid][-(train_number+2*batch_size):]
            counter = (batch_size+1)

        #preprocess the data for anomaly detection
        sensor_fire_reshape = sensor_fire.reshape(1,-1)
        sensor_temp_reshape = sensor_temp.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_fire = model_fire.decision_function(sensor_fire_reshape)
        anomaly_score_temp = model_temp.decision_function(sensor_temp_reshape)

        #print(anomaly_score_temp)
        #print(sensor_temp[0])
        #print(anomaly_score_fire)
        #print(sensor_fire[0])

        #clustering between normal & abnormal
        #Fire sensor
        if float(sensor_fire) == 0:
            #normal condition
            sensor_fire_status = 'normal/no_fire'
        else:
            #abnormal condition
            sensor_fire_status = 'abnormal/fire'

        #Temperature sensor
        
        if float(sensor_temp[0]) > threshold_temp_lower:
            if float(sensor_temp[0]) < threshold_temp_higher:
                if anomaly_score_temp > anomaly_threshVal1:
                    #normal condition
                    sensor_temp_status = 'normal'
                else:
                    #abnormal condition detected by isolation forest
                    sensor_temp_status = 'abnormal'
            else:
                #abnormal condition detected by thresholdAD
                sensor_temp_status = 'abnormal/too high'
        else:
            #abnormal condition detected by thresholdAD
            sensor_temp_status = 'abnormal/too low'

        #append value of anomaly score and sensor status
        nid_library[score_nid] = np.append(nid_library[score_nid],float(anomaly_score_temp))
        nid_library[status_nid] = np.append(nid_library[status_nid],sensor_temp_status)

        nid_library_2[score_nid] = np.append(nid_library_2[score_nid],float(anomaly_score_fire))
        nid_library_2[status_nid] = np.append(nid_library_2[status_nid],sensor_fire_status)
        
        #store the data in order to send it back to IoT.own
        changedata = {}
        changedata['dtype'] = message['dtype']
        changedata['nid'] = message['nid']
        changedata['val0'] = float(sensor_fire[0])
        changedata['val1'] = float(sensor_temp[0])
        changedata['result_val0'] = sensor_fire_status
        changedata['result_val1'] = sensor_temp_status
        changedata['anomaly_score_val0'] = float(anomaly_score_fire)
        changedata['anomaly_score_val1'] = float(anomaly_score_temp)
        changedata['anomaly_score_threshold_temp'] = float(anomaly_threshVal1)
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 10:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal1] [up_thresVal1] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal1]")
        exit(1)
    
    #IoT.own setting
    postproc_name = 'fire'
    url = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    #thresholding value
    threshold_temp_lower = int(sys.argv[4])
    threshold_temp_higher = int(sys.argv[5])

    #sliding window setting
    batch_size = int(sys.argv[6])
    train_number = int(sys.argv[7])

    #model parameter
    outlier_fraction_param = float(sys.argv[8])

    #clustering setting
    anomaly_threshVal1_param = float(sys.argv[9])

    #initialize nid library (depends on the number of sensor)
    nid_library = {} #for fire
    nid_library_2 = {} #for temperature
    
    postprocess(url,postproc_name,post_process, username, password)
