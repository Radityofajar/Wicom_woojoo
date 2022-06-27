from pyiotown import post
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

def train(sensor_nid, outlier_fraction1, outlier_fraction2, outlier_fraction3):
    global nid_library, nid_library_2, nid_library_3
    global outlier_fraction_param

    #model setting
    estimator = 100
    samples = 500
    randstate = 42

    #outlier parameter
    if outlier_fraction1 == 0:
        outlier_fraction1 = 0.01 # 1% of contamination
    elif outlier_fraction1 >= outlier_fraction_param:
        outlier_fraction1 = outlier_fraction_param
    else:
        outlier_fraction1 = outlier_fraction1

    if outlier_fraction2 == 0:
        outlier_fraction2 = 0.01 # 1% of contamination
    elif outlier_fraction2 >= outlier_fraction_param:
        outlier_fraction2 = outlier_fraction_param
    else:
        outlier_fraction2 = outlier_fraction2
    
    if outlier_fraction3 == 0:
        outlier_fraction3 = 0.01 # 1% of contamination
    elif outlier_fraction3 >= outlier_fraction_param:
        outlier_fraction3 = outlier_fraction_param
    else:
        outlier_fraction3 = outlier_fraction3
    
    #model initialization
    model_hum = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction1)
    model_temp = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction2)
    model_door = IsolationForest(n_estimators=estimator, max_samples=samples, random_state=randstate, contamination=outlier_fraction3)

    #data preprocess
    nid_library['data'] = nid_library['data'].reshape(-1,1)
    nid_library_2['data'] = nid_library_2['data'].reshape(-1,1)
    nid_library_3['data'] = nid_library_3['data'].reshape(-1,1)

    #model training
    model_hum.fit(nid_library['data'])
    model_temp.fit(nid_library_2['data'])
    model_door.fit(nid_library_3['data'])

    #filename
    var1 = 'model\model_'
    var_hum = '_hum.joblib'
    var_temp = '_temp.joblib'
    var_door = '_door.joblib'
    filename_hum_model = var1 + sensor_nid + var_hum
    filename_temp_model = var1 + sensor_nid + var_temp
    filename_door_model = var1 + sensor_nid + var_door

    #save/overwrite model
    dump(model_hum, filename_hum_model)
    dump(model_temp, filename_temp_model)
    dump(model_door, filename_door_model)

    print('update the model')

def receive_data(rawdata): #feel free to change or update this function depends on the data input format
    raw_data = rawdata['data'] #take the data
    print(raw_data)
    # raw_data --> 'POST /api/v1.0/data HTTP/1.1\r\nContent-Type: application/json\r\nAccept: application/json\r\nContent-Length: 120\r\nToken: 079da3dd77569523551c9ddd8c8e57c4f4ee71bea5e848d68785b06545d098ac\r\n\r\n{"type": "2","nid": "WS000001FFFF123456", "data": {"dtype":"tdhd", "nid":"WS000001FFFF123456", "val0":27.8,"val1":56.2,"val2":1}}\r\n\r\n'
    test = len(raw_data.split())
    if raw_data == 'Missing':
        details = 'Data is not complete'
        print(details)
    elif test >= 11:
        #take only useful information
        msg = raw_data.split()
        datasensor = str(msg[11:])
        #datasensor --> ['{"type":', '"2","nid":', '"WS000001FFFF123456",', '"data":', '{"dtype":"tdhd",', '"nid":"WS000001FFFF123456",', '"val0":20.5,', '"val1":39.0,', '"val2":1}}']
        datasensor = datasensor.replace(']', '')
        datasensor = datasensor.replace('[', '')
        datasensor = datasensor.replace("', '", '')
        datasensor = datasensor.replace("'", '')
        pythonObj = json.loads(json.loads(json.dumps(datasensor))) #change to json format
        sensor = pythonObj['data']
        #sensor --> {"dtype":"tdhd","nid":"WS000001FFFF123456","val0":25.6,"val1":57.8,"val2":1}
    return sensor

def post_process(rawdata):
    global counter
    global model_hum, model_temp, model_door
    global threshold_hum_lower, threshold_hum_higher
    global threshold_temp_lower, threshold_temp_higher
    global door_thresh
    global batch_size, train_number
    global anomaly_threshVal0, anomaly_threshVal1
    global anomaly_threshVal0_param, anomaly_threshVal1_param
    global nid_library, nid_library_2, nid_library_3

    #receive data from sensor
    message = receive_data(rawdata=rawdata)
    #print(message)
    sensor_type = message['dtype']
    if sensor_type == 'tdhd':
        sensor_temp = np.array([message['val0']]).T #take temperature data
        sensor_hum = np.array([message['val1']]).T #take humidity data
        sensor_door = np.array([message['val2']]).T #take door sensor

        #check sensor nid
        sensor_nid = message['nid']
        if sensor_nid not in nid_library.keys(): #check whether nid is new or not
            nid_library[sensor_nid] = np.array([[]]) #make a new array for new nid (temperature)
            nid_library_2[sensor_nid] = np.array([[]]) #make a new array for new nid (humidity)
            nid_library_3[sensor_nid] = np.array([[]]) #make a new array for new nid (door)

        #input stream data to the window
        nid_library[sensor_nid] = np.append(nid_library[sensor_nid], sensor_temp) #temp
        nid_library_2[sensor_nid] = np.append(nid_library_2[sensor_nid], sensor_hum) #hum
        nid_library_3[sensor_nid] = np.append(nid_library_3[sensor_nid], sensor_door) #door
        print(nid_library[sensor_nid])

        if counter == 1:
            #mode1: using initial mode
            try: #if spesified model is already built
                #filename
                var1 = 'model\model_'
                var_hum = '_hum.joblib'
                var_temp = '_temp.joblib'
                var_door = '_door.joblib'
                filename_hum_model = var1 + sensor_nid + var_hum
                filename_temp_model = var1 + sensor_nid + var_temp
                filename_door_model = var1 + sensor_nid + var_door
                #load model
                model_temp = load(filename_temp_model)
                model_hum = load(filename_hum_model)
                model_door = load(filename_door_model)
            except: #if there is no spesificied model
                #filename
                filename_temp_model = 'model\model_temp1.joblib'
                filename_hum_model = 'model\model_hum1.joblib'
                filename_door_model = 'model\model_door.joblib'
                #load model
                model_temp = load(filename_temp_model)
                model_hum = load(filename_hum_model)
                model_door = load(filename_door_model)
                print('Take initial model')
            else:
                print('Using specified model')
            finally:
                print(filename_temp_model)
                print(filename_hum_model)
                print(filename_door_model)
                anomaly_threshVal0 = 0.0
                anomaly_threshVal1 = 0.0
                anomaly_threshVal2 = 0.0
                counter += 1

        elif counter <= batch_size:
            #mode2: Keep using initial model until the data stored in array
            counter += 1

        elif counter == (batch_size + 1):
            #mode 3: retrain the model

            #calculate the outlier fraction
            outlier1 = Counter(nid_library['anomaly_status']) #temp
            outlier2 = Counter(nid_library_2['anomaly_status']) #hum
            outlier3 = Counter(nid_library_3['anomaly_status']) #door
            outlier_fraction1 = outlier1['abnormal'] / len(nid_library['anomaly_status']) #temp
            outlier_fraction2 = outlier2['abnormal'] / len(nid_library_2['anomaly_status']) #hum
            outlier_fraction3 = outlier3['abnormal'] / len(nid_library_3['anomaly_status']) #door


            #Multithreading            
            thread = threading.Thread(target=train, args=(sensor_nid, outlier_fraction1, outlier_fraction2, outlier_fraction3))
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
            var_hum = '_hum.joblib'
            var_temp = '_temp.joblib'
            var_door = '_door.joblib'
            filename_hum_model = var1 + sensor_nid + var_hum
            filename_temp_model = var1 + sensor_nid + var_temp
            filename_door_model = var1 + sensor_nid + var_door

            #load model
            model_temp = load(filename_temp_model)
            model_hum = load(filename_hum_model)
            model_door = load(filename_door_model)
            #print('model loaded')

            #calculate the anomaly score threshold for temperature
            anomaly_score_temp_mean = nid_library['anomaly_score'].mean()
            anomaly_score_temp_std = nid_library['anomaly_score'].std()
            anomaly_score_temp_cal = anomaly_score_temp_mean - (anomaly_score_temp_std*anomaly_threshVal0_param)
            
            if anomaly_score_temp_cal <= -0.15:
                anomaly_threshVal0 = -0.15
            elif anomaly_score_temp_cal >= 0.02:
                anomaly_threshVal0 = 0.02
            else:
                anomaly_threshVal0 = anomaly_score_temp_cal

            #calculate the anomaly score threshold for humidity
            anomaly_score_hum_mean = nid_library_2['anomaly_score'].mean()
            anomaly_score_hum_std = nid_library_2['anomaly_score'].std()
            anomaly_score_hum_cal = anomaly_score_hum_mean + (anomaly_score_hum_std*anomaly_threshVal1_param)
            
            if anomaly_score_hum_cal <= -0.15:
                anomaly_threshVal1 = -0.15
            elif anomaly_score_hum_cal >= 0.02:
                anomaly_threshVal1 = 0.02
            else:
                anomaly_threshVal1 = anomaly_score_hum_cal

            #calculate the anomaly score threshold for door
            anomaly_score_door_mean = nid_library_3['anomaly_score'].mean()
            anomaly_score_door_std = nid_library_3['anomaly_score'].std()
            anomaly_score_door_cal = anomaly_score_door_mean - (anomaly_score_door_std*1.5)
            
            if anomaly_score_door_cal <= -0.15:
                anomaly_threshVal2 = -0.15
            else:
                anomaly_threshVal2 = anomaly_score_door_cal

            counter += 1

        elif counter <= (batch_size + batch_size):
            #mode 5: sliding window method
            counter += 1

        else:
            #optimize the array size of sliding window for temperature
            nid_library[sensor_nid] = nid_library[sensor_nid][-(train_number+batch_size):]
            nid_library['anomaly_score'] = nid_library['anomaly_score'][-(train_number+batch_size):]
            nid_library['anomaly_status'] = nid_library['anomaly_status'][-(train_number+batch_size):]
            #optimize the array size of sliding window for humidity
            nid_library_2[sensor_nid] = nid_library_2[sensor_nid][-(train_number+batch_size):]
            nid_library_2['anomaly_score'] = nid_library_2['anomaly_score'][-(train_number+batch_size):]
            nid_library_2['anomaly_status'] = nid_library_2['anomaly_status'][-(train_number+batch_size):]
            #optimize the array size of sliding window for door
            nid_library_3['data'] = nid_library_3['data'][-(train_number+batch_size):]
            nid_library_3['anomaly_score'] = nid_library_3['anomaly_score'][-(train_number-batch_size):]
            nid_library_3['anomaly_status'] = nid_library_3['anomaly_status'][-(train_number+batch_size):]
            counter = (batch_size+1)

        #preprocess the data for anomaly detection
        sensor_temp_reshape = sensor_temp.reshape(1,-1)
        sensor_hum_reshape = sensor_hum.reshape(1,-1)
        sensor_door_reshape = sensor_door.reshape(1,-1)

        #anomaly detection / Isoloation forest prediction
        anomaly_score_temp = model_temp.decision_function(sensor_temp_reshape)
        anomaly_score_hum = model_hum.decision_function(sensor_hum_reshape)
        anomaly_score_door = model_door.decision_function(sensor_door_reshape)

        #print(anomaly_score_temp)
        #print(sensor_temp[0])
        #print(anomaly_score_hum)
        #print(sensor_hum[0])

        #clustering between normal & abnormal
        if anomaly_score_temp > anomaly_threshVal0 and float(sensor_temp[0]) > threshold_temp_lower and float(sensor_temp[0]) < threshold_temp_higher:
            #normal condition
            sensor_temp_status = 'normal'
        else:
            #abnormal condition
            sensor_temp_status = 'abnormal'

        if anomaly_score_hum > anomaly_threshVal1 and float(sensor_hum[0]) > threshold_hum_lower and float(sensor_hum[0]) < threshold_hum_higher:
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
        changedata['val0'] = float(sensor_temp[0])
        changedata['val1'] = float(sensor_hum[0])
        changedata['val2'] = float(sensor_door[0])
        changedata['result_temp'] = sensor_temp_status
        changedata['result_hum'] = sensor_hum_status
        changedata['result_door'] = sensor_door_status
        changedata['anomaly_score_temp'] = float(anomaly_score_temp)
        changedata['anomaly_score_hum'] = float(anomaly_score_hum)
        changedata['anomaly_score_door'] = float(anomaly_score_door)
        rawdata['data'] = changedata
        print(rawdata)
        return rawdata
    else:
        print('Sensor is not supported')

if __name__ == '__main__':
    if len(sys.argv) != 15:
        print(f"Usage: {sys.argv[0]} [URL] [name] [token] [low_threshVal0] [up_threshVal0] [low_threshVal1] [up_thresVal1] [door NC/NO] [batchsize] [train_number] [outlier_fraction] [anomaly_threshVal0] [anomaly_threshVal1]")
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
    outlier_fraction_param = float(sys.argv[11])

    #clustering setting
    anomaly_threshVal0_param = float(sys.argv[12])
    anomaly_threshVal1_param = float(sys.argv[13])

    #initialize nid library (depends on the number of sensor)
    nid_library = {} #for val0
    nid_library_2 = {} #for val1
    nid_library_3 = {} #for val2
    
    postprocess(url,postproc_name,post_process, username, password)
